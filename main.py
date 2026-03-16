#!/usr/bin/env python3
"""
Script de exemplo rápido para iniciar o sistema de detecção de futebol.
Executa: python main.py
"""

import logging
import os
import re
import sys
from pathlib import Path

# Configurar display e Qt para OpenCV
if os.environ.get('DISPLAY'):
    # Se tem display disponível, usar xcb
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
else:
    # Senão, usar offscreen
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Suprimir avisos de Qt
os.environ['QT_DEBUG_PLUGINS'] = '0'

# Importar cv2 após configurar as variáveis de ambiente
import cv2
cv2.setNumThreads(1)  # Melhorar estabilidade com display

# Log em estilo log: nível, timestamp, módulo, mensagem (detecções em tempo real)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Diretório base do projeto (funciona no host e no container)
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"
VIDEOS_DIR = BASE_DIR / "videos"
RUNS_DIR = BASE_DIR / "runs"

# Adicionar src ao path
sys.path.insert(0, str(SRC_DIR))

from football_analyzer.football_detector import FootballDetector


def main():
    print("\n" + "="*60)
    print("🎯 SISTEMA DE DETECÇÃO DE FUTEBOL COM YOLOv8m")
    print("="*60)
    
    print("\nEscolha uma opção:\n")
    print("1. 📷 Usar câmera em tempo real (detector simples)")
    print("2. 🎬 Processar um arquivo de vídeo (detector simples)")
    print("3. ⚙️ Ajustar configurações")
    print("4. 🧪 Treinar modelo customizado (avançado)")
    print("5. 🧠 Treinar todos os modelos da spec")
    print("6. 🛡️ Treinar verificação (goleiro/juiz/bola/times)")
    print("7. ℹ️ Sair\n")
    
    choice = input("Digite sua escolha (1-7): ").strip()
    
    if choice == "1":
        # Usar câmera com configurações otimizadas
        print("\n⏳ Inicializando câmera com detecção otimizada...")
        print("   Modelo: yolov8m.pt")
        print("   Confiança: 0.65 (rigor anti-dedução da spec)")
        print("   Filtros: Ativados para evitar erros de classificação")
        
        # Perguntar se quer gravar
        gravar = input("\nDeseja gravar o vídeo? (s/n): ").strip().lower()
        output_path = None
        
        if gravar == "s":
            VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = str(VIDEOS_DIR / "captura_camera.mp4")
            print(f"Vídeo será salvo em: {output_path}")
        
        detector = FootballDetector(
            conf_threshold=0.65,
        )
        
        print("\n📷 Câmera iniciada!")
        print("Pressione 'Q' para parar\n")
        
        detector.process_camera(
            camera_id=0,
            output_path=output_path,
            display=True
        )
        
    elif choice == "2":
        # Processar vídeo
        video_path = input("\nDigite o caminho do vídeo: ").strip()
        
        if not Path(video_path).exists():
            print(f"❌ Arquivo não encontrado: {video_path}")
            return
        
        # Perguntar se quer gravar o resultado
        gravar = input("Deseja gravar o resultado? (s/n): ").strip().lower()
        output_path = None
        
        if gravar == "s":
            nome_saida = input("Nome do arquivo de saída (padrão: resultado.mp4): ").strip() or "resultado.mp4"
            VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = str(VIDEOS_DIR / nome_saida)
            print(f"Vídeo será salvo em: {output_path}")

        # Para vídeo externo, perguntar explicitamente sobre interface gráfica
        usar_gui = input("Deseja exibir janela em tempo real? (s/n): ").strip().lower()
        display_video = usar_gui == "s"
        
        print("\n⏳ Carregando detector otimizado...")

        detector = FootballDetector(
            conf_threshold=0.65,
        )
        
        print("🎬 Processando vídeo com filtros de detecção...\n")
        
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display=display_video
        )
    
    elif choice == "3":
        # Ajustar configurações
        print("\n⚙️  CONFIGURAÇÕES PERSONALIZADAS")
        print("="*60)
        
        model = input("\nModelo (yolov8x/s/m/l, padrão: yolov11m): ").strip() or "yolov11m"
        conf_str = input("Confiança (0.65-1.0, padrão: 0.65): ").strip() or "0.65"
        
        try:
            conf = float(conf_str)
            if conf < 0 or conf > 1:
                raise ValueError()
        except:
            print("❌ Valor inválido")
            return
        
        print(f"\n⚙️  Usando: {model}.pt com confiança {conf}")
        print("Pressione 'Q' para parar\n")
        
        detector = FootballDetector(
            conf_threshold=conf
        )
        
        detector.process_camera(camera_id=0, display=True)
    
    elif choice == "4":
        # Treinar modelo customizado
        from football_analyzer.train_custom_model import train_football_model
        import torch
        
        dataset_dir_input = input("\nCaminho da pasta do dataset (contendo data.yaml): ").strip()
        dataset_dir = Path(dataset_dir_input).expanduser().resolve()

        if not dataset_dir.exists() or not dataset_dir.is_dir():
            print(f"❌ Pasta inválida: {dataset_dir_input}")
            return

        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists() or not data_yaml.is_file():
            print(f"❌ Arquivo data.yaml não encontrado em: {dataset_dir}")
            return
        
        epochs_str = input("Número de épocas (padrão: 6): ").strip() or "6"
        patience_str = input("Early stopping (paciência, padrão: 20): ").strip() or "20"
        
        try:
            epochs = int(epochs_str)
            if epochs <= 0:
                raise ValueError()
        except:
            print("❌ Valor inválido")
            return

        try:
            patience = int(patience_str)
            if patience <= 0:
                raise ValueError()
        except:
            print("❌ Valor inválido para early stopping")
            return
        
        if not torch.cuda.is_available():
            print("❌ Não há GPU disponível")
            return

        device = 0
        dataset_root_name = dataset_dir.name
        run_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", dataset_root_name).strip("_").lower()
        if not run_name:
            run_name = "football_detector"
        print(f"Treinando em device={device}")
        print(f"Nome do experimento: {run_name}")
        print(f"data.yaml: {data_yaml}")
        train_football_model(
            str(data_yaml),
            epochs,
            imgsz=640,
            batch_size=2,
            device=device,
            name=run_name,
            early_stopping_patience=patience,
        )

    elif choice == "5":
        from football_analyzer.train_custom_model import train_spec_models
        import torch

        epochs_str = input("Número de épocas para todos os modelos (padrão: 50): ").strip() or "50"
        patience_str = input("Early stopping global (padrão: usar perfil por modelo): ").strip()

        try:
            epochs = int(epochs_str)
            if epochs <= 0:
                raise ValueError()
        except:
            print("❌ Valor inválido")
            return

        global_patience = None
        if patience_str:
            try:
                global_patience = int(patience_str)
                if global_patience <= 0:
                    raise ValueError()
            except:
                print("❌ Valor inválido para early stopping")
                return

        if not torch.cuda.is_available():
            print("❌ Não há GPU disponível para treinar todos os modelos")
            return

        print("\n🧠 Iniciando treino completo da spec...")
        print(f"Épocas: {epochs}")
        outputs = train_spec_models(
            epochs=epochs,
            early_stopping_patience=global_patience,
        )
        print("\n✅ Treino da spec concluído.")
        for model_name, engine_path in outputs.items():
            status = engine_path if engine_path else "não exportado"
            print(f"- {model_name}: {status}")
    
    elif choice == "6":
        from football_analyzer.train_custom_model import train_player_roles_verification
        import torch

        if not torch.cuda.is_available():
            print("❌ Não há GPU disponível para treino de verificação.")
            return

        epochs_str = input("Número de épocas (padrão: 120): ").strip() or "120"
        patience_str = input("Early stopping (padrão: 40): ").strip() or "40"
        try:
            epochs = int(epochs_str)
            patience = int(patience_str)
            if epochs <= 0 or patience <= 0:
                raise ValueError()
        except:
            print("❌ Valores inválidos.")
            return

        print("\n🛡️ Iniciando treino de verificação de papéis...")
        output = train_player_roles_verification(
            epochs=epochs,
            early_stopping_patience=patience,
        )
        status = output if output else "não exportado"
        print(f"✅ Treino concluído: {status}")

    else:
        print("👋 Saindo...")
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


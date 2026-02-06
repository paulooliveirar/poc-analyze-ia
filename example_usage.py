#!/usr/bin/env python3
"""
Script de exemplo rápido para iniciar o sistema de detecção de futebol.
Executa: python example_usage.py
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from football_detector import FootballDetector


def main():
    print("\n" + "="*60)
    print("🎯 SISTEMA DE DETECÇÃO DE FUTEBOL COM YOLOv11")
    print("="*60)
    
    print("\nEscolha uma opção:\n")
    print("1. 📷 Usar câmera em tempo real")
    print("2. 🎬 Processar um arquivo de vídeo")
    print("3. ⚙️  Ajustar configurações")
    print("4. ℹ️  Sair\n")
    
    choice = input("Digite sua escolha (1-4): ").strip()
    
    if choice == "1":
        # Usar câmera com configurações otimizadas
        print("\n⏳ Inicializando câmera com detecção otimizada...")
        print("   Modelo: yolo26m.pt (melhor precisão)")
        print("   Confiança: 0.60 (menos falsos positivos)")
        print("   Filtros: Ativados para evitar erros de classificação")
        
        # Perguntar se quer gravar
        gravar = input("\nDeseja gravar o vídeo? (s/n): ").strip().lower()
        output_path = None
        
        if gravar == "s":
            output_path = "/home/paulo/Documentos/novo_projeto/poc-analyze-ia/videos/captura_camera.mp4"
            print(f"Vídeo será salvo em: {output_path}")
        
        detector = FootballDetector(
            model_path="yolo26m.pt",
            conf_threshold=0.60
        )
        
        print("\n📷 Câmera iniciada!")
        print("Pressione 'Q' para parar\n")
        
        detector.process_camera(
            camera_id=0,
            output_path=output_path
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
            output_path = f"/home/paulo/Documentos/novo_projeto/poc-analyze-ia/videos/{nome_saida}"
            print(f"Vídeo será salvo em: {output_path}")
        
        print("\n⏳ Carregando detector otimizado...")
        detector = FootballDetector(
            model_path="yolo26m.pt",
            conf_threshold=0.60
        )
        
        print("🎬 Processando vídeo com filtros de detecção...\n")
        
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display=True
        )
    
    elif choice == "3":
        # Ajustar configurações
        print("\n⚙️  CONFIGURAÇÕES PERSONALIZADAS")
        print("="*60)
        
        model = input("\nModelo (yolov8x/s/m/l, padrão: yolov11m): ").strip() or "yolov11m"
        conf_str = input("Confiança (0.0-1.0, padrão: 0.60): ").strip() or "0.60"
        
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
            model_path=f"{model}.pt",
            conf_threshold=conf
        )
        
        detector.process_camera(camera_id=0)
        
    else:
        print("👋 Saindo...")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


#!/usr/bin/env python3
"""
Script para usar o melhor modelo YOLOv8 para detecção de futebol.

Recomendações:
1. Use yolo26m.pt (médio) ao invés de nano para melhor precisão
2. Configure com limiar mais alto para evitar falsos positivos
3. Para máxima precisão, treine um modelo customizado
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from football_detector import FootballDetector


def usar_modelo_otimizado():
    """Usa configurações otimizadas para detecção precisa de futebol"""
    
    print("⚙️  Inicializando detector otimizado...")
    print("   Modelo: yolo26m.pt (melhor precisão)")
    print("   Confiança: 0.60 (menos falsos positivos)")
    print()
    
    detector = FootballDetector(
        model_path="yolo26m.pt",      # Usar medium ao invés de nano
        conf_threshold=0.60            # Limiar mais alto
    )
    
    print("✓ Detector inicializado")
    print()
    print("Escolha uma opção:")
    print("1. Câmera em tempo real")
    print("2. Processar vídeo")
    print()
    
    choice = input("Digite sua escolha (1-2): ").strip()
    
    if choice == "1":
        print("\n📷 Iniciando câmera...")
        print("   Pressione 'Q' para parar\n")
        detector.process_camera(camera_id=0)
        
    elif choice == "2":
        video_path = input("\nCaminho do vídeo: ").strip()
        
        if not Path(video_path).exists():
            print(f"❌ Arquivo não encontrado: {video_path}")
            return
        
        print("\n🎬 Processando vídeo...")
        detector.process_video(
            video_path=video_path,
            output_path=Path(video_path).stem + "_detectado.mp4",
            display=True
        )
    else:
        print("Opção inválida")


def ajustar_confianca():
    """Permite ajustar o limiar de confiança"""
    
    conf = float(input("Digite o limiar de confiança (0.0-1.0, padrão 0.60): ") or "0.60")
    
    if conf < 0 or conf > 1:
        print("❌ Valor inválido")
        return
    
    print(f"\n⚙️  Usando confiança: {conf}")
    
    detector = FootballDetector(
        model_path="yolo26m.pt",
        conf_threshold=conf
    )
    
    detector.process_camera(camera_id=0)


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 DETECTOR DE FUTEBOL - VERSÃO OTIMIZADA")
    print("=" * 60)
    print()
    
    menu = input("Menu:\n1. Usar otimizado\n2. Ajustar confiança\nEscolha (1-2): ").strip()
    
    if menu == "1":
        usar_modelo_otimizado()
    elif menu == "2":
        ajustar_confianca()
    else:
        print("Opção inválida")

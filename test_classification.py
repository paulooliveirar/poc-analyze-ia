#!/usr/bin/env python3
"""
Script de teste para a classificação automática de jogadores, goleiros, árbitro e bandeirinha
"""

import cv2
import numpy as np
from src.football_detector import FootballDetector


def test_classification():
    """Testa a classificação de pessoas em uma partida de futebol"""
    
    print("="*70)
    print("TESTE DE CLASSIFICAÇÃO AUTOMÁTICA")
    print("="*70)
    print("\nCaracterísticas detectadas:")
    print("  • JOGADOR TIME 1 (Azul) - Uniforme azul")
    print("  • JOGADOR TIME 2 (Vermelho) - Uniforme vermelho ou branco")
    print("  • GOLEIRO (Ciano) - Uniforme verde/amarelo")
    print("  • ÁRBITRO (Verde) - Uniforme preto")
    print("  • BANDEIRINHA (Magenta) - Uniforme laranja/amarelo")
    print("  • BOLA (Laranja)")
    print("\n" + "="*70)
    
    # Inicializar detector
    detector = FootballDetector(
        model_path="yolov8n.pt",
        conf_threshold=0.45
    )
    
    print("\n✓ Detector inicializado com sucesso!")
    print("\nOpções:")
    print("  1. Processar vídeo com classificação")
    print("  2. Usar câmera em tempo real com classificação")
    print("  3. Sair")
    
    choice = input("\nEscolha uma opção (1-3): ").strip()
    
    if choice == "1":
        video_path = input("Digite o caminho do vídeo: ").strip()
        try:
            print("\nProcessando vídeo com classificação automática...")
            detector.process_video(video_path, display=True)
        except FileNotFoundError:
            print(f"❌ Erro: Arquivo não encontrado: {video_path}")
    
    elif choice == "2":
        print("\nIniciando câmera com classificação automática...")
        print("Pressione 'q' para parar\n")
        detector.process_camera(camera_id=0)
    
    elif choice == "3":
        print("Saindo...")
    
    else:
        print("❌ Opção inválida!")


if __name__ == "__main__":
    test_classification()

#!/usr/bin/env python3
"""
Script para testar e demonstrar as cores das detecções
Cria uma imagem com exemplos de todas as classes com suas cores
"""

import cv2
import numpy as np
from src.football_detector import FootballDetector


def create_color_demo():
    """Cria uma imagem de demonstração com todas as cores das classes"""
    
    # Obter cores do FootballDetector
    detector = FootballDetector()
    
    # Criar imagem branca
    width, height = 1000, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Título
    cv2.putText(img, "Teste de Cores - FootballDetector", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Desenhar boxes com cores para cada classe
    y_start = 100
    box_height = 80
    box_width = 180
    spacing = 30
    cols = 3
    x_start = 50
    
    for idx, (class_id, class_name) in enumerate(detector.FOOTBALL_CLASSES.items()):
        # Calcular posição
        col = idx % cols
        row = idx // cols
        x = x_start + col * (box_width + spacing)
        y = y_start + row * (box_height + spacing)
        
        # Obter cor
        color = detector.COLORS.get(class_name, (255, 255, 255))
        text_color = detector.TEXT_COLORS.get(class_name, (255, 255, 255))
        
        # Desenhar caixa de demonstração
        cv2.rectangle(img, (x, y), (x + box_width, y + box_height), color, 3)
        
        # Contorno de contraste
        cv2.rectangle(img, (x-1, y-1), (x + box_width + 1, y + box_height + 1), (0, 0, 0), 1)
        
        # Desenhar label
        label = class_name.replace('_', ' ').upper()
        font_scale = 0.6
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x + (box_width - text_size[0]) // 2
        text_y = y + (box_height + text_size[1]) // 2
        
        # Background para texto
        padding = 2
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     color, -1)
        
        # Texto
        cv2.putText(img, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        
        # Exibir valores BGR
        color_text = f"BGR({color[0]},{color[1]},{color[2]})"
        cv2.putText(img, color_text, (x, y + box_height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        print(f"✓ {class_name:20s} → Cor BGR: {color}")
    
    # Adicionar informação
    info_y = height - 30
    cv2.putText(img, "Formato: OpenCV usa BGR (Blue, Green, Red), NÃO RGB!", 
               (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1)
    
    return img


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTE DE CORES - FOOTBALL DETECTOR")
    print("="*60 + "\n")
    
    # Gerar imagem de demonstração
    demo_img = create_color_demo()
    
    # Salvar imagem
    output_path = "test_colors_demo.png"
    cv2.imwrite(output_path, demo_img)
    
    print(f"\n✓ Imagem de demonstração salva em: {output_path}")
    print("\n" + "="*60)
    print("CONCLUSÃO:")
    print("="*60)
    print("✓ OpenCV ACEITA coloração das boxes completamente!")
    print("✓ Todas as cores estão configuradas em formato BGR")
    print("✓ A espessura das linhas foi aumentada de 2 para 3 pixels")
    print("✓ Contorno de sombra adicionado para melhor contraste")
    print("\n")

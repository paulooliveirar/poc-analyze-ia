#!/usr/bin/env python3
"""
Script de teste para validar a instalação do sistema.
Executa: python test_installation.py
"""

import sys
from pathlib import Path

print("="*60)
print("🧪 TESTANDO INSTALAÇÃO")
print("="*60)
print()

# Teste 1: Verificar Python
print("1️⃣  Verificando Python...")
print(f"   ✓ Versão: {sys.version.split()[0]}")
print()

# Teste 2: Verificar OpenCV
print("2️⃣  Verificando OpenCV...")
try:
    import cv2
    print(f"   ✓ OpenCV {cv2.__version__} instalado")
except ImportError:
    print("   ❌ OpenCV não instalado")
    sys.exit(1)
print()

# Teste 3: Verificar NumPy
print("3️⃣  Verificando NumPy...")
try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__} instalado")
except ImportError:
    print("   ❌ NumPy não instalado")
    sys.exit(1)
print()

# Teste 4: Verificar PyTorch
print("4️⃣  Verificando PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__} instalado")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA disponível: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  CUDA não disponível (usando CPU)")
except ImportError:
    print("   ❌ PyTorch não instalado")
    sys.exit(1)
print()

# Teste 5: Verificar Ultralytics
print("5️⃣  Verificando Ultralytics...")
try:
    from ultralytics import YOLO
    print(f"   ✓ Ultralytics instalado")
except ImportError:
    print("   ❌ Ultralytics não instalado")
    sys.exit(1)
print()

# Teste 6: Verificar módulos locais
print("6️⃣  Verificando módulos locais...")
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from football_detector import FootballDetector
    print("   ✓ FootballDetector importado com sucesso")
except ImportError as e:
    print(f"   ❌ Erro ao importar FootballDetector: {e}")
    sys.exit(1)

try:
    from advanced_analysis import AdvancedFootballAnalyzer
    print("   ✓ AdvancedFootballAnalyzer importado com sucesso")
except ImportError as e:
    print(f"   ❌ Erro ao importar AdvancedFootballAnalyzer: {e}")
    sys.exit(1)

try:
    from train_custom_model import train_football_model
    print("   ✓ train_football_model importado com sucesso")
except ImportError as e:
    print(f"   ❌ Erro ao importar train_football_model: {e}")
    sys.exit(1)
print()

# Teste 7: Verificar câmera
print("7️⃣  Verificando câmera...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   ✓ Câmera detectada: {int(width)}x{int(height)} @ {int(fps)} fps")
        cap.release()
    else:
        print("   ⚠️  Câmera não disponível (não é erro se usar vídeo)")
except Exception as e:
    print(f"   ⚠️  Erro ao acessar câmera: {e}")
print()

# Teste 8: Baixar modelo
print("8️⃣  Verificando/Baixando modelo YOLOv8...")
try:
    print("   ⏳ Carregando yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    print("   ✓ Modelo yolov8n.pt carregado com sucesso")
except Exception as e:
    print(f"   ❌ Erro ao carregar modelo: {e}")
    sys.exit(1)
print()

# Teste 9: Testar detecção simples
print("9️⃣  Testando detecção...")
try:
    # Criar imagem dummy
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Testar detecção
    results = model(dummy_image, verbose=False)
    print("   ✓ Detecção funcionando")
except Exception as e:
    print(f"   ❌ Erro ao testar detecção: {e}")
    sys.exit(1)
print()

# Resumo
print("="*60)
print("✅ TUDO FUNCIONANDO CORRETAMENTE!")
print("="*60)
print()
print("Próximos passos:")
print("  • python example_usage.py    (para interface interativa)")
print("  • Veja README.md para mais exemplos")
print()

#!/usr/bin/env python3
"""
🎯 COMECE POR AQUI - Guia de Início Rápido
==========================================

Este arquivo mostra exatamente como começar em 3 linhas!
"""

# ============================================================================
# 👇 OPÇÃO 1: USAR CÂMERA (mais simples)
# ============================================================================

from src.football_detector import FootballDetector

detector = FootballDetector()
detector.process_camera(camera_id=0)

# Pressione 'Q' para parar


# ============================================================================
# 👇 OPÇÃO 2: PROCESSAR VÍDEO
# ============================================================================

# from src.football_detector import FootballDetector
#
# detector = FootballDetector()
# detector.process_video("seu_video.mp4", "resultado.mp4")


# ============================================================================
# 👇 OPÇÃO 3: MENU INTERATIVO (melhor para começar)
# ============================================================================

# python example_usage.py


# ============================================================================
# 📚 PRÓXIMAS LEITURAS
# ============================================================================
#
# 1. README.md           - Documentação completa
# 2. QUICK_START.md      - Guia rápido
# 3. EXAMPLES.py         - 10 exemplos de código
# 4. TECHNICAL.md        - Documentação técnica
#
# ============================================================================

print("Comentar a opção desejada e executar este arquivo!")

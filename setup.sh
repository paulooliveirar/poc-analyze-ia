#!/bin/bash

# Script de instalação para o sistema de detecção de futebol
# Uso: bash setup.sh

echo "========================================"
echo "🎯 Setup - Detecção de Futebol com YOLOv8"
echo "========================================"
echo ""

# Verificar Python
echo "✓ Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não está instalado"
    exit 1
fi

python_version=$(python3 --version | awk '{print $2}')
echo "  Python $python_version encontrado"
echo ""

# Criar ambiente virtual
echo "✓ Criando ambiente virtual..."
if [ -d "venv" ]; then
    echo "  Ambiente virtual já existe"
else
    python3 -m venv venv
    echo "  Ambiente virtual criado"
fi
echo ""

# Ativar ambiente
echo "✓ Ativando ambiente virtual..."
source venv/bin/activate
echo "  Ambiente ativado"
echo ""

# Instalar dependências
echo "✓ Instalando dependências..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt

echo ""
echo "✓ Instalação concluída!"
echo ""
echo "========================================"
echo "🚀 Próximos passos:"
echo "========================================"
echo ""
echo "1. Ativar o ambiente:"
echo "   source venv/bin/activate"
echo ""
echo "2. Executar o exemplo:"
echo "   python example_usage.py"
echo ""
echo "3. Ou importar no seu script:"
echo "   from src.football_detector import FootballDetector"
echo ""
echo "Para mais informações, veja README.md"
echo ""

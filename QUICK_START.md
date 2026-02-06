# 🚀 Guia Rápido - Sistema de Detecção de Futebol

## ⚡ Start Rápido (5 minutos)

### 1. Instalação
```bash
# Clonar/navegar até o projeto
cd poc-analyze-ia

# Executar setup
bash setup.sh

# Ativar ambiente
source venv/bin/activate
```

### 2. Testar Instalação
```bash
python test_installation.py
```

Deve retornar: ✅ TUDO FUNCIONANDO CORRETAMENTE!

### 3. Usar o Sistema

**Opção A: Interface Interativa**
```bash
python example_usage.py
```

**Opção B: Câmera em Tempo Real (Python)**
```python
from src.football_detector import FootballDetector

detector = FootballDetector()
detector.process_camera(camera_id=0)
```

**Opção C: Processar Vídeo (Python)**
```python
from src.football_detector import FootballDetector

detector = FootballDetector()
detector.process_video(
    video_path="seu_video.mp4",
    output_path="resultado.mp4"
)
```

---

## 📊 Comparação de Modelos

| Modelo | Tamanho | CPU FPS | GPU FPS | Precisão | Recomendado |
|--------|---------|---------|---------|----------|-------------|
| yolov8n | 3.2 MB  | 5 fps   | 250 fps | ⭐⭐⭐    | **CPU/Rápido** |
| yolov8s | 11.2 MB | 2 fps   | 150 fps | ⭐⭐⭐⭐  | GPU |
| yolov8x | 49.7 MB | 1 fps   | 80 fps  | ⭐⭐⭐⭐⭐ | **Melhor** |

---

## 🎨 Customização Rápida

### Alterar Modelo
```python
detector = FootballDetector(model_path="yolo26m.pt")
```

### Ajustar Sensibilidade
```python
detector = FootballDetector(conf_threshold=0.3)  # Mais sensível
detector = FootballDetector(conf_threshold=0.7)  # Menos sensível
```

### Mudar Cores
Editar em [src/football_detector.py](src/football_detector.py#L20):
```python
COLORS = {
    "bola": (0, 165, 255),  # Laranja
    # Adicionar outras cores...
}
```

---

## 🐛 Problemas Comuns

### Erro: "No module named 'ultralytics'"
```bash
pip install ultralytics --upgrade
```

### Câmera não aparece
```bash
# Tente outro ID de câmera
detector.process_camera(camera_id=1)
```

### Muito lento na CPU?
```bash
# Use modelo nano
detector = FootballDetector(model_path="yolov8n.pt")

# Ou use GPU se disponível
# Ver: https://pytorch.org/get-started/locally/
```

---

## 📁 Estrutura

```
poc-analyze-ia/
├── src/                           # Código principal
│   ├── football_detector.py       # 🎯 Detector principal
│   ├── advanced_analysis.py       # Análise com rastreamento
│   └── train_custom_model.py      # Treinar modelo próprio
├── example_usage.py               # 📝 Exemplo interativo
├── test_installation.py           # 🧪 Verificar instalação
├── config.py                      # ⚙️  Configurações
├── requirements.txt               # 📦 Dependências
├── setup.sh                       # 🚀 Instalação automática
├── QUICK_START.md                 # Este arquivo
└── README.md                      # Documentação completa
```

---

## 💡 Exemplos Úteis

### Salvar Vídeo Processado
```python
detector = FootballDetector()
detector.process_video(
    video_path="entrada.mp4",
    output_path="saida.mp4"
)
```

### Análise Avançada (com rastreamento)
```python
from src.advanced_analysis import AdvancedFootballAnalyzer

analyzer = AdvancedFootballAnalyzer()
analyzer.process_video_advanced("video.mp4", "analise.mp4")
```

### Usar Configurações Customizadas
```python
from config import *

detector = FootballDetector(
    model_path=MODEL,
    conf_threshold=CONFIDENCE_THRESHOLD
)
```

---

## 🎓 Próximos Passos

1. **Básico** ✅ Você já está aqui! Testar com câmera/vídeo
2. **Intermediário** 🔧 Customizar cores, modelos, parâmetros
3. **Avançado** 🚀 Treinar modelo com seus dados (ver README.md)

---

## 📞 Ajuda

**Erro de instalação?**
→ Rode `python test_installation.py`

**Dúvida sobre uso?**
→ Veja os exemplos em [README.md](README.md)

**Problema com GPU?**
→ Instale PyTorch corretamente: https://pytorch.org/get-started/locally/

---

## ⌨️ Atalhos

| Ação | Comando |
|------|---------|
| Parar execução | Pressione `Q` |
| Ativar ambiente | `source venv/bin/activate` |
| Desativar ambiente | `deactivate` |
| Testar instalação | `python test_installation.py` |
| Interface interativa | `python example_usage.py` |

---

## 🔗 Links Úteis

- [Documentação Ultralytics](https://docs.ultralytics.com/)
- [Roboflow (datasets)](https://roboflow.com/)
- [OpenCV Docs](https://docs.opencv.org/)

---

**Desenvolvido com ❤️ para análise de futebol**

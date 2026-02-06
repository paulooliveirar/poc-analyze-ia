# Sistema de Detecção de Futebol com YOLOv8 🎯⚽

Um sistema completo para análise em tempo real de partidas de futebol usando inteligência artificial. Detecta automaticamente jogadores, bola, árbitro e outros elementos do jogo.

## 🎯 Características

- ✅ **Detecção em Tempo Real**: Câmera ou vídeo pré-gravado
- ✅ **Múltiplas Classes**: Detecta bola, jogadores, árbitro, bandeirinha
- ✅ **Rastreamento**: Segue a movimentação da bola e jogadores
- ✅ **Visualização Avançada**: Rastros, heatmaps e overlay de campo
- ✅ **Exportação**: Salva vídeos processados em MP4
- ✅ **Model Customizado**: Possibilidade de treinar com seu próprio dataset
- ✅ **Performance**: Suporta GPU para processamento rápido

## 📋 Pré-requisitos

- Python 3.8+
- CUDA 11.8+ (opcional, para GPU)
- Webcam ou arquivo de vídeo

## 🚀 Instalação

### 1. Clone ou copie o repositório
```bash
cd /home/paulo/Documentos/novo_projeto/poc-analyze-ia
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

**Nota sobre GPU**: Se tiver CUDA instalado:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Estrutura do Projeto

```
poc-analyze-ia/
├── src/
│   ├── football_detector.py       # Detector principal
│   ├── train_custom_model.py      # Script de treinamento
│   └── advanced_analysis.py       # Análise avançada
├── models/                         # Modelos salvos
├── videos/                         # Vídeos de entrada/saída
├── requirements.txt               # Dependências
└── README.md                      # Este arquivo
```

## 🎮 Como Usar

### Opção 1: Usar Câmera em Tempo Real

```bash
python src/football_detector.py
```

O sistema irá:
- Acessar sua câmera padrão
- Detectar elementos em tempo real
- Exibir labels com confiança
- Salvar vídeo processado (opcional)

Pressione `Q` para parar.

### Opção 2: Processar um Vídeo

```python
from src.football_detector import FootballDetector

detector = FootballDetector(
    model_path="yolov8n.pt",  # nano, small, medium, large
    conf_threshold=0.45
)

detector.process_video(
    video_path="/caminho/seu_video.mp4",
    output_path="/caminho/saida.mp4",
    display=True
)
```

### Opção 3: Análise Avançada com Rastreamento

```python
from src.advanced_analysis import AdvancedFootballAnalyzer

analyzer = AdvancedFootballAnalyzer()
analyzer.process_video_advanced(
    video_path="/seu_video.mp4",
    output_path="/saida.mp4"
)
```

Inclui:
- Rastro da bola
- Overlay de campo
- Heatmaps de movimento

## 🎓 Treinar Modelo Customizado

### Passo 1: Preparar Dataset

1. Acesse [Roboflow](https://roboflow.com/)
2. Crie um projeto para "Football Detection"
3. Faça upload de imagens de partidas
4. Anote os objetos:
   - `bola`
   - `jogador_time_1`
   - `jogador_time_2`
   - `arbitro`
   - `goleiro`
   - `bandeirinha`

5. Exporte em formato **YOLOv8**

### Passo 2: Estrutura do Dataset

```
seu_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Arquivo `data.yaml`:
```yaml
path: /caminho/seu_dataset
train: images/train
val: images/val
test: images/test

nc: 6
names:
  - bola
  - jogador_time_1
  - jogador_time_2
  - arbitro
  - goleiro
  - bandeirinha
```

### Passo 3: Treinar o Modelo

```python
from src.train_custom_model import train_football_model

train_football_model(
    dataset_path="/caminho/seu_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch_size=16
)
```

O modelo será salvo em: `runs/detect/football_detector/weights/best.pt`

### Usar o Modelo Treinado

```python
detector = FootballDetector(
    model_path="runs/detect/football_detector/weights/best.pt"
)
```

## 📊 Modelos YOLOv8 Disponíveis

| Modelo | Tamanho | Velocidade | Precisão | Uso Recomendado |
|--------|---------|-----------|----------|-----------------|
| yolov8n | 3.2 MB  | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Tempo real (CPU) |
| yolov8s | 11.2 MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Tempo real (GPU) |
| yolov8x | 49.7 MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Melhor precisão |
| yolov8l | 95.7 MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Alta precisão |

## 🎨 Personalização

### Alterar Cores de Detecção

Em `football_detector.py`, modificar `COLORS`:

```python
COLORS = {
    "bola": (0, 165, 255),          # Laranja
    "jogador_time_1": (255, 0, 0),  # Azul
    "jogador_time_2": (0, 0, 255),  # Vermelho
    # Adicione suas cores aqui
}
```

### Ajustar Limiar de Confiança

```python
detector = FootballDetector(conf_threshold=0.5)  # 0-1
```

- Valores altos = menos detecções, mas mais confiáveis
- Valores baixos = mais detecções, mas podem incluir falsos positivos

## ⚙️ Parâmetros Avançados

### Detecção
```python
detector.model(frame, 
    conf=0.45,      # Limiar de confiança
    iou=0.45,       # Limiar IoU para NMS
    max_det=300     # Máximo de detecções
)
```

### Treinamento
```python
model.train(
    epochs=100,
    batch=16,
    patience=20,     # Early stopping
    device=0,        # GPU ID
    lr0=0.01,        # Taxa de aprendizado inicial
    optimizer='SGD'  # SGD, Adam, ...
)
```

## 🔧 Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install ultralytics --upgrade
```

### Câmera não funciona
```bash
# Verificar câmeras disponíveis
v4l2-ctl --list-devices  # Linux

# Tente usar camera_id=1 ou 2
detector.process_camera(camera_id=1)
```

### CUDA não detectado
```bash
python -c "import torch; print(torch.cuda.is_available())"

# Se False, instale a versão CUDA correta
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Desempenho lento
- Use `yolov8n.pt` ao invés de modelos maiores
- Reduza `conf_threshold`
- Diminua resolução da câmera
- Use GPU (CUDA)

## 📈 Benchmarks

Tempo de processamento por frame (em fps):

| Hardware | yolov8n | yolov8s | yolov8x |
|----------|---------|---------|---------|
| CPU (i7) | 5 fps | 2 fps | 1 fps |
| GPU (RTX 3080) | 250+ fps | 150+ fps | 80+ fps |

## 🤝 Datasets Recomendados

1. **Roboflow Sports Datasets**
   - https://universe.roboflow.com/search?q=football

2. **Soccer/Football Open Data**
   - SoccerNet
   - TrackerNet

3. **Seus Próprios Dados**
   - Filme partidas locais
   - Anote no Roboflow
   - Treine modelo customizado

## 📝 Exemplos de Uso

### Exemplo 1: Análise Simples
```python
from src.football_detector import FootballDetector

detector = FootballDetector()
detector.process_video("meu_video.mp4", "saida.mp4")
```

### Exemplo 2: Tempo Real com Câmera
```python
detector = FootballDetector(conf_threshold=0.4)
detector.process_camera(camera_id=0, output_path="captura.mp4")
```

### Exemplo 3: Análise Avançada
```python
from src.advanced_analysis import AdvancedFootballAnalyzer

analyzer = AdvancedFootballAnalyzer()
analyzer.process_video_advanced("video.mp4", "analise.mp4")
```

## 🚀 Próximas Melhorias

- [ ] Estatísticas do jogo (passes, chutes, posição)
- [ ] Identificação de times por cor
- [ ] Análise tática
- [ ] Interface web
- [ ] API REST
- [ ] Integração com sistemas de broadcast

## 📄 Licença

Este projeto usa:
- **YOLOv8**: AGPL-3.0 (Ultralytics)
- **OpenCV**: Apache 2.0
- **PyTorch**: BSD

Use em conformidade com as licenças.

## 🤔 FAQ

**P: Posso usar este sistema para análise profissional?**
R: Sim, mas você pode querer treinar um modelo customizado com seus dados.

**P: Funciona offline?**
R: Sim, completamente offline após baixar o modelo.

**P: Qual a resolução máxima recomendada?**
R: 1920x1080 @ 30fps é ideal para tempo real. Resoluções maiores reduzem FPS.

**P: Como melhorar a precisão?**
R: Treine um modelo customizado com seus dados específicos.

## 📞 Suporte

Para problemas:
1. Verifique as dependências: `pip list`
2. Teste com `yolov8n.pt`
3. Verifique os logs de erro
4. Consulte a documentação: https://docs.ultralytics.com/

## 🎬 Referências

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow Vision](https://roboflow.com/)
- [OpenCV Docs](https://docs.opencv.org/)
- [PyTorch Docs](https://pytorch.org/docs/)

---

**Desenvolvido com ❤️ para análise de futebol**
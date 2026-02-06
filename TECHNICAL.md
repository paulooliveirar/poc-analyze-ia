# 🎓 Documentação Técnica - Sistema de Detecção de Futebol

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Tecnologias](#tecnologias)
3. [Arquitetura](#arquitetura)
4. [Algoritmos](#algoritmos)
5. [Customização Avançada](#customização-avançada)

---

## 🎯 Visão Geral

Este sistema implementa detecção de objetos em tempo real para análise de partidas de futebol usando:

- **YOLOv8** (You Only Look Once v8) para detecção
- **OpenCV** para processamento de vídeo
- **PyTorch** como backend de Deep Learning

### Características Principais

```
Input (Vídeo/Câmera)
    ↓
[Preprocessamento OpenCV]
    ↓
[YOLOv8 Neural Network]
    ↓
[Post-processamento NMS]
    ↓
[Rastreamento e Análise]
    ↓
Output (Vídeo com Labels)
```

---

## 🔧 Tecnologias

### YOLOv8
- **Arquitetura**: CNN (Convolutional Neural Network)
- **Entrada**: Imagem RGB (640x640 padrão)
- **Saída**: Bounding boxes + Confiança + Classes
- **Vantagens**:
  - ⚡ Muito rápido (real-time)
  - 🎯 Alta precisão
  - 📦 Modelos leves disponíveis
  - 🔧 Fácil de customizar

### OpenCV
- Captura de câmera/vídeo
- Manipulação de imagens
- Desenho de anotações
- Codificação de vídeos

### PyTorch
- Framework de Deep Learning
- Suporte GPU (CUDA)
- Inferência otimizada

---

## 🏗️ Arquitetura

### Estrutura de Classes

```python
FootballDetector
├── __init__(model_path, conf_threshold)
├── detect(frame)                    # Detecção bruta
├── draw_detections(frame, results)  # Renderização
├── process_video(...)               # Processamento de arquivo
└── process_camera(...)              # Processamento de câmera

AdvancedFootballAnalyzer
├── track_ball(results, frame)
├── calculate_distance()
├── draw_ball_trail()
├── draw_heatmap()
└── process_video_advanced()
```

### Pipeline de Processamento

```
Frame OpenCV (H×W×3 uint8)
    ↓
[Normalização] → [0, 1] float32
    ↓
[Resize] → 640×640
    ↓
[YOLOv8 Forward Pass] → Predictions
    ↓
[NMS - Non-Maximum Suppression]
    ↓
[Filtro por Confiança]
    ↓
[Extração de coordenadas]
    ↓
[Desenho de bounding boxes]
    ↓
Salvar/Exibir frame
```

---

## 🧠 Algoritmos

### 1. YOLO - You Only Look Once

**Princípio**: Divide a imagem em grid e prediz caixas em cada célula

```
Imagem (640×640)
    ↓
Grid (20×20)
    ↓
Cada célula prediz:
  - bx, by (centro da caixa)
  - bw, bh (dimensões)
  - P(objectness)
  - P(class₁), P(class₂), ...
    ↓
Decodificar coordenadas
    ↓
Aplicar Sigmoid/Softmax
```

### 2. Non-Maximum Suppression (NMS)

Remove detecções duplicadas:

```
for cada detecção ordenada por confiança:
  Manter a de maior confiança
  Remover todas com IoU > threshold
    ↓
    Resultado: Detecções únicas
```

IoU (Intersection over Union):
```
IoU = Área_Interseção / Área_União
```

### 3. Rastreamento (Advanced)

Implementação simples usando centroide:

```
Frame N: Jogador em (100, 150)
Frame N+1: Jogador em (105, 155)
    ↓
Centroide anterior está próximo?
SIM → Mesmo jogador (conectar para rastro)
NÃO → Novo jogador
```

### 4. Detecção de Bola (Custom)

Heurísticas aplicadas:
- Tamanho menor que jogador
- Cor próxima ao branco/laranja
- Área < 2000 pixels

---

## 🎨 Customização Avançada

### Modificar Classes Detectadas

Em [src/football_detector.py](src/football_detector.py):

```python
FOOTBALL_CLASSES = {
    0: "bola",
    1: "jogador_time_1",
    2: "jogador_time_2",
    3: "arbitro",
    4: "goleiro",
    5: "bandeirinha"
}

COLORS = {
    "bola": (B, G, R),  # BGR format OpenCV
    # ...
}
```

### Ajustar NMS e Confiança

```python
detector.model(frame, 
    conf=0.45,        # Confidence threshold
    iou=0.45,         # NMS IoU threshold
    max_det=300       # Max detections
)
```

### Usar Diferentes Backbones

```python
# Nano - Mais rápido
detector = FootballDetector("yolov8n.pt")

# Small - Balanceado
detector = FootballDetector("yolov8s.pt")

# Medium - Mais preciso
detector = FootballDetector("yolo26m.pt")

# Large - Mais preciso ainda
detector = FootballDetector("yolov8l.pt")
```

---

## 📊 Métricas de Performance

### Tempo de Inferência (ms por frame)

| Modelo | CPU | GPU RTX 3080 |
|--------|-----|--------------|
| yolov8n | 200ms | 4ms |
| yolov8s | 500ms | 7ms |
| yolov8x | 1000ms | 12ms |

### Memória RAM

| Modelo | Uso |
|--------|-----|
| yolov8n | ~50 MB |
| yolov8s | ~150 MB |
| yolov8x | ~500 MB |

### Memória GPU (VRAM)

| Modelo | Batch=1 | Batch=4 |
|--------|---------|---------|
| yolov8n | 200 MB | 400 MB |
| yolov8s | 500 MB | 1.2 GB |
| yolov8x | 1.5 GB | 3.5 GB |

---

## 🔬 Análise de Resultados

### Interpretar Detecções

```
Resultado YOLOv8:
├── boxes: Tensor [N, 4] - (x1, y1, x2, y2) em pixels
├── conf: Tensor [N] - Confiança [0, 1]
├── cls: Tensor [N] - ID da classe
└── N: Número de objetos detectados
```

### Exemplo de Processamento

```python
results = model(frame)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Coordenadas
        conf = box.conf[0]              # Confiança
        cls = int(box.cls[0])           # Classe
        
        # Usar em lógica customizada
        if conf > 0.7:
            # Confiança alta, processar
            pass
```

---

## 🚀 Otimizações Possíveis

### 1. Exportar para ONNX (30% mais rápido)
```python
model = YOLO('yolov8n.pt')
model.export(format='onnx')
```

### 2. Quantização (4x mais rápido)
```python
# uint8 quantization
model.export(format='tflite', int8=True)
```

### 3. Multi-Threading
```python
from threading import Thread

def detect_worker(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        results = detector.detect(frame)
        result_queue.put(results)

Thread(target=detect_worker, daemon=True).start()
```

### 4. Caching de Frames
```python
# Processar apenas keyframes
frame_count = 0
if frame_count % 5 == 0:  # Processar a cada 5 frames
    results = detector.detect(frame)
```

---

## 📚 Referências

### Papers Científicos
- YOLOv8: https://github.com/ultralytics/ultralytics
- YOLO original: https://arxiv.org/abs/1506.02640
- YOLOv4: https://arxiv.org/abs/2004.10934

### Documentação
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [OpenCV Docs](https://docs.opencv.org/)

### Datasets
- [Roboflow Universe](https://universe.roboflow.com/)
- [SoccerNet](https://www.soccer-net.org/)
- [COCO Dataset](https://cocodataset.org/)

---

## 🔄 Fluxo Completo de Execução

```
1. Inicialização
   ├── Carregar modelo YOLOv8
   └── Configurar câmera/vídeo

2. Loop de Processamento
   ├── Ler frame
   ├── Preprocessar (resize, normalize)
   ├── Forward pass neural network
   ├── Aplicar NMS
   ├── Filtrar por confiança
   ├── Extrair coordenadas
   ├── Renderizar anotações
   ├── Salvar/exibir
   └── Repetir até EOF/Q

3. Limpeza
   ├── Liberar câmera
   ├── Fechar vídeo
   └── Destruir janelas
```

---

## 🎓 Conceitos Importantes

### IoU (Intersection over Union)
```
      ╔════════════════╗
      ║ Predito        ║
      ║  ╔─────────────╫─┐
      ║  ║ Ground Truth║ │
      ╚══╫─────────────╬─┘
         └─────────────┘

IoU = Área Amarela / Área Total
```

### Confidence Threshold
- **Alto** (0.8): Apenas detecções muito confiáveis (menos falsos positivos)
- **Médio** (0.5): Balanceado (padrão)
- **Baixo** (0.2): Muitas detecções (mais falsos positivos)

### NMS IoU Threshold
- **0.45**: NMS agressivo (menos boxes sobrepostas)
- **0.7**: NMS suave (mantém mais boxes)

---

**Última atualização**: 5 de fevereiro de 2026

Para dúvidas técnicas, consulte a documentação do Ultralytics.

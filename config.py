"""
Arquivo de configuração para o sistema de detecção de futebol.
Modifique os valores conforme necessário.
"""

# ==========================================
# CONFIGURAÇÕES DE MODELO
# ==========================================

# Modelo a usar: yolov8n, yolov8s, yolov8x, yolov8l
# Recomendado: yolov8x para melhor precisão em futebol
MODEL = "yolov8x"

# Limiar de confiança (0.0 - 1.0)
# Aumentado para 0.60 para evitar falsos positivos (jogadores detectados como bola)
# Valores altos = menos detecções, mas mais confiáveis
# Valores baixos = mais detecções
CONFIDENCE_THRESHOLD = 0.60

# Limiar IoU para Non-Maximum Suppression
IOU_THRESHOLD = 0.45

# Máximo de detecções por frame
MAX_DETECTIONS = 300

# ==========================================
# CONFIGURAÇÕES DE VÍDEO/CÂMERA
# ==========================================

# Resolução preferida para câmera (width, height)
CAMERA_RESOLUTION = (1280, 720)

# FPS da câmera (0 = automático)
CAMERA_FPS = 0

# ID da câmera (0 = câmera padrão)
CAMERA_ID = 0

# ==========================================
# CONFIGURAÇÕES DE CORES (BGR)
# ==========================================

COLORS = {
    "bola": (0, 165, 255),          # Laranja
    "jogador_time_1": (255, 0, 0),  # Azul
    "jogador_time_2": (0, 0, 255),  # Vermelho
    "arbitro": (0, 255, 0),         # Verde
    "goleiro": (255, 255, 0),       # Ciano
    "bandeirinha": (255, 0, 255),   # Magenta
    "default": (255, 255, 255)      # Branco
}

# ==========================================
# CONFIGURAÇÕES DE EXIBIÇÃO
# ==========================================

# Mostrar estatísticas no vídeo
SHOW_STATS = True

# Tamanho da fonte
FONT_SIZE = 0.6
FONT_THICKNESS = 2

# Espessura das bounding boxes
BBOX_THICKNESS = 2

# ==========================================
# CONFIGURAÇÕES DE PROCESSAMENTO
# ==========================================

# Usar GPU (device=0) ou CPU (device='cpu')
DEVICE = 0

# Número de threads para CPU
NUM_THREADS = 4

# ==========================================
# CONFIGURAÇÕES DE RASTREAMENTO
# ==========================================

# Tamanho máximo do rastro da bola
BALL_TRAIL_LENGTH = 20

# Tamanho máximo do rastro de jogadores
PLAYER_TRAIL_LENGTH = 10

# ==========================================
# CONFIGURAÇÕES DE ARQUIVO
# ==========================================

# Codec de vídeo: 'mp4v', 'mjpg', 'xvid'
VIDEO_CODEC = 'mp4v'

# Diretório de modelos
MODELS_DIR = './models'

# Diretório de vídeos
VIDEOS_DIR = './videos'

# ==========================================
# CONFIGURAÇÕES AVANÇADAS
# ==========================================

# Habilitar modo debug
DEBUG_MODE = False

# Salvar frames individuais
SAVE_FRAMES = False

# Intervalo de frames para salvar (1 = todos)
FRAME_SAVE_INTERVAL = 30

# Aplicar aumentação de dados (mais lento, mais preciso)
USE_AUGMENTATION = False

# ==========================================
# CONFIGURAÇÕES DE CLASSIFICAÇÃO CUSTOMIZADA
# ==========================================

# Classe padrão de futebol
FOOTBALL_CLASSES = {
    0: "bola",
    1: "jogador_time_1",
    2: "jogador_time_2",
    3: "arbitro",
    4: "goleiro",
    5: "bandeirinha"
}

# Tamanho aproximado de cada objeto em pixels
OBJECT_SIZES = {
    "bola": (30, 30),           # Pequena
    "jogador_time_1": (150, 200),  # Grande
    "jogador_time_2": (150, 200),  # Grande
    "arbitro": (150, 200),      # Grande
    "goleiro": (150, 200),      # Grande
    "bandeirinha": (100, 150)   # Média
}

# ==========================================
# CONFIGURAÇÕES DE TREINAMENTO
# ==========================================

# Número de épocas
TRAINING_EPOCHS = 100

# Tamanho do batch
BATCH_SIZE = 16

# Tamanho das imagens de entrada
IMG_SIZE = 640

# Taxa de aprendizado inicial
LEARNING_RATE = 0.01

# Otimizador: 'SGD', 'Adam', 'AdamW'
OPTIMIZER = 'SGD'

# Paciência para early stopping (épocas)
PATIENCE = 20

# ==========================================
# CONFIGURAÇÕES DE VALIDAÇÃO
# ==========================================

# Porcentagem de validação
VAL_SPLIT = 0.2

# Porcentagem de teste
TEST_SPLIT = 0.1

# Métrica de avaliação: 'mAP', 'F1', 'precision', 'recall'
EVAL_METRIC = 'mAP'

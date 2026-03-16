FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Variáveis de ambiente para execução Python e runtime NVIDIA
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/.config/Ultralytics \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Dependências de sistema necessárias para OpenCV, vídeo e Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libx11-xcb1 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-randr0 \
    libxcb-image0 \
    libxcb-icccm4 \
    libsm6 \
    libxext6 \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Garante comandos python/pip consistentes no container
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --upgrade pip setuptools wheel

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas requirements primeiro para aproveitar cache de build
COPY requirements.txt /app/requirements.txt

# Instala stack PyTorch com CUDA 12.1 + dependências do projeto
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio \
    && python -m pip install \
    ultralytics onnxruntime-gpu filterpy scikit-learn \
    && python - <<'PY'
from pathlib import Path
import re
import subprocess
req = Path("/app/requirements.txt").read_text(encoding="utf-8").splitlines()
safe_requirements = []
for line in req:
    item = line.strip()
    if not item or item.startswith("#"):
        continue
    if re.match(r"^(torch|torchvision|torchaudio)([<>=!~].*)?$", item):
        continue
    safe_requirements.append(item)
if safe_requirements:
    subprocess.check_call(["python", "-m", "pip", "install", *safe_requirements])
PY

# Copia o restante do código
COPY . /app

# Pasta padrão de vídeos e resultados (montada via volume no docker-compose)
RUN mkdir -p /app/videos /app/runs /app/.config/Ultralytics \
    && chmod -R 777 /app/.config

# Comando padrão: pode ser sobrescrito no docker-compose
CMD ["python", "main.py"]


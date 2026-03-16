PROJECT_NAME=football-analyzer
COMPOSE=docker compose -f docker-compose.yml

.PHONY: help build up up-detached down logs shell rebuild train-all train-homography gpu-check nvidia-check nvidia-toolkit-ubuntu24

help:
	@echo "Comandos disponíveis:"
	@echo "  make build        - Build da imagem Docker"
	@echo "  make up           - Sobe o container com GPU em modo interativo (menu main.py)"
	@echo "  make up-detached  - Sobe o container com GPU em background"
	@echo "  make down         - Para e remove o container"
	@echo "  make logs         - Mostra logs do container"
	@echo "  make shell        - Entra em um shell bash dentro do container em execução"
	@echo "  make rebuild      - Rebuild forçado (sem cache) e sobe o container"
	@echo "  make train-all E=50 - Treina os 5 modelos da spec e exporta .engine"
	@echo "  make train-homography E=50 - Treina modelo de homografia/pose (football-field-detection) por E épocas"
	@echo "  make gpu-check    - Verifica se a GPU está ativa no container"
	@echo "  make nvidia-check - Verifica suporte NVIDIA no Docker host"
	@echo "  make nvidia-toolkit-ubuntu24 - Instala toolkit NVIDIA no Ubuntu 24.04"

build:
	$(COMPOSE) build

up:
	# Necessário em Linux para permitir janelas OpenCV saírem do container
	xhost +local:docker || true
	$(COMPOSE) up

up-detached:
	xhost +local:docker || true
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

shell:
	docker exec -it $(PROJECT_NAME) /bin/bash

rebuild:
	xhost +local:docker || true
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

# Treina os modelos isolados da spec e exporta TensorRT (.engine)
# Uso: make train-all E=50
nvidia-toolkit-ubuntu24:
	@echo "Configurando NVIDIA Container Toolkit no Ubuntu 24.04..."
	@sudo apt-get update
	@sudo apt-get install -y curl gpg ca-certificates
	@sudo mkdir -p /etc/apt/keyrings
	@curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
	@curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
	@sudo apt-get update
	@sudo apt-get install -y nvidia-container-toolkit
	@sudo nvidia-ctk runtime configure --runtime=docker
	@sudo systemctl restart docker
	@echo "✅ Toolkit instalado. Execute: make nvidia-check"

nvidia-check:
	@command -v nvidia-smi >/dev/null 2>&1 || (echo "❌ Driver NVIDIA não encontrado no host (nvidia-smi indisponível)."; exit 1)
	@$(COMPOSE) run --rm $(PROJECT_NAME) python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 || (echo "❌ GPU não disponível no container de treino."; echo "   Se aparecer 'cuda>=12.8', atualize driver NVIDIA do host ou use uma imagem CUDA menor."; echo "   Para Ubuntu 24.04, execute: make nvidia-toolkit-ubuntu24"; echo "   Depois valide com: make nvidia-check"; exit 1)

train-all: nvidia-check
	@if [ -z "$(E)" ]; then \
	  E=50; \
	else \
	  E=$(E); \
	fi; \
	echo "Treinando todos os datasets por $$E épocas com GPU..."; \
	$(COMPOSE) run --rm $(PROJECT_NAME) python -m src.train_custom_model $$E

gpu-check: nvidia-check
	$(COMPOSE) run --rm $(PROJECT_NAME) python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"

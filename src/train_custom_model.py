"""
Treinamento SDD para o sistema de análise tática (spec em specs/soccer_analysis_system.mdc).

Fase 1 da spec:
- Treinamento isolado por modelo especializado
- Exportação obrigatória para TensorRT (.engine)
"""

from pathlib import Path
import shutil
import importlib.util
from typing import Optional

import torch
from ultralytics import YOLO

def train_football_model(
    dataset_path: Optional[str] = None,
    epochs: int = 100,
    imgsz: int = 416,
    batch_size: int = 4,
    device: int | str = "cpu",
    name: str = "football_detector",
    early_stopping_patience: int = 20,
):
    """
    Treina um modelo YOLOv8 customizado para futebol.
    
    Args:
        dataset_path: Caminho para dataset em formato YOLO
        epochs: Número de épocas de treinamento
        imgsz: Tamanho das imagens
        batch_size: Tamanho do batch
        device: ID do dispositivo (GPU) para treinamento
    """
    
    if dataset_path is None:
        print("=" * 60)
        print("COMO PREPARAR SEU DATASET")
        print("=" * 60)
        print("""
1. Use o Roboflow para coletar e anotar dados:
   - Acesse: https://roboflow.com/
   - Crie um dataset com imagens de partidas de futebol
   - Anote: jogadores, bola, árbitro, etc.
   
2. Exporte em formato YOLOv8:
   - Dataset > Export > YOLOv8
   
3. Estrutura esperada:
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   
4. Arquivo data.yaml:
   path: /caminho/para/dataset
   train: images/train
   val: images/val
   test: images/test
   nc: 3  # número de classes
   names: ['bola', 'jogador_time_1', 'jogador_time_2']
        """)
        return
    
    # Carregar modelo base
    model = YOLO("yolo11m.pt")
    
    # Treinar
    print(f"\nIniciando treinamento...")
    print(f"Dataset: {dataset_path}")
    print(f"Épocas: {epochs}")
    print(f"Tamanho das imagens: {imgsz}")
    print(f"Batch size: {batch_size}")
    
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=early_stopping_patience,  # Early stopping
        save=True,
        device=device,  # GPU id (0) ou 'cpu'
        name=name,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
    )
    
    print("\nTreinamento concluído!")
    print(f"Melhor modelo salvo em: runs/detect/{name}/weights/best.pt")
    
    return results


def export_to_onnx(model_path: str) -> str:
    """
    Exporta um modelo YOLO para ONNX como fallback de compatibilidade.
    """
    model = YOLO(model_path)
    output = model.export(format="onnx", opset=17)
    print(f"Modelo exportado para ONNX: {output}")
    return str(output)


def _has_tensorrt_sm_support() -> bool:
    """
    Verifica se a GPU atual atende o requisito mínimo de SM para TensorRT.

    Observação: TensorRT recente não suporta GPUs Pascal (ex.: SM 6.1).
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU detectada: capability SM {major}.{minor}")
    return major >= 7


def _has_tensorrt_python_package() -> bool:
    """
    Verifica se o binding Python do TensorRT está instalado.
    """
    return importlib.util.find_spec("tensorrt") is not None


def export_to_tensorrt(model_path: str) -> str:
    """
    Exporta um modelo YOLO para TensorRT (.engine), conforme regra de ouro da spec.
    """
    if not _has_tensorrt_sm_support():
        print(
            "[AVISO] GPU atual não suporta export TensorRT desta versão "
            "(requer SM >= 7.0). Usando fallback para ONNX."
        )
        return export_to_onnx(model_path)
    if not _has_tensorrt_python_package():
        print(
            "[AVISO] Pacote Python 'tensorrt' não encontrado no ambiente. "
            "Evitando AutoUpdate em runtime e aplicando fallback para ONNX."
        )
        return export_to_onnx(model_path)

    model = YOLO(model_path)
    try:
        output = model.export(format="engine", device=0, half=True)
        print(f"Modelo exportado para TensorRT: {output}")
        return str(output)
    except RuntimeError as exc:
        print(
            f"[AVISO] Falha no TensorRT ({exc}). "
            "Aplicando fallback para ONNX sem interromper o pipeline."
        )
        return export_to_onnx(model_path)


def evaluate_model(model_path: str, dataset_path: str) -> None:
    """
    Avalia o desempenho do modelo no conjunto de validação.
    
    Args:
        model_path: Caminho do modelo
        dataset_path: Caminho do dataset (data.yaml)
    """
    model = YOLO(model_path)
    metrics = model.val(data=dataset_path)
    print(metrics)


def _resolve_data_yaml(dataset_folder: Path) -> Path:
    data_yaml = dataset_folder / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml não encontrado em {dataset_folder}")
    return data_yaml


def _train_isolated_model(
    base_dir: Path,
    *,
    run_name: str,
    task: str,
    dataset_dir_name: str,
    base_model: str,
    epochs: int,
    imgsz: int = 960,
    batch: int = 4,
    patience: int = 30,
    lr0: float = 0.003,
    lrf: float = 0.01,
    export_name: str = "",
) -> Optional[str]:
    """
    Treina um único modelo de forma isolada e exporta para .engine.
    """
    dataset_path = base_dir / dataset_dir_name
    try:
        data_yaml = _resolve_data_yaml(dataset_path)
    except FileNotFoundError as exc:
        print(f"[AVISO] {exc}")
        return None

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n▶ Treinando {run_name} ({task})")
    print(f"   dataset: {data_yaml}")
    print(f"   base_model: {base_model}")

    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=run_name,
        task=task,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        cache=True,
        amp=True,
        workers=4,
        patience=patience,
        lr0=lr0,
        lrf=lrf,
    )

    best_weights = base_dir / "runs" / task / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"[ERRO] best.pt não encontrado para {run_name}: {best_weights}")
        return None
    exported_path = export_to_tensorrt(str(best_weights))
    if not export_name:
        return exported_path

    model_store = base_dir / "models-ia" / "football-analysis"
    model_store.mkdir(parents=True, exist_ok=True)
    suffix = Path(exported_path).suffix
    target_path = model_store / f"{export_name}{suffix}"
    shutil.copy2(exported_path, target_path)
    print(f"Modelo copiado para: {target_path}")
    return str(target_path)


def train_spec_models(
    epochs: int = 50,
    early_stopping_patience: Optional[int] = None,
) -> dict[str, Optional[str]]:
    """
    Implementa os treinamentos da spec (Fase 1), cada modelo de forma isolada.
    """
    base_dir = Path(__file__).resolve().parents[1]
    specs = [
        {
            "run_name": "player-referee-detection",
            "task": "detect",
            "dataset_dir_name": "soccer.v2i.yolov11",
            "base_model": "yolov8m.pt",
            "imgsz": 960,
            "batch": 4,
            "patience": 30,
            "lr0": 0.003,
            "lrf": 0.01,
            "export_name": "player-referee-detection",
        },
        {
            "run_name": "ball-detection",
            "task": "detect",
            "dataset_dir_name": "ball-0zqmb",
            "base_model": "yolov8m.pt",
            "imgsz": 1280,
            "batch": 2,
            "patience": 40,
            "lr0": 0.002,
            "lrf": 0.01,
            "export_name": "ball-detection",
        },
        {
            "run_name": "football-field-detection",
            "task": "detect",
            "dataset_dir_name": "football-field-detection.v15i.yolov8",
            "base_model": "yolov8m.pt",
            "imgsz": 960,
            "batch": 4,
            "patience": 30,
            "lr0": 0.003,
            "lrf": 0.01,
            "export_name": "football-field-detection",
        },
        {
            "run_name": "pitch-segmentation",
            "task": "segment",
            "dataset_dir_name": "football-pitch-segmentation.v2i.yolov8",
            "base_model": "yolov8n-seg.pt",
            "imgsz": 960,
            "batch": 2,
            "patience": 35,
            "lr0": 0.0025,
            "lrf": 0.01,
            "export_name": "pitch-segmentation",
        },
        {
            "run_name": "football-field-detection-pose",
            "task": "pose",
            "dataset_dir_name": "football-field-detection.v15i.yolov8",
            "base_model": "yolov8n-pose.pt",
            "imgsz": 960,
            "batch": 2,
            "patience": 35,
            "lr0": 0.0025,
            "lrf": 0.01,
            "export_name": "football-field-detection-pose",
        },
    ]

    outputs: dict[str, Optional[str]] = {}
    for item in specs:
        engine_path = _train_isolated_model(
            base_dir,
            run_name=item["run_name"],
            task=item["task"],
            dataset_dir_name=item["dataset_dir_name"],
            base_model=item["base_model"],
            epochs=epochs,
            imgsz=item["imgsz"],
            batch=item["batch"],
            patience=early_stopping_patience if early_stopping_patience is not None else item["patience"],
            lr0=item["lr0"],
            lrf=item["lrf"],
            export_name=item["export_name"],
        )
        outputs[item["run_name"]] = engine_path
    return outputs


def train_player_roles_verification(
    epochs: int = 120,
    early_stopping_patience: int = 40,
) -> Optional[str]:
    """
    Treina um modelo de verificação de papéis (goleiro, juiz, bola e times)
    a partir do checkpoint já treinado de player_detection_soccer_v2.
    """
    base_dir = Path(__file__).resolve().parents[1]
    pretrained_weights = (
        base_dir / "runs" / "detect" / "player_detection_soccer_v2" / "weights" / "best.pt"
    )
    if not pretrained_weights.exists():
        print(
            "[AVISO] Checkpoint base não encontrado em runs/detect/player_detection_soccer_v2."
            " Usando yolo11m.pt como fallback."
        )
        base_model = "yolo11m.pt"
    else:
        base_model = str(pretrained_weights)

    verification_output = _train_isolated_model(
        base_dir,
        run_name="player_detection_soccer_v2_roles_verification",
        task="detect",
        dataset_dir_name="models-ia/football-analysis/soccer-6zdkc",
        base_model=base_model,
        epochs=epochs,
        imgsz=1280,
        batch=4,
        patience=early_stopping_patience,
        lr0=0.0015,
        lrf=0.01,
        export_name="player-roles-verification",
    )
    return verification_output

if __name__ == "__main__":
    import sys

    # Uso principal deste módulo passa a ser via Makefile,
    # mas mantemos uma CLI simples para compatibilidade.
    print(sys.argv)
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        # python -m src.train_custom_model 50 -> treino isolado conforme spec
        train_spec_models(epochs=int(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1] == "roles-verification":
        # python -m src.train_custom_model roles-verification
        output = train_player_roles_verification()
        print(f"Saída do treino de verificação: {output}")
    elif len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        train_football_model(dataset_path=dataset_path)
    else:
        train_football_model()

"""
Treinamento SDD para o sistema de análise tática (spec em specs/soccer_analysis_system.mdc).

Fase 1 da spec:
- Treinamento isolado por modelo especializado
- Exportação obrigatória para TensorRT (.engine)
"""

from pathlib import Path
import shutil
import importlib.util
import argparse
from typing import Any, Optional

import torch
import yaml
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


def _resolve_training_device(device: int | str = "auto") -> int | str:
    """
    Resolve o device para treino, aceitando:
    - auto  -> usa 0,1 se houver >=2 GPUs, senão 0
    - cpu   -> força CPU
    - 0     -> GPU única
    - 0,1   -> multi-GPU (DDP no Ultralytics)
    """
    if isinstance(device, str) and device.lower() == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        print("[AVISO] CUDA indisponível no ambiente. Usando CPU.")
        return "cpu"

    gpu_count = torch.cuda.device_count()
    if device == "auto":
        if gpu_count >= 2:
            return "0,1"
        return "0"

    if isinstance(device, int):
        if device < 0 or device >= gpu_count:
            raise ValueError(
                f"Device GPU inválido: {device}. GPUs disponíveis: 0..{gpu_count - 1}"
            )
        return device

    if isinstance(device, str):
        parts = [item.strip() for item in device.split(",") if item.strip()]
        if not parts:
            raise ValueError("Parâmetro de device vazio. Use 'auto', 'cpu', '0' ou '0,1'.")
        for part in parts:
            if not part.isdigit():
                raise ValueError(f"Device inválido: {device}. Formato esperado: '0' ou '0,1'.")
            gpu_id = int(part)
            if gpu_id < 0 or gpu_id >= gpu_count:
                raise ValueError(
                    f"GPU {gpu_id} não existe. GPUs disponíveis: 0..{gpu_count - 1}"
                )
        return ",".join(parts)

    raise ValueError(f"Tipo de device não suportado: {type(device)}")


def _default_base_model_for_task(task: str) -> str:
    if task == "segment":
        return "yolo11m-seg.pt"
    if task == "pose":
        return "yolo11m-pose.pt"
    if task == "obb":
        return "yolov8n-obb.pt"
    return "yolo11m.pt"


def _augmentation_profile_for_task(task: str) -> dict[str, Any]:
    """
    Perfil de augmentations para robustez em cenário real:
    zoom, baixa luz, blur/movimento, oclusão e variação de escala.
    """
    profile: dict[str, Any] = {
        "hsv_h": 0.02,
        "hsv_s": 0.75,
        "hsv_v": 0.55,
        "degrees": 7.5,
        "translate": 0.10,
        "scale": 0.45,
        "shear": 2.0,
        "perspective": 0.0008,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.15,
        "erasing": 0.4,
    }
    if task == "segment":
        profile["copy_paste"] = 0.2
    else:
        profile["copy_paste"] = 0.0
    return profile


def _infer_task_from_dataset(dataset_folder: Path, config: dict[str, Any]) -> str:
    folder_name = dataset_folder.name.lower()
    if "kpt_shape" in config:
        return "pose"
    if "obb" in folder_name:
        return "obb"
    if "segmentation" in folder_name or "segment" in folder_name:
        return "segment"
    return "detect"


def _has_train_split(data_yaml: Path, config: dict[str, Any]) -> bool:
    train_ref = config.get("train")
    if not train_ref:
        return False
    train_path = (data_yaml.parent / str(train_ref)).resolve()
    return train_path.exists()


def _discover_folder_training_profiles(base_dir: Path) -> list[dict[str, Any]]:
    """
    Descobre datasets em models-ia/football-analysis contendo data.yaml.
    """
    dataset_root = base_dir / "models-ia" / "football-analysis"
    if not dataset_root.exists():
        return []

    custom_profiles: dict[str, dict[str, Any]] = {
        "soccer.v2i.yolov11": {
            "run_name": "player-referee-detection",
            "task": "detect",
            "base_model": "yolo11m.pt",
            "imgsz": 960,
            "batch": 16,
            "patience": 35,
            "lr0": 0.0025,
            "lrf": 0.01,
            "export_name": "player-referee-detection",
        },
        "soccer_ball_v1i_yolov11": {
            "run_name": "ball-detection",
            "task": "detect",
            "base_model": "yolo11m.pt",
            "imgsz": 1280,
            "batch": 8,
            "patience": 45,
            "lr0": 0.0020,
            "lrf": 0.01,
            "export_name": "ball-detection",
        },
        "football-players-detection.v19-yolo11m.yolov8": {
            "run_name": "players-detection-v19",
            "task": "detect",
            "base_model": "yolo11m.pt",
            "imgsz": 960,
            "batch": 16,
            "patience": 35,
            "lr0": 0.0025,
            "lrf": 0.01,
            "export_name": "players-detection-v19",
        },
        "pitch-segmentation.engine": {
            "run_name": "pitch-segmentation",
            "task": "segment",
            "base_model": "yolo11m-seg.pt",
            "imgsz": 1024,
            "batch": 8,
            "patience": 40,
            "lr0": 0.0020,
            "lrf": 0.01,
            "export_name": "pitch-segmentation",
        },
        "field-keypoints-detection.engine": {
            "run_name": "field-keypoints-detection",
            "task": "pose",
            "base_model": "yolo11m-pose.pt",
            "imgsz": 960,
            "batch": 8,
            "patience": 40,
            "lr0": 0.0020,
            "lrf": 0.01,
            "export_name": "field-keypoints-detection",
        },
        "soccer.v1i.yolov8-obb": {
            "run_name": "players-ball-detection-obb",
            "task": "obb",
            "base_model": "yolov8n-obb.pt",
            "imgsz": 960,
            "batch": 16,
            "patience": 35,
            "lr0": 0.0025,
            "lrf": 0.01,
            "export_name": "players-ball-detection-obb",
        },
    }

    profiles: list[dict[str, Any]] = []
    for dataset_folder in sorted(dataset_root.iterdir()):
        if not dataset_folder.is_dir():
            continue
        data_yaml = dataset_folder / "data.yaml"
        if not data_yaml.exists():
            continue

        with data_yaml.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        if not _has_train_split(data_yaml, config):
            print(f"[AVISO] Dataset sem split de treino válido, ignorando: {data_yaml}")
            continue

        base_profile = custom_profiles.get(dataset_folder.name, {})
        inferred_task = _infer_task_from_dataset(dataset_folder, config)
        task = base_profile.get("task", inferred_task)
        run_name = base_profile.get(
            "run_name", dataset_folder.name.replace(".", "-").replace("_", "-")
        )
        profile = {
            "run_name": run_name,
            "task": task,
            "data_yaml": str(data_yaml),
            "base_model": base_profile.get("base_model", _default_base_model_for_task(task)),
            "imgsz": int(base_profile.get("imgsz", 960)),
            "batch": int(base_profile.get("batch", 8)),
            "patience": int(base_profile.get("patience", 35)),
            "lr0": float(base_profile.get("lr0", 0.0025)),
            "lrf": float(base_profile.get("lrf", 0.01)),
            "export_name": base_profile.get("export_name", run_name),
        }
        profiles.append(profile)
    return profiles


def train_dataset_with_profile(
    *,
    data_yaml: str,
    run_name: str,
    task: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    lr0: float,
    lrf: float,
    export_name: str,
    device: int | str = "auto",
) -> Optional[str]:
    """
    Treina um dataset único e exporta para TensorRT/ONNX.
    """
    base_dir = Path(__file__).resolve().parents[1]
    data_yaml_path = Path(data_yaml).resolve()
    if not data_yaml_path.exists():
        print(f"[AVISO] data.yaml não encontrado: {data_yaml_path}")
        return None

    resolved_device = _resolve_training_device(device)
    print(f"\n▶ Treinando {run_name} ({task})")
    print(f"   data.yaml: {data_yaml_path}")
    print(f"   base_model: {base_model}")
    print(f"   device: {resolved_device}")

    model = YOLO(base_model)
    train_args = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": resolved_device,
        "name": run_name,
        "task": task,
        "pretrained": True,
        "optimizer": "AdamW",
        "cos_lr": True,
        "warmup_epochs": 3,
        "close_mosaic": 12,
        "cache": True,
        "amp": True,
        "workers": 8,
        "patience": patience,
        "lr0": lr0,
        "lrf": lrf,
        **_augmentation_profile_for_task(task),
    }
    model.train(**train_args)

    best_weights = base_dir / "runs" / task / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"[ERRO] best.pt não encontrado para {run_name}: {best_weights}")
        return None

    exported_path = export_to_tensorrt(str(best_weights))
    model_store = base_dir / "models-ia" / "football-analysis"
    model_store.mkdir(parents=True, exist_ok=True)
    suffix = Path(exported_path).suffix
    target_path = model_store / f"{export_name}{suffix}"
    shutil.copy2(exported_path, target_path)
    print(f"Modelo copiado para: {target_path}")
    return str(target_path)


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
    device: int | str = "auto",
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

    resolved_device = _resolve_training_device(device)
    print(f"\n▶ Treinando {run_name} ({task})")
    print(f"   dataset: {data_yaml}")
    print(f"   base_model: {base_model}")
    print(f"   device: {resolved_device}")

    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        name=run_name,
        task=task,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=12,
        cache=True,
        amp=True,
        workers=8,
        patience=patience,
        lr0=lr0,
        lrf=lrf,
        **_augmentation_profile_for_task(task),
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
    device: int | str = "auto",
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
            device=device,
        )
        outputs[item["run_name"]] = engine_path
    return outputs


def train_all_folder_models(
    epochs: int = 80,
    early_stopping_patience: Optional[int] = None,
    device: int | str = "auto",
) -> dict[str, Optional[str]]:
    """
    Treina automaticamente todos os datasets encontrados em:
    models-ia/football-analysis/**/data.yaml
    """
    base_dir = Path(__file__).resolve().parents[1]
    profiles = _discover_folder_training_profiles(base_dir)
    if not profiles:
        print("[AVISO] Nenhum dataset com data.yaml válido foi encontrado.")
        return {}

    outputs: dict[str, Optional[str]] = {}
    for profile in profiles:
        result = train_dataset_with_profile(
            data_yaml=profile["data_yaml"],
            run_name=profile["run_name"],
            task=profile["task"],
            base_model=profile["base_model"],
            epochs=epochs,
            imgsz=profile["imgsz"],
            batch=profile["batch"],
            patience=(
                early_stopping_patience
                if early_stopping_patience is not None
                else profile["patience"]
            ),
            lr0=profile["lr0"],
            lrf=profile["lrf"],
            export_name=profile["export_name"],
            device=device,
        )
        outputs[profile["run_name"]] = result
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

    # Compatibilidade com chamadas antigas.
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        train_spec_models(epochs=int(sys.argv[1]))
        sys.exit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "roles-verification":
        output = train_player_roles_verification()
        print(f"Saída do treino de verificação: {output}")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Treino de modelos de análise de futebol com suporte multi-GPU."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_all_parser = subparsers.add_parser(
        "train-folder", help="Treina todos os datasets válidos da pasta football-analysis."
    )
    train_all_parser.add_argument("--epochs", type=int, default=80)
    train_all_parser.add_argument("--patience", type=int, default=0)
    train_all_parser.add_argument("--device", type=str, default="auto")

    train_one_parser = subparsers.add_parser(
        "train-one", help="Treina um único dataset com parâmetros explícitos."
    )
    train_one_parser.add_argument("--data", type=str, required=True)
    train_one_parser.add_argument("--run-name", type=str, required=True)
    train_one_parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment", "pose", "obb"])
    train_one_parser.add_argument("--base-model", type=str, default="")
    train_one_parser.add_argument("--epochs", type=int, default=80)
    train_one_parser.add_argument("--imgsz", type=int, default=960)
    train_one_parser.add_argument("--batch", type=int, default=8)
    train_one_parser.add_argument("--patience", type=int, default=35)
    train_one_parser.add_argument("--lr0", type=float, default=0.0025)
    train_one_parser.add_argument("--lrf", type=float, default=0.01)
    train_one_parser.add_argument("--export-name", type=str, default="")
    train_one_parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    if args.command == "train-folder":
        patience_override = args.patience if args.patience > 0 else None
        outputs = train_all_folder_models(
            epochs=args.epochs,
            early_stopping_patience=patience_override,
            device=args.device,
        )
        print("\n✅ Treino em lote concluído.")
        for model_name, output_path in outputs.items():
            print(f"- {model_name}: {output_path}")
    elif args.command == "train-one":
        base_model = args.base_model or _default_base_model_for_task(args.task)
        export_name = args.export_name or args.run_name
        output = train_dataset_with_profile(
            data_yaml=args.data,
            run_name=args.run_name,
            task=args.task,
            base_model=base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            lr0=args.lr0,
            lrf=args.lrf,
            export_name=export_name,
            device=args.device,
        )
        print(f"✅ Treino concluído: {output}")
    else:
        train_football_model()

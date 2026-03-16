"""
Script isolado para testar o dataset
`football-field-detection.v15i.yolov8` (homografia/pose).

Uso dentro do container:
    python -m src.test_football_field_detection
ou (via Makefile, se desejar criar um alvo específico).
"""

from pathlib import Path

from ultralytics import YOLO


def main(epochs: int = 1) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_yaml = base_dir / "football-field-detection.v15i.yolov8" / "data.yaml"

    if not data_yaml.exists():
        print(f"[ERRO] data.yaml não encontrado em {data_yaml}")
        return

    print("\n" + "=" * 80)
    print("🧪 Teste rápido do dataset football-field-detection.v15i.yolov8")
    print(f"    data.yaml: {data_yaml}")
    print(f"    épocas: {epochs}")
    print("=" * 80)

    # Modelo de pose do Ultralytics (baixado automaticamente se não existir)
    model = YOLO("yolov8m-pose.pt")

    # Usa GPU se disponível, senão CPU
    device = 0 if model.device.type == "cuda" else "cpu"

    # Treino extremamente curto só para verificar se o dataset é aceito
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=8,
        patience=3,
        save=False,
        device=device,
        name="football_field_detection_test",
    )

    print("\n✅ Teste de dataset concluído.")
    print("Se não houver RuntimeError de labels/imagens, o dataset está formatado corretamente.")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        main(epochs=int(sys.argv[1]))
    else:
        main()


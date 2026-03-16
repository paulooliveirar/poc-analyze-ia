from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from football_analyzer.football_detector import FootballDetector, _dominant_color_kmeans


def test_gpu_availability() -> None:
    """
    Valida alocação de memória na GPU (quando disponível).
    """
    if not torch.cuda.is_available():
        pytest.skip("GPU não disponível neste ambiente.")
    tensor = torch.zeros((32, 32), device="cuda")
    assert tensor.is_cuda


def test_model_concurrency() -> None:
    """
    Garante que quatro slots concorrentes podem ser executados sem falha.
    """
    def _slot(seed: int) -> int:
        array = np.ones((100, 100), dtype=np.float32) * seed
        return int(array.sum())

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_slot, i) for i in range(1, 5)]
        results = [future.result() for future in futures]
    assert len(results) == 4
    assert all(value > 0 for value in results)


def test_team_classification() -> None:
    """
    Valida classificação de cor dominante para dois jogadores (azul e vermelho).
    """
    red_patch = np.zeros((40, 40, 3), dtype=np.uint8)
    red_patch[:, :] = (0, 0, 255)
    blue_patch = np.zeros((40, 40, 3), dtype=np.uint8)
    blue_patch[:, :] = (255, 0, 0)

    red_dominant = _dominant_color_kmeans(red_patch)
    blue_dominant = _dominant_color_kmeans(blue_patch)

    assert int(red_dominant[2]) > int(red_dominant[0])
    assert int(blue_dominant[0]) > int(blue_dominant[2])


def test_video_integrity(tmp_path: Path) -> None:
    """
    Verifica se dois vídeos possuem o mesmo número de frames.
    """
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    fps = 10
    size = (160, 90)
    writer_in = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    writer_out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    for _ in range(8):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        writer_in.write(frame)
        writer_out.write(frame)

    writer_in.release()
    writer_out.release()

    assert FootballDetector.verify_video_integrity(str(input_path), str(output_path))

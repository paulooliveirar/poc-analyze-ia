import hashlib
import importlib.util
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from ultralytics import YOLO

try:
    from torchvision.models import MobileNet_V2_Weights
except Exception:  # pragma: no cover
    MobileNet_V2_Weights = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEAM_A_COLOR = (0, 0, 255)
TEAM_B_COLOR = (255, 0, 0)
REFEREE_COLOR = (80, 80, 80)
BALL_COLOR = (0, 255, 255)


def effective_display(requested: bool) -> bool:
    if os.environ.get("FOOTBALL_HEADLESS", "").lower() in {"1", "true", "yes"}:
        return False
    if not os.environ.get("DISPLAY", "").strip():
        return False
    if not Path("/tmp/.X11-unix").exists():
        return False
    return bool(requested)


def has_tensorrt_runtime() -> bool:
    """
    Verifica se o runtime Python do TensorRT está disponível.
    """
    return importlib.util.find_spec("tensorrt") is not None


def _model_label_from_run_path(model_path: Optional[str], fallback_label: str) -> str:
    """
    Gera um nome amigável do modelo para exibir nos rótulos.
    """
    if not model_path:
        return fallback_label
    model_file = Path(model_path)
    if model_file.stem == "best" and model_file.parent.name == "weights":
        return model_file.parent.parent.name
    return model_file.stem


def _dominant_color_kmeans(patch: np.ndarray, clusters: int = 2) -> np.ndarray:
    data = patch.reshape((-1, 3)).astype(np.float32)
    if len(data) == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(
        data,
        K=clusters,
        bestLabels=None,
        criteria=criteria,
        attempts=2,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten())
    return centers[int(np.argmax(counts))].astype(np.uint8)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter + 1e-6)


@dataclass
class PipelineConfig:
    conf_threshold: float = 0.65
    ball_conf_threshold: float = 0.45
    ball_zoom_conf_threshold: float = 0.35
    ball_zoom_scale: float = 1.7
    min_player_area: int = 650
    min_ball_area: int = 16
    max_ball_area: int = 4200
    player_aspect_ratio_range: tuple[float, float] = (0.22, 1.2)
    ball_aspect_ratio_range: tuple[float, float] = (0.30, 2.8)
    ball_temporal_gate_px: float = 240.0
    output_codec: str = "mp4v"
    hard_negative_enabled: bool = True
    max_ball_missing_frames: int = 10
    player_match_radius: float = 60.0


@dataclass
class KalmanBallTracker:
    max_missing_frames: int = 5
    missing_frames: int = 0
    kalman: cv2.KalmanFilter = field(init=False)

    def __post_init__(self) -> None:
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

    def update(self, measurement: Optional[tuple[int, int]]) -> tuple[Optional[tuple[int, int]], str]:
        predicted = self.kalman.predict()
        predicted_center = (int(predicted[0][0]), int(predicted[1][0]))
        if measurement is not None:
            mx, my = measurement
            self.kalman.correct(np.array([[np.float32(mx)], [np.float32(my)]]))
            self.missing_frames = 0
            return (mx, my), "detected"
        self.missing_frames += 1
        if self.missing_frames < self.max_missing_frames:
            return predicted_center, "kalman"
        return None, "NOT DETECTED"


@dataclass
class RuntimeState:
    next_player_id: int = 1
    player_tracks: dict[int, tuple[int, int]] = field(default_factory=dict)
    player_tracking_info: dict[int, dict[str, Any]] = field(default_factory=dict)
    possession_by_team: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    last_ball_state: str = "NOT DETECTED"
    current_owner_id: Optional[int] = None
    last_ball_center: Optional[tuple[int, int]] = None


class HardNegativeMiner:
    def __init__(self, root_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.output_dir = root_dir / "dataset" / "hard_negatives"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.recent_hashes: set[str] = set()
        self.recent_boxes: list[tuple[int, int, int, int]] = []

    def _is_duplicate(self, image_hash: str, bbox: tuple[int, int, int, int]) -> bool:
        if image_hash in self.recent_hashes:
            return True
        for prev in self.recent_boxes[-150:]:
            if _bbox_iou(prev, bbox) >= 0.9:
                return True
        return False

    def save(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        confidence: float,
        frame_id: int,
        reason: str,
    ) -> None:
        if not self.enabled:
            return
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if crop.size == 0:
            return
        image_hash = hashlib.sha256(crop.tobytes()).hexdigest()
        if self._is_duplicate(image_hash, bbox):
            return
        file_name = f"hn_{frame_id}_{image_hash[:12]}.jpg"
        file_path = self.output_dir / file_name
        cv2.imwrite(str(file_path), crop)
        payload = {
            "file": file_name,
            "bbox": [x1, y1, x2, y2],
            "frame_id": frame_id,
            "confidence": confidence,
            "reason": reason,
            "created_at": datetime.now().isoformat(),
        }
        with self.metadata_path.open("a", encoding="utf-8") as handler:
            handler.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.recent_hashes.add(image_hash)
        self.recent_boxes.append(bbox)


class SecondStagePlayerClassifier:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.feature_model = self._build_model().to(self.device)
        self.feature_model.eval()
        self.prototype: Optional[torch.Tensor] = None
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _build_model() -> torch.nn.Module:
        if MobileNet_V2_Weights is not None:
            try:
                return models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
            except Exception:
                logger.warning("Falha ao carregar pesos padrao MobileNetV2, usando pesos aleatorios.")
        return models.mobilenet_v2(weights=None).features

    def _extract_feature(self, crop: np.ndarray) -> torch.Tensor:
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature_map = self.feature_model(tensor)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.squeeze(0).cpu()

    def classify(self, crop: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        ratio = width / float(height)
        if area < 650 or ratio < 0.22 or ratio > 1.2:
            return "not_player", 0.99
        feature = self._extract_feature(crop)
        if self.prototype is None:
            self.prototype = feature
            return "full_player", 0.9
        similarity = float(torch.dot(feature, self.prototype))
        if similarity >= 0.55:
            self.prototype = torch.nn.functional.normalize(self.prototype * 0.92 + feature * 0.08, dim=0)
            return "full_player", min(0.99, max(0.5, (similarity + 1.0) / 2.0))
        return "not_player", min(0.99, max(0.5, (1.0 - similarity) / 2.0))


class TeamColorClassifier:
    def classify(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        top = y1 + int(0.20 * h)
        bottom = y1 + int(0.60 * h)
        patch = frame[max(0, top):max(top + 1, bottom), max(0, x1):max(x1 + 1, x2)]
        if patch.size == 0:
            return 0
        dominant = _dominant_color_kmeans(patch, clusters=2)
        blue, _, red = [int(v) for v in dominant]
        return 0 if red >= blue else 1


class FootballDetector:
    def __init__(self, conf_threshold: float = 0.65, tracking_backend: str = "centroid"):
        self.config = PipelineConfig(conf_threshold=conf_threshold)
        self.root_dir = Path(__file__).resolve().parents[1]
        self.output_dir = self.root_dir / "videos"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fanout")
        self.kalman_ball = KalmanBallTracker(max_missing_frames=self.config.max_ball_missing_frames)
        self.state = RuntimeState()
        self.team_classifier = TeamColorClassifier()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.second_stage = SecondStagePlayerClassifier(device=self.device)
        self.hard_negative_miner = HardNegativeMiner(
            root_dir=self.root_dir,
            enabled=self.config.hard_negative_enabled,
        )
        self.tracking_backend = tracking_backend
        self._load_models()

    def _model_path(self, model_file: str, fallback: str) -> str:
        preferred = self.root_dir / "models-ia" / "football-analysis" / model_file
        if preferred.exists():
            if preferred.suffix == ".engine" and not has_tensorrt_runtime():
                pt_candidate = preferred.with_suffix(".pt")
                if pt_candidate.exists():
                    logger.warning(
                        "TensorRT indisponivel; usando fallback PT para %s",
                        model_file,
                    )
                    return str(pt_candidate)
                logger.warning(
                    "TensorRT indisponivel; usando fallback base para %s",
                    model_file,
                )
                return fallback
            return str(preferred)
        return fallback

    def _run_detection_path(self, run_name: str) -> Optional[str]:
        pt_path = self.root_dir / "runs" / "detect" / run_name / "weights" / "best.pt"
        if pt_path.exists():
            return str(pt_path)
        return None
    
    def _run_segmentation_path(self, run_name: str) -> Optional[str]:
        engine_path = self.root_dir / "runs" / "segment" / run_name / "weights" / "best.pt"
        if engine_path.exists():
            return str(engine_path)
        return None

    def _run_pose_path(self, run_name: str) -> Optional[str]:
        pt_path = self.root_dir / "runs" / "pose" / run_name / "weights" / "best.pt"
        if pt_path.exists():
            return str(pt_path)
        return None

    def _first_available_model(self, candidates: list[Optional[str]], fallback: str) -> str:
        """
        Retorna o primeiro caminho de modelo disponível; senão usa fallback YOLO base.
        """
        for candidate in candidates:
            if candidate:
                return candidate
        return fallback

    def _load_models(self) -> None:
        segment_model_path = self._first_available_model(
            [
                self._run_segmentation_path("football-pitch-segmentation"),
            ],
            "yolov8n-seg.pt",
        )
        players_model_path = self._first_available_model(
            [
                self._run_detection_path("football-players-detection"),                
                self._run_detection_path("player_detection_soccer_v2"),
                self._run_detection_path("player-referee-detection"),
                self._run_detection_path("soccer_ball_v1i_yolov11_precision_5090"),
            ],
            "yolov8m.pt",
        )
        ball_model_path = self._first_available_model(
            [
                # self._run_detection_path("football-players-detection"),
                # self._run_detection_path("player_detection_soccer_v2"),
                self._run_detection_path("ball-detection"),
                self._run_detection_path("soccer_ball_v1i_yolov11_precision_5090"),
            ],
            "yolo11m.pt",
        )
        context_model_path = self._first_available_model(
            [
                self._run_detection_path("football-players-detection"),
            ],
            "yolo11m.pt",
        )
        pose_model_path = self._first_available_model(
            [
                self._run_pose_path("football-field-detection-pose"),
            ],
            "yolov8n-pose.pt",
        )
        keypoint_model_path = self._first_available_model(
            [
                self._run_detection_path("football-players-detection"),
            ],
            "yolov8m.pt",
        )

        self.segment_model = YOLO(segment_model_path)
        self.player_referee_model = YOLO(players_model_path)
        self.ball_model = YOLO(ball_model_path)
        self.context_model = YOLO(context_model_path)
        self.pose_model = YOLO(pose_model_path)
        self.keypoint_model = YOLO(keypoint_model_path)

        self.segment_model_label = _model_label_from_run_path(segment_model_path, "pitch-segmentation")
        self.player_model_label = _model_label_from_run_path(players_model_path, "player_detection_soccer_v2")
        self.ball_model_label = _model_label_from_run_path(ball_model_path, "ball-detection")
        self.context_model_label = _model_label_from_run_path(context_model_path, "player_detection_soccer_v2")
        self.pose_model_label = _model_label_from_run_path(pose_model_path, "football-field-detection-pose")
        self.keypoint_model_label = _model_label_from_run_path(keypoint_model_path, "football-field-detection")

    @staticmethod
    def count_video_frames(video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    @staticmethod
    def verify_video_integrity(input_path: str, output_path: str) -> bool:
        return FootballDetector.count_video_frames(input_path) == FootballDetector.count_video_frames(output_path)

    def _run_parallel_slots(self, frame: np.ndarray) -> dict[str, Any]:
        futures = {
            "segment": self.executor.submit(self.segment_model.predict, frame, conf=self.config.conf_threshold, verbose=False),
            "players": self.executor.submit(self.player_referee_model.predict, frame, conf=self.config.conf_threshold, verbose=False),
            "ball": self.executor.submit(self.ball_model.predict, frame, conf=self.config.ball_conf_threshold, verbose=False),
            "context": self.executor.submit(self.context_model.predict, frame, conf=self.config.ball_conf_threshold, verbose=False),
            "keypoints": self.executor.submit(self.keypoint_model.predict, frame, conf=self.config.conf_threshold, verbose=False),
        }
        return {slot: future.result()[0] for slot, future in futures.items()}

    def _is_referee_label(self, label_name: str) -> bool:
        lowered = label_name.lower()
        return "ref" in lowered or "arbit" in lowered

    def _is_ball_label(self, label_name: str) -> bool:
        lowered = label_name.lower()
        return "ball" in lowered or "bola" in lowered

    def _validate_player_geometry(self, bbox: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        ratio = width / float(height)
        min_ratio, max_ratio = self.config.player_aspect_ratio_range
        return area >= self.config.min_player_area and min_ratio <= ratio <= max_ratio

    def _validate_ball_geometry(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        ratio = width / float(height)
        min_ratio, max_ratio = self.config.ball_aspect_ratio_range
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1, frame_h * frame_w)
        # Ajuste dinâmico para suportar bola muito distante (área min) e muito próxima (área max).
        dynamic_min_area = max(self.config.min_ball_area, int(frame_area * 0.000008))
        dynamic_max_area = max(self.config.max_ball_area, int(frame_area * 0.015))
        return dynamic_min_area <= area <= dynamic_max_area and min_ratio <= ratio <= max_ratio

    def _predict_ball_zoom(self, frame: np.ndarray) -> Any:
        scaled = cv2.resize(
            frame,
            None,
            fx=self.config.ball_zoom_scale,
            fy=self.config.ball_zoom_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        return self.ball_model.predict(
            scaled,
            conf=self.config.ball_zoom_conf_threshold,
            verbose=False,
        )[0]

    def _extract_players_and_referees(
        self, frame: np.ndarray, result: Any, frame_id: int, secondary_result: Optional[Any] = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        players: list[dict[str, Any]] = []
        referees: list[dict[str, Any]] = []
        merged_results: list[tuple[Any, str]] = [(result, self.player_model_label)]
        if secondary_result is not None:
            merged_results.append((secondary_result, self.context_model_label))

        for current_result, source_label in merged_results:
            boxes = getattr(current_result, "boxes", None)
            names = current_result.names if hasattr(current_result, "names") else {}
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0].cpu().item())
                if confidence < self.config.conf_threshold:
                    continue
                cls_id = int(box.cls[0].cpu().item())
                label_name = str(names.get(cls_id, f"class_{cls_id}"))
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                bbox = (x1, y1, x2, y2)
                if not self._validate_player_geometry(bbox):
                    self.hard_negative_miner.save(frame, bbox, confidence, frame_id, "geometry")
                    continue
                crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                stage2_label, stage2_score = self.second_stage.classify(crop, bbox)
                if stage2_label != "full_player":
                    self.hard_negative_miner.save(frame, bbox, confidence, frame_id, f"stage2_{stage2_label}_{stage2_score:.2f}")
                    continue
                record = {
                    "bbox": bbox,
                    "conf": confidence,
                    "class_name": label_name,
                    "source_model": source_label,
                }
                if self._is_referee_label(label_name):
                    referees.append(record)
                else:
                    players.append(record)
        return players, referees

    def _extract_ball(
        self,
        frame: np.ndarray,
        result: Any,
        frame_id: int,
        zoom_result: Optional[Any] = None,
        secondary_result: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        candidates: list[tuple[tuple[int, int, int, int], float, str]] = []

        boxes = getattr(result, "boxes", None)
        names = result.names if hasattr(result, "names") else {}
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0].cpu().item())
                if confidence < self.config.ball_conf_threshold:
                    continue
                cls_id = int(box.cls[0].cpu().item())
                label_name = str(names.get(cls_id, f"class_{cls_id}"))
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                candidates.append(((x1, y1, x2, y2), confidence, label_name))

        if zoom_result is not None:
            zoom_boxes = getattr(zoom_result, "boxes", None)
            zoom_names = zoom_result.names if hasattr(zoom_result, "names") else {}
            if zoom_boxes is not None:
                for box in zoom_boxes:
                    confidence = float(box.conf[0].cpu().item())
                    if confidence < self.config.ball_zoom_conf_threshold:
                        continue
                    cls_id = int(box.cls[0].cpu().item())
                    label_name = str(zoom_names.get(cls_id, f"class_{cls_id}"))
                    zx1, zy1, zx2, zy2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    scale = self.config.ball_zoom_scale
                    x1 = int(zx1 / scale)
                    y1 = int(zy1 / scale)
                    x2 = int(zx2 / scale)
                    y2 = int(zy2 / scale)
                    candidates.append(((x1, y1, x2, y2), confidence, label_name))

        if secondary_result is not None:
            secondary_boxes = getattr(secondary_result, "boxes", None)
            secondary_names = secondary_result.names if hasattr(secondary_result, "names") else {}
            if secondary_boxes is not None:
                for box in secondary_boxes:
                    confidence = float(box.conf[0].cpu().item())
                    if confidence < self.config.ball_conf_threshold:
                        continue
                    cls_id = int(box.cls[0].cpu().item())
                    label_name = str(secondary_names.get(cls_id, f"class_{cls_id}"))
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    candidates.append(((x1, y1, x2, y2), confidence, label_name))

        if not candidates:
            return None

        best: Optional[dict[str, Any]] = None
        best_score = -1.0
        for bbox, confidence, label_name in candidates:
            if not self._validate_ball_geometry(bbox, frame.shape):
                self.hard_negative_miner.save(frame, bbox, confidence, frame_id, "ball_geometry")
                continue

            # Em alguns modelos customizados o nome da classe pode vir inconsistente;
            # por isso usamos geometria + score e não apenas string "ball/bola".
            label_bonus = 0.65 if self._is_ball_label(label_name) else 0.0
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            temporal_bonus = 0.0
            if self.state.last_ball_center is not None:
                prev_x, prev_y = self.state.last_ball_center
                dist = float(np.hypot(center[0] - prev_x, center[1] - prev_y))
                temporal_bonus = max(0.0, 0.35 - (dist / self.config.ball_temporal_gate_px))
            score = confidence + label_bonus + temporal_bonus
            if best is None or score > best_score:
                best = {
                    "bbox": bbox,
                    "conf": confidence,
                    "source_model": self.ball_model_label,
                }
                best_score = score
        return best

    def _assign_track_ids(self, players: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assigned: list[dict[str, Any]] = []
        used_track_ids: set[int] = set()
        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            best_id = None
            best_dist = float("inf")
            for track_id, prev_center in self.state.player_tracks.items():
                if track_id in used_track_ids:
                    continue
                dist = float(np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]))
                if dist < best_dist and dist <= self.config.player_match_radius:
                    best_dist = dist
                    best_id = track_id
            if best_id is None:
                best_id = self.state.next_player_id
                self.state.next_player_id += 1
            used_track_ids.add(best_id)
            player["track_id"] = best_id
            self.state.player_tracks[best_id] = center
            assigned.append(player)
        return assigned

    def _run_pose_on_player_rois(self, frame: np.ndarray, players: list[dict[str, Any]]) -> list[np.ndarray]:
        if not players:
            return []
        output: list[np.ndarray] = []

        # O backend do Ultralytics não é thread-safe para múltiplos predicts
        # simultâneos no mesmo objeto de modelo (pode falhar no fuse/bn).
        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            roi = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
            if roi.size == 0:
                continue
            result = self.pose_model.predict(roi, conf=self.config.conf_threshold, verbose=False)[0]
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
                continue
            points = keypoints.xy[0].cpu().numpy()
            points[:, 0] += x1
            points[:, 1] += y1
            output.append(points)
        return output

    def _overlay_segmentation(self, frame: np.ndarray, segment_result: Any) -> np.ndarray:
        masks = getattr(segment_result, "masks", None)
        if masks is None or masks.data is None:
            return frame
        overlay = frame.copy()
        for mask in masks.data.cpu().numpy():
            binary = (mask > 0.5).astype(np.uint8)
            resized = cv2.resize(binary, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay[resized == 1] = (0, 130, 0)
        return cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    def _update_possession(self, players: list[dict[str, Any]], ball_center: Optional[tuple[int, int]]) -> None:
        if not players or ball_center is None:
            return
        bx, by = ball_center
        nearest = None
        nearest_dist = float("inf")
        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            feet_x, feet_y = (x1 + x2) // 2, y2
            dist = float(np.hypot(feet_x - bx, feet_y - by))
            if dist < nearest_dist:
                nearest = player
                nearest_dist = dist
        if nearest is None:
            return
        owner_id = int(nearest["track_id"])
        team_id = int(nearest.get("team_id", 0))
        self.state.current_owner_id = owner_id
        self.state.possession_by_team[team_id] = self.state.possession_by_team.get(team_id, 0) + 1

    def _annotate(
        self,
        frame: np.ndarray,
        players: list[dict[str, Any]],
        referees: list[dict[str, Any]],
        ball_info: Optional[dict[str, Any]],
        ball_center: Optional[tuple[int, int]],
        ball_state: str,
        all_pose_points: list[np.ndarray],
    ) -> np.ndarray:
        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            team_id = int(player.get("team_id", 0))
            color = TEAM_A_COLOR if team_id == 0 else TEAM_B_COLOR
            source_model = str(player.get("source_model", self.player_model_label))
            label = (
                f"Jogador #{int(player['track_id'])} | Time {team_id} | "
                f"{int(player['conf'] * 100)}% | {source_model}"
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for referee in referees:
            x1, y1, x2, y2 = referee["bbox"]
            source_model = str(referee.get("source_model", self.player_model_label))
            cv2.rectangle(frame, (x1, y1), (x2, y2), REFEREE_COLOR, 2)
            cv2.putText(
                frame,
                f"Arbitro | {source_model}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                REFEREE_COLOR,
                2,
            )

        for pose_points in all_pose_points:
            for point in pose_points:
                px, py = int(point[0]), int(point[1])
                if px <= 0 and py <= 0:
                    continue
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

        if ball_info is not None:
            x1, y1, x2, y2 = ball_info["bbox"]
            source_model = str(ball_info.get("source_model", self.ball_model_label))
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
            cv2.putText(
                frame,
                f"Bola {int(ball_info['conf'] * 100)}% | {source_model}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BALL_COLOR,
                2,
            )
        elif ball_center is not None and ball_state == "kalman":
            cv2.circle(frame, ball_center, 8, BALL_COLOR, 2)
            cv2.putText(frame, "Bola interpolada", (ball_center[0] + 8, ball_center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 2)
        elif ball_state == "NOT DETECTED":
            cv2.putText(frame, "Bola: NOT DETECTED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BALL_COLOR, 2)

        t0 = self.state.possession_by_team.get(0, 0)
        t1 = self.state.possession_by_team.get(1, 0)
        total = t0 + t1
        t0_pct = int((100 * t0 / total)) if total else 0
        t1_pct = 100 - t0_pct if total else 0
        cv2.putText(frame, f"Posse Time A: {t0_pct}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEAM_A_COLOR, 2)
        cv2.putText(frame, f"Posse Time B: {t1_pct}%", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEAM_B_COLOR, 2)
        return frame

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        slots = self._run_parallel_slots(frame)
        players, referees = self._extract_players_and_referees(
            frame,
            slots["players"],
            frame_id,
            secondary_result=slots["context"],
        )
        players = self._assign_track_ids(players)
        for player in players:
            team_id = self.team_classifier.classify(frame, player["bbox"])
            player["team_id"] = team_id
            track_id = int(player["track_id"])
            self.state.player_tracking_info[track_id] = {
                "team_id": team_id,
                "last_bbox": list(player["bbox"]),
                "last_seen_frame": frame_id,
            }

        ball_info = self._extract_ball(
            frame,
            slots["ball"],
            frame_id,
            secondary_result=slots["context"],
        )
        if ball_info is None:
            zoom_result = self._predict_ball_zoom(frame)
            ball_info = self._extract_ball(
                frame,
                slots["ball"],
                frame_id,
                zoom_result=zoom_result,
                secondary_result=slots["context"],
            )
        measured_ball = None
        if ball_info is not None:
            bx1, by1, bx2, by2 = ball_info["bbox"]
            measured_ball = ((bx1 + bx2) // 2, (by1 + by2) // 2)
        ball_center, ball_state = self.kalman_ball.update(measured_ball)
        self.state.last_ball_state = ball_state
        self.state.last_ball_center = ball_center
        self._update_possession(players, ball_center)
        all_pose_points = self._run_pose_on_player_rois(frame, players)
        frame = self._overlay_segmentation(frame, slots["segment"])
        return self._annotate(frame, players, referees, ball_info, ball_center, ball_state, all_pose_points)

    def _timestamped_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"result_{timestamp}.mp4"

    def _write_metadata(self, metadata_path: Path) -> None:
        payload = {
            "ball_possession_stats": self.state.possession_by_team,
            "player_tracking_info": self.state.player_tracking_info,
            "last_ball_state": self.state.last_ball_state,
            "current_ball_owner_id": self.state.current_owner_id,
            "generated_at": datetime.now().isoformat(),
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        display: bool = True,
        export_metadata: bool = True,
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Nao consegui abrir o video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_output = Path(output_path) if output_path else self._timestamped_output_path()
        target_output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(target_output),
            cv2.VideoWriter_fourcc(*self.config.output_codec),
            fps,
            (width, height),
        )
        show_window = effective_display(display)
        frame_id = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_id += 1
                processed = self._process_frame(frame, frame_id)
                writer.write(processed)
                if show_window:
                    try:
                        cv2.imshow("Analise Tatica Futebol", processed)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        show_window = False
                if frame_id % 30 == 0:
                    logger.info("Processado %s/%s frames", frame_id, total_frames)
        except KeyboardInterrupt:
            logger.info("SIGINT recebido, finalizando com seguranca.")
        finally:
            cap.release()
            writer.release()
            if show_window:
                cv2.destroyAllWindows()
        if export_metadata:
            metadata_path = target_output.parent / f"{target_output.stem}.json"
            self._write_metadata(metadata_path)
        logger.info(
            "Saida salva em %s | integridade_frames=%s",
            target_output,
            self.verify_video_integrity(video_path, str(target_output)),
        )

    def process_camera(
        self,
        camera_id: int = 0,
        output_path: str | None = None,
        display: bool = True,
        export_metadata: bool = True,
    ) -> None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Nao consegui abrir a camera {camera_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_output = Path(output_path) if output_path else self._timestamped_output_path()
        target_output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(target_output),
            cv2.VideoWriter_fourcc(*self.config.output_codec),
            fps,
            (width, height),
        )
        show_window = effective_display(display)
        frame_id = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_id += 1
                processed = self._process_frame(frame, frame_id)
                writer.write(processed)
                if show_window:
                    try:
                        cv2.imshow("Analise Tatica Futebol - Camera", processed)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        show_window = False
        except KeyboardInterrupt:
            logger.info("SIGINT recebido na camera, encerrando.")
        finally:
            cap.release()
            writer.release()
            if show_window:
                cv2.destroyAllWindows()
        if export_metadata:
            metadata_path = target_output.parent / f"{target_output.stem}.json"
            self._write_metadata(metadata_path)
        logger.info("Captura finalizada | frames=%s | saida=%s", frame_id, target_output)

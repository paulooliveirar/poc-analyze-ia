import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2

logger = logging.getLogger(__name__)


@dataclass
class Detector:
    """
    Wrapper para YOLOv8/YOLOv10 focado em COCO (person, sports ball).
    """

    model_path: str = "yolov8m.pt"
    conf_threshold: float = 0.4
    device: str = "cuda"  # será usado se disponível, senão cai para cpu

    def __post_init__(self):
        self.model = YOLO(self.model_path)

        # COCO ids relevantes
        self.person_id = 0
        self.ball_id = 32  # sports ball no COCO

    def detect(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections]:
        """
        Retorna detecções de jogadores e bola como `supervision.Detections`.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device if self.model.device.type == "cuda" else "cpu",
        )

        result = results[0]
        detections = sv.Detections.from_ultralytics(result)

        player_mask = detections.class_id == self.person_id
        ball_mask = detections.class_id == self.ball_id

        players = detections[player_mask]
        ball = detections[ball_mask]

        return players, ball


@dataclass
class PoseFieldDetector:
    """
    Detector de pose/campo usando modelo de pose (ex.: yolov8m-pose.pt).

    Usado para auxiliar o `FieldMapper` e o mapa 2D com keypoints do campo.
    """

    model_path: str = "yolov8m-pose.pt"
    conf_threshold: float = 0.4
    device: str = "cuda"

    def __post_init__(self):
        self.model = YOLO(self.model_path)

    def detect_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Retorna keypoints da pose/campo para o primeiro detecção do frame.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device if self.model.device.type == "cuda" else "cpu",
        )
        if not results:
            return None
        result = results[0]
        if result.keypoints is None:
            return None
        # shape (num_instances, num_keypoints, 2)
        kps = result.keypoints.xy
        if kps is None or len(kps) == 0:
            return None
        # Para o mapa 2D usamos apenas a primeira instância
        return kps[0].cpu().numpy()


@dataclass
class Tracker:
    """
    Tracker baseado em ByteTrack (via supervision).
    Mantém IDs consistentes para jogadores e bola.
    """

    track_players: sv.ByteTrack = field(default_factory=sv.ByteTrack)
    track_ball: sv.ByteTrack = field(default_factory=sv.ByteTrack)

    def update(
        self,
        player_dets: sv.Detections,
        ball_dets: sv.Detections,
    ) -> Tuple[sv.Detections, sv.Detections]:
        tracked_players = self.track_players.update_with_detections(player_dets)
        tracked_ball = self.track_ball.update_with_detections(ball_dets)
        return tracked_players, tracked_ball


@dataclass
class FieldMapper:
    """
    Responsável pela homografia e zonas de interesse (ROI).
    """

    # Pontos em coordenadas da imagem (x, y)
    src_points: Optional[np.ndarray] = None
    # Pontos correspondentes no plano do campo (ex.: dimensões normalizadas)
    dst_points: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None

    # Polígonos de escanteio no bird-eye (lista de np.ndarray de shape (N, 2))
    corner_rois: List[np.ndarray] = field(default_factory=list)

    # Últimos keypoints de pose/campo estimados pelo modelo de pose
    last_pose_keypoints: Optional[np.ndarray] = None

    def set_homography(self, src: List[Point], dst: List[Point]) -> None:
        self.src_points = np.array(src, dtype=np.float32)
        self.dst_points = np.array(dst, dtype=np.float32)
        self.homography, _ = cv2.findHomography(self.src_points, self.dst_points)

    def warp_point(self, p: Point) -> Optional[Point]:
        if self.homography is None:
            return None
        pts = np.array([[p]], dtype=np.float32)  # shape (1,1,2)
        warped = cv2.perspectiveTransform(pts, self.homography)[0, 0]
        return int(warped[0]), int(warped[1])

    def point_in_polygon(self, p: Point, poly: np.ndarray) -> bool:
        return cv2.pointPolygonTest(poly, p, False) >= 0

    def ball_in_corner_from_outside(
        self, ball_traj_bev: List[Point]
    ) -> bool:
        """
        Heurística simples para escanteios:
        - Trajetória recente da bola em coordenadas bird-eye.
        - Se último ponto está dentro de uma ROI de escanteio
          e pelo menos um ponto anterior estava fora, dispara.
        """
        if len(ball_traj_bev) < 2 or not self.corner_rois:
            return False

        last = ball_traj_bev[-1]
        prev = ball_traj_bev[:-1]

        for roi in self.corner_rois:
            inside_last = self.point_in_polygon(last, roi)
            if not inside_last:
                continue
            was_outside = any(
                not self.point_in_polygon(p, roi) for p in prev
            )
            if was_outside:
                return True
        return False

    def update_from_pose(self, keypoints: np.ndarray) -> None:
        """
        Atualiza os últimos keypoints de pose/campo.

        A lógica de conversão desses pontos em homografia ou zonas extras
        do mapa 2D pode ser feita em etapas futuras, usando estes pontos
        como base de calibragem dinâmica.
        """
        if keypoints is None or len(keypoints) == 0:
            return
        self.last_pose_keypoints = keypoints


@dataclass
class TeamClassifier:
    """
    Classificador de equipes baseado em K-Means nas cores do uniforme.

    Atribui um team_id a cada track_id de jogador e mantém esse mapeamento
    estável ao longo do tempo.
    """

    # track_id -> team_id
    team_by_player: Dict[int, int] = field(default_factory=dict)
    # cores de referência por time em BGR, usadas para calibragem fina (opcional)
    reference_colors_bgr: Dict[int, np.ndarray] = field(default_factory=dict)

    def _extract_jersey_patch(
        self, frame: np.ndarray, bbox: BBox
    ) -> Optional[np.ndarray]:
        """
        Recorta um patch do tórax/peito do jogador para estimar a cor do uniforme.
        Faz clamp nas coordenadas, exige tamanho mínimo e evita regiões fora da imagem.
        """
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Garantir coordenadas dentro da imagem
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img))

        if x2 <= x1 or y2 <= y1:
            return None

        h = y2 - y1
        # Focar na região de camisa (entre ~20% e 60% da altura do bbox)
        top = y1 + int(0.20 * h)
        bottom = y1 + int(0.60 * h)
        if top >= bottom:
            return None

        patch = frame[top:bottom, x1:x2]
        # Descartar patches muito pequenos (poucos pixels) que atrapalham o K-Means
        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            return None
        return patch

    def _dominant_color_kmeans(
        self, patch: np.ndarray, k: int = 2
    ) -> np.ndarray:
        data = patch.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data,
            K=k,
            bestLabels=None,
            criteria=criteria,
            attempts=3,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        counts = np.bincount(labels.flatten())
        dominant_idx = int(np.argmax(counts))
        return centers[dominant_idx].astype(np.uint8)

    def _assign_team_by_color(self, color: np.ndarray) -> int:
        """
        Atribui time pela cor dominante.

        Estratégia:
          - Se existirem cores de referência calibradas em `reference_colors_bgr`,
            escolhe o time com menor distância euclidiana em BGR.
          - Caso contrário, usa fallback simples por canal dominante (vermelho x azul).
        """
        bgr = color.astype(np.float32)

        if self.reference_colors_bgr:
            best_team = 2
            best_dist = float("inf")
            for team_id, ref in self.reference_colors_bgr.items():
                d = float(np.linalg.norm(bgr - ref.astype(np.float32)))
                if d < best_dist:
                    best_dist = d
                    best_team = team_id
            return int(best_team)

        # Fallback: heurística simples por canal dominante
        b, g, r = map(int, color)
        if r > g and r > b:
            return 0  # Time A (vermelho)
        if b > g and b > r:
            return 1  # Time B (azul)
        # Goleiro/árbitro ou cor mista: agrupar como time 2
        return 2

    def calibrate_team_color(self, team_id: int, patch: np.ndarray) -> None:
        """
        Atualiza a cor de referência de um time a partir de um patch de uniforme.

        Pode ser chamado com um recorte manual (ROI) ou com algum frame
        específico de calibragem antes do jogo.
        """
        if patch.size == 0:
            return
        dominant = self._dominant_color_kmeans(patch)
        self.reference_colors_bgr[team_id] = dominant

    def classify_players(
        self, frame: np.ndarray, tracked_players: sv.Detections
    ) -> Dict[int, int]:
        """
        Atualiza o dicionário team_by_player usando K-Means nas cores de uniforme.
        """
        for bbox, track_id in zip(
            tracked_players.xyxy, tracked_players.tracker_id
        ):
            if track_id is None:
                continue
            tid = int(track_id)
            if tid in self.team_by_player:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            patch = self._extract_jersey_patch(frame, (x1, y1, x2, y2))
            if patch is None:
                continue
            dominant = self._dominant_color_kmeans(patch)
            team_id = self._assign_team_by_color(dominant)
            self.team_by_player[tid] = team_id

        return self.team_by_player


@dataclass
class StatsAggregator:
    """
    Responsável por manter estatísticas em tempo real e eventos táticos.
    """

    possession_by_team: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    current_ball_owner: Optional[int] = None  # track_id do jogador
    last_ball_positions_img: List[Point] = field(default_factory=list)
    passes: int = 0
    shots: int = 0
    corners: int = 0

    def _bbox_bottom_center(self, bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = y2
        return cx, cy

    def update_possession_and_events(
        self,
        frame: np.ndarray,
        tracked_players: sv.Detections,
        tracked_ball: sv.Detections,
        fps: float,
        field_mapper: Optional[FieldMapper],
        team_by_player: Dict[int, int],
    ) -> None:
        """
        Atualiza:
        - posse de bola (usando homografia quando disponível)
        - passes certos/errados
        - roubadas de bola (tackle) em raio < 1.5m
        - chutes (bola acelerando para longe do dono)
        - escanteios (via FieldMapper)
        """
        # 1) Atualizar trajetória da bola em coordenadas de imagem
        if len(tracked_ball) > 0:
            ball_bbox = tracked_ball.xyxy[0]
            x1, y1, x2, y2 = map(int, ball_bbox)
            ball_center_img = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.last_ball_positions_img.append(ball_center_img)
            if len(self.last_ball_positions_img) > 20:
                self.last_ball_positions_img.pop(0)
        else:
            ball_center_img = None

        # 2) Calcular velocidade aproximada da bola em "unidades de homografia"
        ball_speed = 0.0
        if len(self.last_ball_positions_img) >= 2 and fps > 0:
            p1_img = self.last_ball_positions_img[-2]
            p2_img = self.last_ball_positions_img[-1]

            if field_mapper and field_mapper.homography is not None:
                p1_w = field_mapper.warp_point(p1_img)
                p2_w = field_mapper.warp_point(p2_img)
                if p1_w is not None and p2_w is not None:
                    dist = math.dist(p1_w, p2_w)  # metros (assumindo dst em metros)
                else:
                    dist = math.dist(p1_img, p2_img)
            else:
                dist = math.dist(p1_img, p2_img)

            ball_speed = dist * fps

        # 3) Determinar jogador mais próximo da bola em coordenadas de homografia
        new_owner: Optional[int] = self.current_ball_owner
        min_dist_m = float("inf")

        if ball_center_img is not None and len(tracked_players) > 0:
            if field_mapper and field_mapper.homography is not None:
                ball_center_world = field_mapper.warp_point(ball_center_img)
            else:
                ball_center_world = ball_center_img

            if ball_center_world is not None:
                for bbox, track_id in zip(
                    tracked_players.xyxy, tracked_players.tracker_id
                ):
                    if track_id is None:
                        continue
                    feet_img = self._bbox_bottom_center(tuple(map(int, bbox)))
                    if field_mapper and field_mapper.homography is not None:
                        feet_world = field_mapper.warp_point(feet_img)
                    else:
                        feet_world = feet_img
                    if feet_world is None:
                        continue

                    d = math.dist(feet_world, ball_center_world)
                    if d < min_dist_m:
                        min_dist_m = d
                        new_owner = int(track_id)

                # Limite de distância em metros para considerar posse
                if field_mapper and field_mapper.homography is not None:
                    if min_dist_m > 2.0:
                        new_owner = None
                else:
                    # fallback em pixels se não houver homografia
                    if min_dist_m > 80:
                        new_owner = None

        prev_owner = self.current_ball_owner
        prev_team = team_by_player.get(prev_owner, None) if prev_owner is not None else None
        new_team = team_by_player.get(new_owner, None) if new_owner is not None else None

        # 4) Eventos de passe/chute com base em mudança de dono + velocidade
        if prev_owner is not None and new_owner is not None and prev_owner != new_owner:
            # Passe certo x errado
            if prev_team is not None and new_team is not None:
                if prev_team == new_team:
                    self.passes += 1
                    print(f"PASSE CERTO: P{prev_owner} -> P{new_owner} (Time {new_team})")
                else:
                    self.passes += 1  # ainda contamos como passe
                    print(
                        f"PASSE ERRADO: P{prev_owner} (Time {prev_team}) -> "
                        f"P{new_owner} (Time {new_team})"
                    )

                    # Roubada de bola se proximidade < 1.5m em homografia
                    if field_mapper and field_mapper.homography is not None and min_dist_m < 1.5:
                        print(
                            f"ROUBADA DE BOLA: P{prev_owner} (Time {prev_team}) "
                            f"=> P{new_owner} (Time {new_team}), dist={min_dist_m:.2f}m"
                        )

        # 5) Atualizar chutes (bola se afastando rapidamente do mesmo dono)
        if (
            prev_owner is not None
            and new_owner is not None
            and prev_owner == new_owner
            and ball_speed > 25.0  # ~ chute em m/s (ajuste conforme necessário)
        ):
            self.shots += 1

        # Atualizar dono atual
        self.current_ball_owner = new_owner

        # 6) Atualizar posse agregada por time
        if self.current_ball_owner is not None and new_team is not None:
            self.possession_by_team[new_team] = self.possession_by_team.get(new_team, 0) + 1

        # 7) Escanteios via homografia + ROIs
        if (
            field_mapper is not None
            and ball_center_img is not None
            and field_mapper.homography is not None
        ):
            traj_bev: List[Point] = []
            for p_img in self.last_ball_positions_img:
                warped = field_mapper.warp_point(p_img)
                if warped is not None:
                    traj_bev.append(warped)
            if field_mapper.ball_in_corner_from_outside(traj_bev):
                self.corners += 1


@dataclass
class VideoProcessor:
    """
    Orquestra detecção, tracking, homografia e estatísticas,
    e faz o desenho final no frame.
    """

    detector: Detector
    # Opcional: modelo de pose/campo para auxiliar o FieldMapper e o mapa 2D
    pose_detector: Optional[PoseFieldDetector] = None
    tracker: Tracker = field(default_factory=Tracker)
    team_classifier: TeamClassifier = field(default_factory=TeamClassifier)
    stats: StatsAggregator = field(default_factory=StatsAggregator)
    field_mapper: FieldMapper = field(default_factory=FieldMapper)

    # Annotators da Supervision: caixas e rótulos separados (BoxAnnotator não aceita labels)
    box_annotator: sv.BoxAnnotator = field(
        default_factory=lambda: sv.BoxAnnotator()
    )
    label_annotator: sv.LabelAnnotator = field(
        default_factory=lambda: sv.LabelAnnotator()
    )

    def process_frame(self, frame: np.ndarray, fps: float) -> np.ndarray:
        # 1) Detecção
        player_dets, ball_dets = self.detector.detect(frame)

        # 1.1) Pose/campo para auxiliar o mapa 2D (se configurado)
        if self.pose_detector is not None:
            kps = self.pose_detector.detect_keypoints(frame)
            if kps is not None:
                self.field_mapper.update_from_pose(kps)

        # 2) Tracking
        tracked_players, tracked_ball = self.tracker.update(
            player_dets, ball_dets
        )

        # 3) Classificar times por cor via K-Means
        team_by_player = self.team_classifier.classify_players(frame, tracked_players)

        # 4) Atualizar posse, passes, chutes, escanteios
        self.stats.update_possession_and_events(
            frame,
            tracked_players,
            tracked_ball,
            fps,
            self.field_mapper,
            team_by_player,
        )

        # Log das detecções em estilo log (jogadores + bola)
        self._log_detections(tracked_players, tracked_ball, team_by_player)

        # 5) Desenhar retângulos e labels em tempo real (estilo versão anterior: cor por elemento + confiança)
        annotated = self._draw_detections_legacy_style(
            frame.copy(),
            tracked_players,
            tracked_ball,
            team_by_player,
        )

        # 6) Dashboard lateral com estatísticas e contagem de detecções
        annotated = self._draw_dashboard(
            annotated, team_by_player, len(tracked_players), len(tracked_ball)
        )

        return annotated

    def _log_detections(
        self,
        tracked_players: sv.Detections,
        tracked_ball: sv.Detections,
        team_by_player: Dict[int, int],
    ) -> None:
        """Registra no log as detecções do frame atual (estilo log estruturado)."""
        parts: List[str] = []
        conf_p = getattr(tracked_players, "confidence", None)
        conf_b = getattr(tracked_ball, "confidence", None)
        tracker_ids = (
            tracked_players.tracker_id
            if tracked_players.tracker_id is not None
            else [None] * len(tracked_players)
        )
        for i, (bbox, track_id) in enumerate(zip(tracked_players.xyxy, tracker_ids)):
            x1, y1, x2, y2 = map(int, bbox)
            team = team_by_player.get(int(track_id), 0) if track_id is not None else -1
            conf = f" {int(conf_p[i] * 100)}%" if conf_p is not None and i < len(conf_p) else ""
            id_label = f"#{int(track_id)}" if track_id is not None else "?"
            parts.append(f"Jogador {id_label} Time{team}{conf} bbox=({x1},{y1},{x2},{y2})")
        for i, bbox in enumerate(tracked_ball.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            conf = f" {int(conf_b[i] * 100)}%" if conf_b is not None and i < len(conf_b) else ""
            parts.append(f"bola{conf} bbox=({x1},{y1},{x2},{y2})")
        if parts:
            logger.info(
                "DETECÇÃO | jogadores=%d bola=%d | %s",
                len(tracked_players),
                len(tracked_ball),
                " | ".join(parts),
            )
        else:
            logger.debug("DETECÇÃO | jogadores=0 bola=0")

    # Cores para retângulos por elemento (BGR), estilo versão anterior do football_detector
    _COLOR_BOLA = (0, 165, 255)   # Laranja
    _COLOR_TIME_0 = (255, 0, 0)   # Azul
    _COLOR_TIME_1 = (0, 0, 255)   # Vermelho
    _COLOR_JOGADOR_NEUTRO = (128, 128, 128)  # Cinza quando time ainda não definido

    def _draw_detections_legacy_style(
        self,
        frame: np.ndarray,
        tracked_players: sv.Detections,
        tracked_ball: sv.Detections,
        team_by_player: Dict[int, int],
    ) -> np.ndarray:
        """
        Desenha retângulos coloridos e labels com confiança em tempo real,
        indicando claramente o que é cada elemento (jogador/time, bola).
        """
        conf_player = getattr(tracked_players, "confidence", None)
        conf_ball = getattr(tracked_ball, "confidence", None)
        tracker_ids = (
            tracked_players.tracker_id
            if tracked_players.tracker_id is not None
            else [None] * len(tracked_players)
        )

        # Jogadores: retângulo por time, label "Jogador #id | Time t XX%"
        for i, (bbox, track_id) in enumerate(zip(tracked_players.xyxy, tracker_ids)):
            x1, y1, x2, y2 = map(int, bbox)
            team = team_by_player.get(int(track_id), 0) if track_id is not None else 0
            if team == 0:
                box_color = self._COLOR_TIME_0
            elif team == 1:
                box_color = self._COLOR_TIME_1
            else:
                box_color = self._COLOR_JOGADOR_NEUTRO
            conf_pct = ""
            if conf_player is not None and i < len(conf_player):
                conf_pct = f" {int(conf_player[i] * 100)}%"
            label = (
                f"Jogador #{int(track_id)} | Time {team}{conf_pct}"
                if track_id is not None
                else f"Jogador{conf_pct}"
            )
            self._draw_box_and_label(frame, (x1, y1, x2, y2), label, box_color)

        # Bola: retângulo laranja, label "bola XX%"
        for i, bbox in enumerate(tracked_ball.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            conf_pct = ""
            if conf_ball is not None and i < len(conf_ball):
                conf_pct = f" {int(conf_ball[i] * 100)}%"
            label = f"bola{conf_pct}"
            self._draw_box_and_label(frame, (x1, y1, x2, y2), label, self._COLOR_BOLA)

        return frame

    def _draw_box_and_label(
        self,
        frame: np.ndarray,
        bbox: BBox,
        label: str,
        box_color: Tuple[int, int, int],
    ) -> None:
        """Desenha um retângulo e label com fundo colorido (estilo versão anterior)."""
        x1, y1, x2, y2 = bbox
        font_scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_y = max(y1 - 6, th + 4)
        pad_x, pad_y = 4, 2
        r1 = (x1 - pad_x, text_y - th - pad_y)
        r2 = (x1 + tw + pad_x, text_y + pad_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(frame, r1, r2, box_color, -1)
        cv2.rectangle(frame, r1, r2, (255, 255, 255), 1)
        text_color = (0, 0, 0) if sum(box_color) > 380 else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_dashboard(
        self,
        frame: np.ndarray,
        team_by_player: Optional[Dict[int, int]] = None,
        n_players: int = 0,
        n_ball: int = 0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_w = int(w * 0.22)
        x0 = w - panel_w

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x0, 0),
            (w, h),
            (15, 15, 15),
            thickness=-1,
        )
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        x = x0 + 10
        y = 30

        def put(text: str, dy: int = 28, color=(255, 255, 255)):
            nonlocal y
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
            y += dy

        put("STATS", dy=30, color=(0, 255, 255))

        # Detecções em tempo real (estilo versão anterior)
        put("DETECCOES:", dy=22, color=(255, 255, 255))
        put(f"Jogadores: {n_players}", color=(200, 200, 200))
        put(f"Bola: {n_ball}", color=self._COLOR_BOLA)
        put("", dy=16)

        # Contagem de jogadores por time (team_by_player: track_id -> team_id)
        team_counts = {0: 0, 1: 0}
        if team_by_player:
            for team in team_by_player.values():
                team_counts[team] = team_counts.get(team, 0) + 1

        put(f"Players T0: {team_counts.get(0, 0)}", color=(255, 200, 0))
        put(f"Players T1: {team_counts.get(1, 0)}", color=(0, 200, 255))

        # Posse de bola (normalizada)
        t0_pos = self.stats.possession_by_team.get(0, 0)
        t1_pos = self.stats.possession_by_team.get(1, 0)
        total_pos = t0_pos + t1_pos
        if total_pos > 0:
            t0_pct = int(100 * t0_pos / total_pos)
            t1_pct = 100 - t0_pct
        else:
            t0_pct = t1_pct = 0

        put(f"Posse T0: {t0_pct}%", color=(255, 200, 0))
        put(f"Posse T1: {t1_pct}%", color=(0, 200, 255))

        put(f"Passes: {self.stats.passes}", color=(180, 255, 180))
        put(f"Chutes: {self.stats.shots}", color=(180, 180, 255))
        put(f"Escanteios: {self.stats.corners}", color=(255, 180, 180))

        owner = (
            f"P{self.stats.current_ball_owner}"
            if self.stats.current_ball_owner is not None
            else "-"
        )
        put(f"Posse atual: {owner}", color=(255, 255, 255))

        return frame


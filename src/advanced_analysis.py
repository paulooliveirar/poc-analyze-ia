import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import logging
from pathlib import Path
import supervision as sv

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:  # pragma: no cover
    DeepSort = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AdvancedFootballAnalyzer:
    """
    Sistema avançado de análise de futebol com:
    - Rastreamento de jogadores e bola
    - Cálculo de posições
    - Heatmaps
    - Estatísticas de movimento
    """
    
    def __init__(self, tracking_backend="deepsort"):
        self.model = YOLO("runs/detect/football-players-detection/weights/best.pt")
        self.ball_trail = deque(maxlen=20)
        self.player_trails = {}
        self.conf_threshold = 0.65
        self.tracking_backend = tracking_backend.lower().strip()
        self.byte_track = sv.ByteTrack()
        self.deep_sort = DeepSort(max_age=30, n_init=2) if DeepSort else None
        self.last_logged_frame = 0
    
    def detect_objects(self, frame):
        """Detecta objetos no frame"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        return results

    def track_objects(self, frame, result):
        """Rastreia detecções com ByteTrack ou DeepSORT."""
        detections = []
        boxes = result.boxes
        names = result.names if hasattr(result, "names") else {}
        if boxes is None:
            return detections

        if self.tracking_backend == "deepsort" and self.deep_sort is not None:
            ds_inputs = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = str(names.get(cls_id, f"class_{cls_id}"))
                ds_inputs.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

            tracks = self.deep_sort.update_tracks(ds_inputs, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                class_name = str(getattr(track, "det_class", "obj"))
                conf = float(getattr(track, "det_conf", 0.0) or 0.0)
                detections.append(
                    {"bbox": (x1, y1, x2, y2), "conf": conf, "track_id": int(track.track_id), "class_name": class_name}
                )
            return detections

        sv_detections = sv.Detections.from_ultralytics(result)
        tracked = self.byte_track.update_with_detections(sv_detections)
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else [None] * len(tracked)
        for i, (bbox, track_id) in enumerate(zip(tracked.xyxy, tracker_ids)):
            x1, y1, x2, y2 = map(int, bbox)
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            class_name = str(names.get(cls_id, f"class_{cls_id}"))
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "track_id": int(track_id) if track_id is not None else -1,
                    "class_name": class_name,
                }
            )
        return detections
    
    def get_centroid(self, box):
        """Calcula o centroide de uma bounding box"""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (int(cx), int(cy))
    
    def calculate_distance(self, p1, p2):
        """Calcula distância euclidiana entre dois pontos"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def track_ball(self, tracked_detections):
        """Rastreia a bola através dos frames com melhor filtragem"""
        ball_detected = False

        for det in tracked_detections:
            class_name = str(det.get("class_name", "")).lower()
            if "ball" not in class_name and "bola" not in class_name:
                continue
            x1, y1, x2, y2 = det["bbox"]
            conf = float(det.get("conf", 0.0))
            if conf < 0.45:
                continue
            width = x2 - x1
            height = y2 - y1
            area = width * height
            if area > 2000:
                continue
            aspect_ratio = width / (height + 1e-5)
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                continue
            centroid = self.get_centroid((x1, y1, x2, y2))
            self.ball_trail.append(centroid)
            ball_detected = True

        return ball_detected
    
    def draw_ball_trail(self, frame):
        """Desenha o rastro da bola"""
        if len(self.ball_trail) > 1:
            points = list(self.ball_trail)
            for i in range(1, len(points)):
                # Gradiente de cor para mostrar movimento recente
                color = (0, 165, 255)  # Laranja
                cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame
    
    def draw_heatmap(self, frame, detections):
        """
        Cria um heatmap de atividade dos jogadores.
        """
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        for result in detections:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                heatmap[max(0, y1):min(frame.shape[0], y2), 
                        max(0, x1):min(frame.shape[1], x2)] += 1
        
        # Aplicar blur para suavizar
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        # Normalizar e garantir uint8 (requisito do applyColorMap)
        max_value = float(heatmap.max())
        if max_value > 0:
            heatmap = (heatmap / max_value * 255).astype(np.uint8)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.uint8)
        
        # Aplicar colormap
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Sobrepor na imagem original
        result = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
        
        return result
    
    def calculate_player_speed(self, player_id, current_pos, prev_pos, fps=30):
        """
        Calcula a velocidade aproximada do jogador (pixels por segundo).
        """
        if prev_pos is None:
            return 0
        
        distance = self.calculate_distance(current_pos, prev_pos)
        time_delta = 1.0 / fps
        speed = distance / time_delta
        
        return speed
    
    def draw_field_overlay(self, frame):
        """Desenha uma sobreposição de campo de futebol"""
        h, w = frame.shape[:2]
        
        # Desenhar linhas do campo
        color = (0, 255, 0)
        thickness = 2
        
        # Linhas horizontais (para linhas)
        cv2.line(frame, (0, h//2), (w, h//2), color, thickness)
        
        # Linhas verticais
        cv2.line(frame, (w//2, 0), (w//2, h), color, thickness)
        
        # Círculo central
        cv2.circle(frame, (w//2, h//2), 50, color, thickness)
        
        # Áreas de penalidade
        penalty_h = int(h * 0.2)
        cv2.rectangle(frame, (0, (h - penalty_h)//2), 
                     (int(w * 0.15), (h + penalty_h)//2), color, thickness)
        cv2.rectangle(frame, (w - int(w * 0.15), (h - penalty_h)//2), 
                     (w, (h + penalty_h)//2), color, thickness)
        
        return frame

    def _build_detection_summary(self, tracked_detections):
        """Monta resumo textual do que foi identificado no frame."""
        class_counts = {}
        detailed_labels = []
        for det in tracked_detections:
            class_name = str(det.get("class_name", "obj"))
            conf = float(det.get("conf", 0.0))
            track_id = int(det.get("track_id", -1))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            detailed_labels.append(f"{class_name} #{track_id} {conf:.2f}")
        return class_counts, detailed_labels

    def _log_detections(self, frame_count, tracked_detections, class_counts, detailed_labels):
        """Registra em log o que foi identificado em tempo real."""
        if not tracked_detections:
            logger.info("Frame %s | nenhum objeto identificado", frame_count)
            return
        counts_text = ", ".join(
            f"{class_name}:{count}" for class_name, count in sorted(class_counts.items())
        )
        details_text = " | ".join(detailed_labels[:8])
        if len(detailed_labels) > 8:
            details_text = f"{details_text} | ..."
        logger.info(
            "Frame %s | total=%s | classes=[%s] | detalhes=%s",
            frame_count,
            len(tracked_detections),
            counts_text,
            details_text,
        )

    def _draw_detection_log_on_screen(self, frame, frame_count, total_frames, class_counts, detailed_labels):
        """Desenha na tela um painel de log com os objetos identificados."""
        overlay = frame.copy()
        panel_height = min(250, 70 + 22 * max(1, len(detailed_labels[:6])))
        cv2.rectangle(overlay, (8, 8), (690, panel_height), (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        cv2.putText(
            frame,
            f"Frame: {frame_count}/{total_frames}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        if class_counts:
            counts_text = " | ".join(
                f"{class_name}:{count}" for class_name, count in sorted(class_counts.items())
            )
        else:
            counts_text = "Nenhum objeto identificado"
        cv2.putText(
            frame,
            counts_text,
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )

        y = 88
        for label in detailed_labels[:6]:
            cv2.putText(
                frame,
                f"- {label}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180),
                1,
            )
            y += 22
        return frame
    
    def process_video_advanced(self, video_path, output_path=None, display=False):
        """
        Processa vídeo com análise avançada.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Erro: Não consegui abrir o vídeo: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processando com análise avançada...")
        print(f"Resolução: {width}x{height} | FPS: {fps}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detecção
                results = self.detect_objects(frame)
                tracked_detections = self.track_objects(frame, results[0])
                
                # Rastreamento de bola
                self.track_ball(tracked_detections)
                
                # Desenhar detecções
                for det in tracked_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    conf = float(det.get("conf", 0.0))
                    track_id = int(det.get("track_id", -1))
                    class_name = str(det.get("class_name", "obj"))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{class_name} #{track_id} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                class_counts, detailed_labels = self._build_detection_summary(tracked_detections)
                self._log_detections(frame_count, tracked_detections, class_counts, detailed_labels)
                
                # Desenhar rastro da bola
                frame = self.draw_ball_trail(frame)
                
                # Desenhar overlay do campo
                frame = self.draw_field_overlay(frame)
                
                # Heatmap
                frame = self.draw_heatmap(frame, results)
                
                # Informações e log visual do que foi identificado
                frame = self._draw_detection_log_on_screen(
                    frame,
                    frame_count,
                    total_frames,
                    class_counts,
                    detailed_labels,
                )
                
                if writer:
                    writer.write(frame)
                
                if display:
                    cv2.imshow("Analise Avancada", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    videos_dir = base_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AdvancedFootballAnalyzer()
    
    video_file = videos_dir / "bundesliga--matchday-20---all-highlights.mp4"
    output_file = videos_dir / "analise_avancada.mp4"
    
    analyzer.process_video_advanced(str(video_file), str(output_file))

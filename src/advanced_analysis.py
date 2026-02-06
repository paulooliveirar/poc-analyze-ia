"""
Análise avançada de partidas de futebol com rastreamento de jogadores e bola.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math


class AdvancedFootballAnalyzer:
    """
    Sistema avançado de análise de futebol com:
    - Rastreamento de jogadores e bola
    - Cálculo de posições
    - Heatmaps
    - Estatísticas de movimento
    """
    
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.ball_trail = deque(maxlen=20)
        self.player_trails = {}
        self.conf_threshold = 0.45
    
    def detect_objects(self, frame):
        """Detecta objetos no frame"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        return results
    
    def get_centroid(self, box):
        """Calcula o centroide de uma bounding box"""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (int(cx), int(cy))
    
    def calculate_distance(self, p1, p2):
        """Calcula distância euclidiana entre dois pontos"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def track_ball(self, results, frame):
        """Rastreia a bola através dos frames com melhor filtragem"""
        ball_detected = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Assumir que classe 0 é bola
                if cls_id != 0:
                    continue
                
                # Filtro 1: Confiança alta
                if conf < 0.65:
                    continue
                
                # Filtro 2: Tamanho apropriado (bola é pequena)
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Bola deve ser menor que 2000 pixels
                if area > 2000:
                    continue
                
                # Filtro 3: Proporção aproximadamente quadrada
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
                alpha = int(255 * (i / len(points)))
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
        
        # Normalizar
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
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
    
    def process_video_advanced(self, video_path, output_path=None):
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
                
                # Rastreamento de bola
                self.track_ball(results, frame)
                
                # Desenhar detecções
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Desenhar rastro da bola
                frame = self.draw_ball_trail(frame)
                
                # Desenhar overlay do campo
                frame = self.draw_field_overlay(frame)
                
                # Heatmap
                # frame = self.draw_heatmap(frame, results)
                
                # Informações
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if writer:
                    writer.write(frame)
                
                cv2.imshow("Analise Avancada", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    analyzer = AdvancedFootballAnalyzer()
    
    video_file = "/home/paulo/Documentos/novo_projeto/poc-analyze-ia/videos/exemplo_futebol.mp4"
    output_file = "/home/paulo/Documentos/novo_projeto/poc-analyze-ia/videos/analise_avancada.mp4"
    
    analyzer.process_video_advanced(video_file, output_file)

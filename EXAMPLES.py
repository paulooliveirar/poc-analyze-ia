"""
Exemplos avançados de uso do sistema de detecção de futebol.
Copie e adapte conforme suas necessidades.
"""

# ==========================================
# EXEMPLO 1: Detecção Simples com Câmera
# ==========================================
def exemplo_camera_basico():
    """Capturar de câmera e mostrar em tempo real"""
    from src.football_detector import FootballDetector
    
    detector = FootballDetector(
        model_path="yolov8n.pt",
        conf_threshold=0.45
    )
    
    detector.process_camera(camera_id=0)


# ==========================================
# EXEMPLO 2: Processamento de Vídeo
# ==========================================
def exemplo_video_basico():
    """Processar vídeo e salvar resultado"""
    from src.football_detector import FootballDetector
    
    detector = FootballDetector()
    
    detector.process_video(
        video_path="matches/game_2026_01_15.mp4",
        output_path="results/game_detected.mp4",
        display=True  # Mostrar enquanto processa
    )


# ==========================================
# EXEMPLO 3: Customizar Cores
# ==========================================
def exemplo_cores_customizadas():
    """Usar cores personalizadas nas detecções"""
    from src.football_detector import FootballDetector
    
    detector = FootballDetector()
    
    # Customizar cores (BGR)
    detector.COLORS = {
        "bola": (0, 255, 255),              # Amarelo
        "jogador_time_1": (200, 100, 50),   # Azul escuro
        "jogador_time_2": (50, 100, 200),   # Vermelho escuro
        "arbitro": (0, 255, 0),             # Verde
        "goleiro": (255, 100, 200),         # Rosa
        "bandeirinha": (255, 200, 0)        # Ciano
    }
    
    detector.process_camera(camera_id=0)


# ==========================================
# EXEMPLO 4: Loop Manual com Controle
# ==========================================
def exemplo_loop_manual():
    """Controle total sobre o processamento"""
    import cv2
    from src.football_detector import FootballDetector
    
    detector = FootballDetector(conf_threshold=0.5)
    
    # Abrir vídeo
    cap = cv2.VideoCapture("video.mp4")
    
    # Informações do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Vídeo: {width}x{height} @ {fps} fps")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # AQUI: Sua lógica customizada
        
        # Detecção
        results = detector.detect(frame)
        
        # Processar resultados customizado
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                # Fazer algo com as coordenadas
                print(f"Frame {frame_count}: {x1},{y1},{x2},{y2} (conf: {conf:.2f})")
        
        # Renderizar
        frame = detector.draw_detections(frame, results)
        
        # Exibir
        cv2.imshow("Resultado", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# EXEMPLO 5: Análise Avançada
# ==========================================
def exemplo_analise_avancada():
    """Usar análise avançada com rastreamento"""
    from src.advanced_analysis import AdvancedFootballAnalyzer
    
    analyzer = AdvancedFootballAnalyzer(model_path="yolo26m.pt")
    
    analyzer.process_video_advanced(
        video_path="partida.mp4",
        output_path="analise_completa.mp4"
    )


# ==========================================
# EXEMPLO 6: Filtrar por Classe
# ==========================================
def exemplo_filtrar_classes():
    """Detectar apenas a bola"""
    import cv2
    from src.football_detector import FootballDetector
    
    detector = FootballDetector()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector.detect(frame)
        
        # Filtrar apenas bola
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                
                # 0 = bola
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Desenhar apenas a bola
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        cv2.imshow("Apenas Bola", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# EXEMPLO 7: Contar Objetos
# ==========================================
def exemplo_contar_objetos():
    """Contar número de cada tipo de objeto"""
    import cv2
    from src.football_detector import FootballDetector
    
    detector = FootballDetector()
    
    cap = cv2.VideoCapture("video.mp4")
    
    contagens = {
        "bola": 0,
        "jogador_time_1": 0,
        "jogador_time_2": 0,
        "arbitro": 0,
        "goleiro": 0,
        "bandeirinha": 0
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector.detect(frame)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = detector.FOOTBALL_CLASSES.get(cls_id, "unknown")
                contagens[class_name] += 1
        
        # Mostrar contagens a cada 30 frames
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
            print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}:")
            for classe, count in contagens.items():
                print(f"  {classe}: {count}")
    
    cap.release()


# ==========================================
# EXEMPLO 8: Diferentes Modelos
# ==========================================
def exemplo_comparar_modelos():
    """Comparar velocidade de diferentes modelos"""
    import cv2
    import time
    from src.football_detector import FootballDetector
    
    modelos = ["yolov8n.pt", "yolov8s.pt", "yolo26m.pt"]
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    for modelo in modelos:
        print(f"\nTestando {modelo}...")
        detector = FootballDetector(model_path=modelo)
        
        times = []
        for _ in range(10):
            inicio = time.time()
            detector.detect(frame)
            tempo = time.time() - inicio
            times.append(tempo)
        
        tempo_medio = sum(times) / len(times)
        fps = 1 / tempo_medio
        
        print(f"  Tempo médio: {tempo_medio*1000:.1f}ms")
        print(f"  FPS: {fps:.1f}")


# ==========================================
# EXEMPLO 9: Salvar Frames com Detecções
# ==========================================
def exemplo_salvar_frames():
    """Salvar frames com detecções acima de confiança"""
    import cv2
    import os
    from pathlib import Path
    from src.football_detector import FootballDetector
    
    detector = FootballDetector(conf_threshold=0.7)
    
    output_dir = Path("frames_detectados")
    output_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture("video.mp4")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        results = detector.detect(frame)
        
        # Se temos detecções, salvar
        if len(results[0].boxes) > 0:
            frame = detector.draw_detections(frame, results)
            
            filename = output_dir / f"frame_{frame_count:05d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Salvo: {filename}")
    
    cap.release()


# ==========================================
# EXEMPLO 10: Medir Distâncias
# ==========================================
def exemplo_medir_distancia():
    """Calcular distância entre objetos detectados"""
    import cv2
    import math
    from src.football_detector import FootballDetector
    
    detector = FootballDetector()
    
    def distancia(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def centroid(x1, y1, x2, y2):
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector.detect(frame)
        
        centroids = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                c = centroid(x1, y1, x2, y2)
                centroids.append(c)
        
        # Calcular distâncias
        if len(centroids) >= 2:
            dist = distancia(centroids[0], centroids[1])
            print(f"Distância: {dist:.1f} pixels")
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# EXECUTAR EXEMPLOS
# ==========================================

if __name__ == "__main__":
    print("Exemplos disponíveis:")
    print("1. exemplo_camera_basico()")
    print("2. exemplo_video_basico()")
    print("3. exemplo_cores_customizadas()")
    print("4. exemplo_loop_manual()")
    print("5. exemplo_analise_avancada()")
    print("6. exemplo_filtrar_classes()")
    print("7. exemplo_contar_objetos()")
    print("8. exemplo_comparar_modelos()")
    print("9. exemplo_salvar_frames()")
    print("10. exemplo_medir_distancia()")
    print("\nCopie e execute no seu script!")

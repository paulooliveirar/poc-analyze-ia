import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from roboflow import Roboflow


class FootballDetector:
    """
    Sistema de detecção de elementos em partidas de futebol usando YOLOv11.
    Detecta: jogadores, bola, árbitro e outros elementos.
    """

    # Definir classes customizadas para futebol
    FOOTBALL_CLASSES = {
        32: "bola",
        6: "jogador_time_1",
        7: "jogador_time_2",
        3: "arbitro",
        4: "goleiro",
        5: "bandeirinha",
        77: "bola_de_futebol"  # Classe extra para bola (se existir no modelo)
    }

    # Cores para visualização (BGR)
    COLORS = {
        "bola": (0, 165, 255),  # Laranja
        "jogador": (255, 0, 0),  # Azul
        "jogador_time_2": (0, 0, 255),  # Vermelho
        "arbitro": (0, 255, 0),  # Verde
        "goleiro": (255, 255, 0),  # Ciano
        "bandeirinha": (255, 0, 255)  # Magenta
    }
    
    # Cores para texto dos labels (BGR)
    TEXT_COLORS = {
        "bola": (255, 255, 255),  # Branco
        "jogador": (255, 255, 255),  # Branco
        "jogador_time_2": (255, 255, 255),  # Branco
        "arbitro": (0, 0, 0),  # Preto
        "goleiro": (0, 0, 0),  # Preto
        "bandeirinha": (255, 255, 255)  # Branco
    }

    def __init__(self, model_path="yolo26m.pt", conf_threshold=0.60):
        """
        Inicializa o detector de futebol.
        
        Args:
            model_path: Caminho para o modelo YOLOv11 (padrão: yolo26m.pt)
            conf_threshold: Limiar de confiança para detecções (0-1)
        """
        self.conf_threshold = conf_threshold
        
        # Carregar modelo YOLOv11
        print(f"Carregando modelo: {model_path}")
        self.model = YOLO(model_path)
        print("Modelo carregado com sucesso!")

        # Treinar com seu dataset 
        self.model.train(data='/home/paulo/Documentos/novo_projeto/poc-analyze-ia/Soccer-1/data.yaml', epochs=6, imgsz=416, batch=4, device=0)

    def detect(self, frame):
        """
        Executa detecção de objetos em um frame.
        
        Args:
            frame: Frame da imagem (numpy array)
            
        Returns:
            results: Resultados da detecção do YOLOv11
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False, iou=1)
        return results
    
    def filter_detections(self, results):
        """
        Filtra e corrige detecções para evitar falsos positivos.
        Remove jogadores detectados incorretamente como bola.
        Equilibra precisão para bola e jogadores.
        
        Args:
            results: Resultados da detecção
            
        Returns:
            results filtrados
        """
        for result in results:
            boxes = result.boxes
            names = result.names
            valid_indices = []
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls_id = box.cls[0].cpu().numpy()
                class_name = self.FOOTBALL_CLASSES.get(int(cls_id), f"classe_{int(cls_id)}")
                print(f"{names[int(cls_id)]} detectado com confiança {conf:.2f} - Classe: {class_name}")

                # Calcular propriedades da bounding box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / (height + 1e-5)
                
                # # BOLA (classe 0) - Filtros mais precisos
                # if class_name == "bola":
                #     print(f"Validando bola de futebol detectado com confiança {conf:.2f} - Área: {area} - Aspect Ratio: {aspect_ratio:.2f} - Altura: {height}")
                #     # Bola precisa de boa confiança (0.70 equilibra bem)
                #     if conf < 0.70:
                #         continue
                    
                #     # Bola é pequena (< 1800 pixels - um pouco mais flexível)
                #     if area > 1800:
                #         continue
                    
                #     # Bola é aproximadamente redonda (aspect ratio perto de 1)
                #     # Mais tolerante para capturar melhor em ângulos
                #     if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                #         continue
                    
                #     # Bola não pode ser muito grande em altura (< 90 pixels)
                #     if height > 90:
                #         continue
                    
                #     # Bola não pode ser muito pequena (> 15 pixels)
                #     if height < 15 or width < 15:
                #         continue
                
                # # JOGADORES (classes 1, 2, 4) - Melhor para detectar jogadores
                # elif class_name in ["jogador", "jogador_time_1", "jogador_time_2", "goleiro"]:
                #     # Jogadores com confiança razoável
                #     if conf < 0.52:
                #         continue
                    
                #     # Jogadores podem ter tamanho variável
                #     # Mínimo: 1800 pixels | Máximo: 500000 pixels
                #     if area < 1800 or area > 500000:
                #         continue
                    
                #     # Jogadores são mais altos que largos (proporção alongada)
                #     # Ser mais flexível: aceitar até 1.0 de proporção
                #     if aspect_ratio > 1.0:
                #         continue
                    
                #     # Altura mínima para ser considerado jogador (120 pixels)
                #     if height < 20:
                #         continue
                
                # # ÁRBITRO (classe 3)
                # elif class_name == "arbitro":
                #     if conf < 0.48:
                #         continue
                #     if area < 1500 or area > 40000:
                #         continue
                #     if height < 100:
                #         continue
                
                # # BANDEIRINHA (classe 5)
                # elif class_name == "bandeirinha":
                #     if conf < 0.48:
                #         continue
                #     if area < 800 or area > 10000:
                #         continue
                
                # # GOLEIRO (classe 4) - já incluso na lista de jogadores
                # # mas com validações adicionais acima
                
                valid_indices.append(idx)
            
            # Manter apenas as detecções válidas
            if len(valid_indices) < len(boxes):
                result.boxes = result.boxes[valid_indices]
        
        return results

    def draw_detections(self, frame, results, show_stats=True):
        """
        Desenha as detecções no frame com labels coloridos e percentual.
        
        Args:
            frame: Frame original
            results: Resultados da detecção
            show_stats: Se True, mostra estatísticas no frame
            
        Returns:
            frame com anotações
        """
        detections_count = {}
        
        # Aplicar filtros
        results = self.filter_detections(results)
        
        # Processar cada detecção
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extrair coordenadas e confiança
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())

                # Obter nome da classe
                class_name = self.FOOTBALL_CLASSES.get(cls_id, f"classe_{cls_id}")
                
                # Contar detecções por classe
                detections_count[class_name] = detections_count.get(class_name, 0) + 1
                
                # Selecionar cor da caixa
                box_color = self.COLORS.get(class_name, (255, 255, 255))
                print(f"Desenhando {class_name} com confiança {conf:.2f} - Caixa: ({x1}, {y1}), ({x2}, {y2})")
                
                # Desenhar bounding box com cor (espessura aumentada para melhor visibilidade)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                
                # Adicionar contorno de sombra preta para melhor contraste
                # cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 1)
                confidence_percent = int(conf * 100)
                label = f"{class_name} {confidence_percent}%"
                
                # Obter cor do texto
                text_color = self.TEXT_COLORS.get(class_name, (255, 255, 255))
                
                # Calcular tamanho do texto
                font_scale = 0.7
                thickness = 1
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Posição do texto (acima da caixa)
                text_x = x1
                text_y = y1 - 10
                
                # Desenhar retângulo de fundo para o texto COM A COR DA CLASSE
                padding_x = 5
                padding_y = 3
                rect_pt1 = (text_x - padding_x, text_y - text_size[1] - padding_y)
                rect_pt2 = (text_x + text_size[0] + padding_x, text_y + padding_y)
                cv2.rectangle(frame, rect_pt1, rect_pt2, box_color, -1)
                
                # Desenhar borda branca ao redor do label para melhor contraste
                cv2.rectangle(frame, rect_pt1, rect_pt2, (255, 255, 255), 1)
                
                # Desenhar texto em contraste
                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    thickness
                )
        
        # Desenhar estatísticas se solicitado (sem afetar as detecções)
        if show_stats and detections_count:
            y_offset = 30
            
            # Texto "Detecções:" com fundo simples
            cv2.putText(frame, "DETECCOES:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            for class_name, count in detections_count.items():
                y_offset += 25
                color = self.COLORS.get(class_name, (255, 255, 255))
                cv2.putText(
                    frame,
                    f"{class_name}: {count}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
        
        return frame

    def process_video(self, video_path, output_path=None, display=True):
        """
        Processa um vídeo e detecta elementos da partida.
        
        Args:
            video_path: Caminho do arquivo de vídeo
            output_path: Caminho para salvar vídeo processado (opcional)
            display: Se True, exibe o vídeo em tempo real
        """
        # Abrir vídeo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Erro: Não consegui abrir o vídeo: {video_path}")
            return
        
        # Obter informações do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessando vídeo: {video_path}")
        print(f"Resolução: {width}x{height} | FPS: {fps} | Total de frames: {total_frames}")
        
        # Preparar writer para salvar vídeo
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Salvando em: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Executar detecção
                results = self.detect(frame)
                
                # Desenhar detecções
                frame_annotated = self.draw_detections(frame, results)
                
                # Adicionar informação de frame
                cv2.putText(
                    frame_annotated,
                    f"Frame: {frame_count}/{total_frames}",
                    (width - 250, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Salvar frame no vídeo de saída
                if writer:
                    writer.write(frame_annotated)
                
                # Exibir frame
                if display:
                    cv2.imshow("Deteccao de Futebol", frame_annotated)
                    
                    # Pressionar 'q' para parar
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processamento interrompido pelo usuário.")
                        break
                
                # Mostrar progresso
                if frame_count % 30 == 0:
                    print(f"Processado: {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"Processamento concluído! {frame_count} frames processados.")

    def process_camera(self, camera_id=0, output_path=None):
        """
        Processa stream de câmera em tempo real.
        
        Args:
            camera_id: ID da câmera (padrão: 0 para câmera padrão)
            output_path: Caminho para salvar vídeo (opcional)
        """
        # Abrir câmera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Erro: Não consegui abrir a câmera {camera_id}")
            return
        
        # Configurar resolução
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nCâmera iniciada - Resolução: {width}x{height} | FPS: {fps}")
        print("Pressione 'q' para parar")
        
        # Preparar writer se output foi especificado
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Salvando em: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Erro ao capturar frame")
                    break
                
                frame_count += 1
                
                # Executar detecção
                results = self.detect(frame)
                
                # Desenhar detecções
                frame_annotated = self.draw_detections(frame, results)
                
                # Adicionar informação de frame
                cv2.putText(
                    frame_annotated,
                    f"Frames: {frame_count}",
                    (width - 200, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Salvar frame
                if writer:
                    writer.write(frame_annotated)
                
                # Exibir frame
                cv2.imshow("Deteccao de Futebol - Camara", frame_annotated)
                
                # Pressionar 'q' para parar
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Captura interrompida pelo usuário.")
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"Captura concluída! {frame_count} frames capturados.")


def main():
    """Função principal com exemplos de uso"""
    
    # Inicializar detector
    detector = FootballDetector(
        model_path="yolo26m.pt",  # Usar nano model (mais rápido)
        conf_threshold=0.45
    )
    
    # Exemplo 1: Processar vídeo
    video_file = "/home/paulo/Documentos/novo_projeto/poc-analyze-ia/videos/exemplo_futebol.mp4"
    
    if os.path.exists(video_file):
        output_file = "/home/paulo/Documentos/novo_projeto/poc-analyze-ia/videos/resultado_deteccao.mp4"
        detector.process_video(
            video_path=video_file,
            output_path=output_file,
            display=True
        )
    else:
        print(f"Arquivo de vídeo não encontrado: {video_file}")
        print("Use a câmera ao invés...")
        
        # Exemplo 2: Usar câmera em tempo real
        print("\nIniciando câmera em tempo real...")
        detector.process_camera(
            camera_id=0,
            output_path="/home/paulo/Documentos/novo_projeto/poc-analyze-ia/videos/captura_camera.mp4"
        )


if __name__ == "__main__":
    main()

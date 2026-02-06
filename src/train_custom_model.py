"""
Script para treinar um modelo YOLOv8 customizado especificamente para detecção de futebol.

Datasets recomendados:
- Roboflow Universe: Soccer/Football datasets
- Dataset próprio anotado no Roboflow
"""

from ultralytics import YOLO
import os


def train_football_model(dataset_path=None, epochs=100, imgsz=640, batch_size=16):
    """
    Treina um modelo YOLOv8 customizado para futebol.
    
    Args:
        dataset_path: Caminho para dataset em formato YOLO
        epochs: Número de épocas de treinamento
        imgsz: Tamanho das imagens
        batch_size: Tamanho do batch
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
    model = YOLO('yolo26m.pt')
    
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
        patience=20,  # Early stopping
        save=True,
        device=0,  # GPU id (0) ou 'cpu'
        project='runs/detect',
        name='football_detector',
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
    )
    
    print("\nTreinamento concluído!")
    print(f"Melhor modelo salvo em: runs/detect/football_detector/weights/best.pt")
    
    return results


def convert_to_onnx(model_path='runs/detect/football_detector/weights/best.pt'):
    """
    Converte o modelo treinado para ONNX (para inferência mais rápida).
    
    Args:
        model_path: Caminho do modelo YOLOv8 treinado
    """
    model = YOLO(model_path)
    model.export(format='onnx')
    print("Modelo convertido para ONNX!")


def evaluate_model(model_path, dataset_path):
    """
    Avalia o desempenho do modelo no conjunto de validação.
    
    Args:
        model_path: Caminho do modelo
        dataset_path: Caminho do dataset (data.yaml)
    """
    model = YOLO(model_path)
    metrics = model.val(data=dataset_path)
    print(metrics)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        train_football_model(dataset_path=dataset_path)
    else:
        # Mostrar instruções
        train_football_model()

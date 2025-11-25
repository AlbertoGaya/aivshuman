!pip install torch torchvision
!pip install -U ultralytics


from ultralytics import YOLO

# Carga el modelo YOLOv8s preentrenado
model = YOLO('yolo11m.pt')

# Entrena el modelo con la configuración ajustada
results = model.train(
    data='label1.yaml',
    epochs=600,
    patience=50,
    batch=16,
    imgsz=640,
    single_cls=True,
    lr0=0.002,
    lrf=0.001,
    optimizer='AdamW',
    cos_lr=True,
    augment=True,
    mixup=0.2,
    translate=0.2,
    scale=0.6,
    erasing=0.5
)
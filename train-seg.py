!pip install torch torchvision
!pip install -U ultralytics

!yolo segment train data=label.yaml model=yolo11s-seg.pt epochs=500 imgsz=640 single_cls=True  lr0=0.001 lrf=0.001 batch=32 mask_ratio=2 scale=0.8
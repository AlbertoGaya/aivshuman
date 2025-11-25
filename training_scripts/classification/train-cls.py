!nvidia-smi
!pip install torch torchvision
!pip install -U ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

!yolo classify train data=datasets model=yolo11m-cls.pt epochs=500 imgsz=640
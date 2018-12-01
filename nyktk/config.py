import os

YOLO_CONFIGS = {
    'YOLOv3-416': {
        'config': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'weight': 'https://pjreddie.com/media/files/yolov3.weights',
        'input_shape': (416, 416,)
    }
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

WEIGHT_DIR = os.path.join(DATA_DIR, 'weights')
os.makedirs(WEIGHT_DIR, exist_ok=True)

PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

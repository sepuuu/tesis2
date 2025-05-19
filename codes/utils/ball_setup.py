import numpy as np
import supervision as sv
from configs.ball import BallTracker, BallAnnotator
from ultralytics import YOLO

BALL_DETECTION_MODEL = YOLO("codes/models/ball.onnx")

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = BALL_DETECTION_MODEL(image_slice, imgsz=1024, conf=0.71)[0]
    return sv.Detections.from_ultralytics(result)

ball_tracker = BallTracker(buffer_size=20)
ball_annotator = BallAnnotator(max_radius=15, buffer_size=20)

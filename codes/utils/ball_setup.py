import os

import numpy as np
import supervision as sv
from configs.ball import BallTracker, BallAnnotator
from ultralytics import YOLO
import config

try:
    import onnxruntime
except ImportError as exc:
    raise ImportError(
        "No se pudo importar `onnxruntime` (fallo al cargar DLL). "
        "Causas comunes: falta el 'Microsoft Visual C++ 2015-2022 Redistributable (x64)' "
        "o hay conflicto entre `onnxruntime` y `onnxruntime-gpu` instalados a la vez. "
        "Sugerencia: desinstala ambos y reinstala solo uno (CPU: `pip install onnxruntime`; "
        "GPU: `pip install onnxruntime-gpu` + CUDA/TensorRT compatibles)."
    ) from exc

BALL_MODEL_PATH = config.PATHS.ball_detector_path
if not os.path.exists(BALL_MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontro el modelo del balon en: {BALL_MODEL_PATH}"
    )

BALL_DETECTION_MODEL = YOLO(BALL_MODEL_PATH, task="detect")

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = BALL_DETECTION_MODEL(image_slice, imgsz=1024, conf=0.71)[0]
    return sv.Detections.from_ultralytics(result)

ball_tracker = BallTracker(buffer_size=20)
ball_annotator = BallAnnotator(max_radius=15, buffer_size=20)

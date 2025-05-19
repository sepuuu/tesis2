import os
import cv2
import supervision as sv
from utils.drawing_utils import get_video_frames_generator


def train_team_classifier(video_path, model, fps):
    print("[INFO] Recolectando crops para TeamClassifier...")
    stride = int(fps // 7)
    crops = []
    from ultralytics import YOLO
    PLAYER_DETECTION_MODEL = YOLO("codes\models\players.onnx")


    for idx, fr in enumerate(get_video_frames_generator(video_path)):
        if idx % stride:
            continue
        res = PLAYER_DETECTION_MODEL.predict(fr, imgsz=1792, iou=0.7)[0]
        det = sv.Detections.from_ultralytics(res)
        players = det[det.class_id == 1]
        players = players.with_nms(threshold=0.5, class_agnostic=True)
        for xyxy in players.xyxy:
            crop = model._crop(fr, xyxy)
            v_mean = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[..., 2].mean()
            if not (60 < v_mean < 240):
                continue
            if crop.size > 0:
                crops.append(crop)
                if len(crops) <= 20:
                    os.makedirs("debug/train_crops", exist_ok=True)
                    outpath = f"debug/train_crops/crop_{len(crops):02}.jpg"
                    cv2.imwrite(outpath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print(f"[INFO] {len(crops)} crops recolectados")
    model.fit(crops)
    print("[INFO] TeamClassifier entrenado")

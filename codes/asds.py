# make_pseudolabels.py
import os, random
import numpy as np
import cv2
from decord import VideoReader, cpu
from ultralytics import YOLO
import supervision as sv
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"
# ------------------ CONFIG ------------------
VIDEOS = [
    ("codes/inputs/video_blancos_corto.mp4", "cam1"),
    ("codes/inputs/video_negros_corto.mp4",  "cam2"),
]
# muestreo de frames (sube/baja para más/menos datos)
FRAME_STRIDE = 10

# modelos ONNX existentes
PLAYER_MODEL_PATH = "codes/models/players.onnx"  # clase player=1 en tu modelo
BALL_MODEL_PATH   = "codes/models/ball.onnx"     # clase ball=0 en tu modelo

# slicing jugadores
PLAYER_SLICE_WH   = (960, 960)
PLAYER_OVERLAP_WH = (160, 160)
PLAYER_CONF = 0.25
PLAYER_IOU  = 0.60
PLAYER_FUSE_NMS = 0.60

# slicing balón
BALL_SLICE_WH   = (640, 640)
BALL_OVERLAP_WH = (0, 0)         # como lo usas hoy
BALL_CONF = 0.50
BALL_IOU  = 0.50
BALL_FUSE_NMS = 0.50

# guardar imágenes sin detección?
KEEP_EMPTY = False

# splits
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
random.seed(0)

# (opcional) ROI por cámara para recortar fuera de cancha (reduce falsos)
# usa tus puntos ya medidos (polígono grande de cancha por cámara)
ROI_IMG_POINTS = {
    "cam1": np.array([
        [  42, 768],[ 149, 369],[604, 499],[912, 203],[1358,265],[1180,237],[ 898,270]
    ], dtype=np.int32),
    "cam2": np.array([
        [256, 829],[451, 506],[811, 596],[1120, 370],[1497, 360],[1368,359],[1082,359]
    ], dtype=np.int32),
}
APPLY_ROI = True
# --------------------------------------------

os.makedirs("dataset_players/images", exist_ok=True)
os.makedirs("dataset_players/labels", exist_ok=True)
os.makedirs("dataset_ball/images", exist_ok=True)
os.makedirs("dataset_ball/labels", exist_ok=True)

player_model = YOLO(PLAYER_MODEL_PATH)
ball_model   = YOLO(BALL_MODEL_PATH)

def apply_roi_mask(frame_rgb, cam_key):
    if not APPLY_ROI or cam_key not in ROI_IMG_POINTS:
        return frame_rgb
    mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(ROI_IMG_POINTS[cam_key]), 255)
    # mantenemos colores dentro de ROI; fuera, negro
    out = frame_rgb.copy()
    out[mask == 0] = 0
    return out

def xyxy_to_yolo(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return [
        (xc / W),
        (yc / H),
        (w / W),
        (h / H)
    ]

# --- callbacks para slicing ---
def player_callback(img_slice: np.ndarray) -> sv.Detections:
    res = player_model(img_slice, imgsz=1792, conf=PLAYER_CONF, iou=PLAYER_IOU)[0]
    det = sv.Detections.from_ultralytics(res)
    return det[det.class_id == 1]  # tu modelo: clase jugador=1

def ball_callback(img_slice: np.ndarray) -> sv.Detections:
    res = ball_model(img_slice, imgsz=1024, conf=BALL_CONF, iou=BALL_IOU)[0]
    det = sv.Detections.from_ultralytics(res)
    return det[det.class_id == 0]  # tu modelo: clase balón=0

slicer_players = sv.InferenceSlicer(
    callback=player_callback,
    slice_wh=PLAYER_SLICE_WH,
    overlap_wh=PLAYER_OVERLAP_WH,
    overlap_ratio_wh=None,
    overlap_filter=sv.OverlapFilter.NONE
)
slicer_ball = sv.InferenceSlicer(
    callback=ball_callback,
    slice_wh=BALL_SLICE_WH,
    overlap_wh=BALL_OVERLAP_WH,
    overlap_ratio_wh=None,
    overlap_filter=sv.OverlapFilter.NONE
)

all_player_imgs = []
all_ball_imgs = []

def save_example(base_dir, img_bgr, label_lines, base_name):
    img_path = os.path.join(base_dir, "images", base_name + ".jpg")
    lbl_path = os.path.join(base_dir, "labels", base_name + ".txt")
    cv2.imwrite(img_path, img_bgr)
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))
    return img_path

for video_path, cam_key in VIDEOS:
    vr = VideoReader(video_path, ctx=cpu(0))
    for idx, fr in enumerate(vr):
        if idx % FRAME_STRIDE:
            continue
        fr = fr.asnumpy()  # RGB
        fr = apply_roi_mask(fr, cam_key)
        H, W = fr.shape[:2]

        # ------- PLAYERS -------
        det_p = slicer_players(fr).with_nms(threshold=PLAYER_FUSE_NMS, class_agnostic=True)
        player_lines = []
        for bb in det_p.xyxy:
            x1, y1, x2, y2 = map(float, bb)
            # filtra cajas muy pequeñas (ruido)
            if (y2 - y1) < 15 or (x2 - x1) < 8:
                continue
            xc, yc, ww, hh = xyxy_to_yolo([x1, y1, x2, y2], W, H)
            # clase única = 0 en dataset_players
            player_lines.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        if player_lines or KEEP_EMPTY:
            base_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{cam_key}_f{idx:06d}"
            img_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            pimg = save_example("dataset_players", img_bgr, player_lines, base_name)
            all_player_imgs.append(pimg)

        # ------- BALL -------
        det_b = slicer_ball(fr).with_nms(threshold=BALL_FUSE_NMS, class_agnostic=True)
        ball_lines = []
        for bb in det_b.xyxy:
            x1, y1, x2, y2 = map(float, bb)
            if (y2 - y1) < 6 or (x2 - x1) < 6:
                continue
            xc, yc, ww, hh = xyxy_to_yolo([x1, y1, x2, y2], W, H)
            # clase única = 0 en dataset_ball
            ball_lines.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        if ball_lines or KEEP_EMPTY:
            base_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{cam_key}_f{idx:06d}"
            img_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            bimg = save_example("dataset_ball", img_bgr, ball_lines, base_name)
            all_ball_imgs.append(bimg)

print(f"[INFO] players: {len(all_player_imgs)} imágenes | ball: {len(all_ball_imgs)} imágenes")

# ---------- split train/val/test ----------
def split_files(img_paths, root_dir):
    random.shuffle(img_paths)
    n = len(img_paths)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train, val, test = img_paths[:n_train], img_paths[n_train:n_train+n_val], img_paths[n_train+n_val:]
    for split_name, split_list in [("train", train), ("val", val), ("test", test)]:
        os.makedirs(os.path.join(root_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "labels", split_name), exist_ok=True)
        for img_path in split_list:
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_src = os.path.join(root_dir, "labels", base + ".txt")
            img_dst = os.path.join(root_dir, "images", split_name, base + ".jpg")
            lbl_dst = os.path.join(root_dir, "labels", split_name, base + ".txt")
            os.replace(img_path, img_dst)
            if os.path.exists(lbl_src):
                os.replace(lbl_src, lbl_dst)
            elif KEEP_EMPTY:
                open(lbl_dst, "w").close()

split_files(all_player_imgs, "dataset_players")
split_files(all_ball_imgs,   "dataset_ball")

# crea YAMLs para entrenar directo
with open("dataset_players.yaml", "w", encoding="utf-8") as f:
    f.write(
        "path: ./dataset_players\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: player\n"
    )
with open("dataset_ball.yaml", "w", encoding="utf-8") as f:
    f.write(
        "path: ./dataset_ball\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: ball\n"
    )
print("[INFO] Datasets y YAMLs listos: dataset_players/, dataset_ball/, dataset_*.yaml")

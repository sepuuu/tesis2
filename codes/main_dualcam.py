
import os
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
import torchreid
from decord import VideoReader, cpu
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

from configs.drawing import PitchRenderer
from configs.soccer import SoccerPitchConfiguration
from configs.team import TeamClassifier
from configs.view_transformer import ViewTransformer
from utils.ball_setup import ball_tracker, callback as ball_callback
from utils.drawing_utils import draw_box, draw_player_box

###############################################################################
# 1. Configuración global                                                    #
###############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Videos de entrada – una ruta por cámara
# Asegúrate de que ambos vídeos tengan **la misma velocidad de fotogramas** y
# empiecen a la vez (o ya estén sincronizados).
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH_CAM1 = "codes/inputs/video_blancos_corto.mp4"
VIDEO_PATH_CAM2 = "codes/inputs/video_negros_corto.mp4"  # ⚠️ cambia si es otro

# Salida (se genera a partir del vídeo ⚠️CAM1)
TARGET_VIDEO_PATH = "codes/outputs/Pruebas-output-multicam.mov"

# Modelos
PLAYER_DETECTION_MODEL = YOLO("codes/models/players.onnx")
BALL_DETECTION_MODEL = YOLO("codes/models/ball.onnx")

###############################################################################
# 2. Inicialización de componentes                                           #
###############################################################################

# Team classifier (SigLIP +k-means) – sin entrenar de momento
team_classifier = TeamClassifier(device=device)

# Re-identificación de jugadores (ResNet50 entrenado previamente)
reid_model = torchreid.models.build_model(name="resnet50", num_classes=12, pretrained=False)
checkpoint = torch.load("codes/models/model.pth.tar-300", map_location=device)
reid_model.load_state_dict(checkpoint["state_dict"])
reid_model = reid_model.to(device).eval()

# Preprocesamiento para el modelo de re-id
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# VideoReaders y FPS
vr1 = VideoReader(VIDEO_PATH_CAM1, ctx=cpu(0))
vr2 = VideoReader(VIDEO_PATH_CAM2, ctx=cpu(0))

fps1 = vr1.get_avg_fps()
fps2 = vr2.get_avg_fps()

if not math.isclose(fps1, fps2, rel_tol=1e-2):
    raise ValueError(f"Los FPS de los vídeos no coinciden: cam1={fps1:.2f}, cam2={fps2:.2f}")

FPS = fps1
TOTAL_FRAMES = min(len(vr1), len(vr2))

###############################################################################
# 3. Puntos de homografía para cada cámara                                    #
###############################################################################

# Cámara 1
points_image_cam1 = np.array([
    [912, 203], [1003, 213], [1063, 221], [1180, 237], [1237, 243], [1358, 265],
    [965, 220], [1216, 253], [898, 270], [42, 768], [149, 369], [604, 499]
], dtype=np.float32)
points_pitch_cam1 = np.array([
    [0, 0], [0, 500], [0, 800], [0, 1200], [0, 1500], [0, 2000],
    [500, 500], [500, 1500], [2000, 1000], [4000, 1500], [3500, 500], [3500, 1500]
], dtype=np.float32)

# Cámara 2
points_image_cam2 = np.array([
    [256, 829], [811, 596], [451, 506], [1066, 413], [1497, 360], [1368, 359],
    [1312, 359], [1206, 360], [1164, 360], [1082, 359], [1348, 371], [1120, 370]
], dtype=np.float32)
points_pitch_cam2 = np.array([
    [0, 500], [500, 500], [500, 1500], [2000, 1000], [4000, 0], [4000, 500],
    [4000, 800], [4000, 1200], [4000, 1500], [4000, 2000], [3500, 500], [3500, 1500]
], dtype=np.float32)

view_transformer_cam1 = ViewTransformer(source=points_image_cam1, target=points_pitch_cam1)
view_transformer_cam2 = ViewTransformer(source=points_image_cam2, target=points_pitch_cam2)

###############################################################################
# 4. Funciones auxiliares                                                     #
###############################################################################

def collect_team_crops(video_path: str, model: TeamClassifier, fps: float) -> List[np.ndarray]:
    """Recolecta crops de jugadores para entrenar el *TeamClassifier*.

    Esto replica la lógica existente pero **solo devuelve** la lista de crops
    sin llamar a `model.fit`, de modo que podamos unir los crops de ambas
    cámaras antes de entrenar.
    """
    print(f"[INFO] Recolectando crops (video: {os.path.basename(video_path)})…")
    stride = int(max(1, fps // 7))
    crops: List[np.ndarray] = []

    vr = VideoReader(video_path, ctx=cpu(0))
    for idx, fr in enumerate(vr):
        if idx % stride:
            continue
        fr = fr.asnumpy()
        res = PLAYER_DETECTION_MODEL.predict(fr, imgsz=1792, iou=0.7)[0]
        det = sv.Detections.from_ultralytics(res)
        players = det[det.class_id == 1].with_nms(threshold=0.5, class_agnostic=True)
        for xyxy in players.xyxy:
            crop = model._crop(fr, xyxy)
            if crop.size == 0:
                continue
            v_mean = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[..., 2].mean()
            if not (60 < v_mean < 240):
                continue
            crops.append(crop)
    print(f"[INFO] {len(crops)} crops recolectados en {os.path.basename(video_path)}")
    return crops


def extract_embedding(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = reid_model(crop).cpu().numpy()[0]
    return emb


def normalized_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return np.linalg.norm(a - b)


def find_best_match(emb: np.ndarray, tracked_embs: Dict[int, np.ndarray], threshold: float = 0.8) -> int | None:
    emb = emb / np.linalg.norm(emb)
    best_id, best_score = None, threshold
    for tid, t_emb in tracked_embs.items():
        score = normalized_distance(emb, t_emb)
        if score < best_score:
            best_id, best_score = tid, score
    return best_id


def detect_ball(frame_rgb: np.ndarray, vt: ViewTransformer):
    """Devuelve (pitch_xy, confidence, bbox) ó (None, 0, None)."""
    slicer = sv.InferenceSlicer(
        callback=ball_callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,
        overlap_wh=(0, 0),
        overlap_filter=sv.OverlapFilter.NONE,
    )
    detections = slicer(frame_rgb).with_nms(threshold=0.05)
    detections = ball_tracker.filter_detections(detections)
    balls = detections[detections.class_id == 0]
    if balls.empty:
        return None, 0.0, None
    idx = int(np.argmax(balls.confidence))
    balls.xyxy = balls.xyxy[idx : idx + 1]
    pitch_xy = vt.transform_points(balls.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))[0]
    return pitch_xy, float(balls.confidence[idx]), balls.xyxy[0]

###############################################################################
# 5. Entrenamiento del TeamClassifier con crops de ambas cámaras              #
###############################################################################

all_crops = collect_team_crops(VIDEO_PATH_CAM1, team_classifier, FPS) + \
            collect_team_crops(VIDEO_PATH_CAM2, team_classifier, FPS)
team_classifier.fit(all_crops)
print("[INFO] TeamClassifier entrenado con crops de ambas cámaras ✓")

###############################################################################
# 6. Estructuras para tracking y resultados                                   #
###############################################################################

tracked_embeddings: Dict[int, np.ndarray] = {}
player_teams: Dict[int, str] = {}
next_id: int = 1

#  Registro para CSV   (Frame, Id, Pos X, Pos Y, Ball X, Ball Y, Team)
posiciones_df = pd.DataFrame(columns=[
    "Frame", "Id", "Pos X", "Pos Y", "Ball X", "Ball Y", "Team",
])

###############################################################################
# 7. Video writer (overlay se hace sobre la cámara 1)                         #
###############################################################################

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
frame0 = vr1[0].asnumpy()
vid_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, FPS, (frame0.shape[1], frame0.shape[0]))

# Radar
CONFIG = SoccerPitchConfiguration()
radar_renderer = PitchRenderer(
    config=CONFIG,
    scale=0.1,
    padding=50,
    background_color=sv.Color(34, 139, 34),
    line_color=sv.Color.WHITE,
)
radar_w = int(frame0.shape[1] * 0.4)
radar_h = int(frame0.shape[0] * 0.3)
radar_pos = (int((frame0.shape[1] - radar_w) / 2), int(frame0.shape[0] - radar_h - 20))

###############################################################################
# 8. Procesamiento frame-a-frame                                              #
###############################################################################

def process_camera(
    frame_rgb: np.ndarray,
    vt: ViewTransformer,
    draw_on_frame: bool = False,
    frame_for_draw: np.ndarray | None = None,
):
    """Detecta, re-identifica y devuelve asignaciones y team-pred.

    Retorna: List[Tuple[id, pitch_xy, team_str, bbox]]
    """
    global next_id
    res = PLAYER_DETECTION_MODEL.predict(frame_rgb, imgsz=1792, iou=0.7)[0]
    detections = sv.Detections.from_ultralytics(res)
    players = detections[detections.class_id == 1]

    results: List[Tuple[int, np.ndarray, str, np.ndarray]] = []
    for bbox in players.xyxy:
        emb = extract_embedding(frame_rgb, bbox)
        if emb is None:
            continue
        match_id = find_best_match(emb, tracked_embeddings)
        if match_id is None:
            match_id = next_id
            next_id += 1
        tracked_embeddings[match_id] = emb

        # Predicción de equipo
        team = team_classifier.predict_team(frame_rgb, bbox, tracker_id=match_id)
        player_teams[match_id] = team

        # Transformar a coordenadas de cancha
        x1, y1, x2, y2 = map(float, bbox)
        bottom_center = np.array([[ (x1 + x2) / 2, y2 ]], dtype=np.float32)
        pitch_xy = vt.transform_points(bottom_center)[0]

        results.append((match_id, pitch_xy, team, bbox))

        if draw_on_frame and frame_for_draw is not None:
            color_map = {
                "equipo_negro":  (0, 0, 0),
                "equipo_blanco": (255, 255, 255),
                "UNKNOWN":       (128, 128, 128),
            }
            draw_player_box(frame_for_draw, bbox, match_id, color_map.get(team, (128, 128, 128)))
    return results

print("[INFO] Procesando frames… (Ctrl-C para interrumpir)")

last_ball_position = (math.nan, math.nan)

for frame_idx in tqdm(range(TOTAL_FRAMES)):
    # ── Lectura y conversión a RGB (OpenCV trabaja BGR pero decord entrega RGB) ──
    fr1 = vr1[frame_idx].asnumpy()
    fr2 = vr2[frame_idx].asnumpy()

    # 
    # 1) Detección de balón en ambas cámaras
    #    Nos quedamos con la detección de mayor confianza;
    #    si existe en ambas, promediamos las posiciones.
    # -------------------------------------------------------------------------
    ball1_xy, conf1, ball_bbox1 = detect_ball(fr1, view_transformer_cam1)
    ball2_xy, conf2, ball_bbox2 = detect_ball(fr2, view_transformer_cam2)

    if conf1 == 0 and conf2 == 0:
        ball_pitch_xy = (math.nan, math.nan)
    elif conf1 >= conf2:
        ball_pitch_xy = ball1_xy
    elif conf2 > conf1:
        ball_pitch_xy = ball2_xy
    else:
        # promedio como fallback (ambas válidas)
        ball_pitch_xy = ((ball1_xy[0] + ball2_xy[0]) / 2, (ball1_xy[1] + ball2_xy[1]) / 2)

    last_ball_position = ball_pitch_xy

    #  2) Jugadores – cam1 y cam2
    assignments_cam1 = process_camera(fr1, view_transformer_cam1, draw_on_frame=True, frame_for_draw=fr1)
    assignments_cam2 = process_camera(fr2, view_transformer_cam2, draw_on_frame=False)

    #  3) Fusionar posiciones: si el mismo id se ve en ambas, promediamos
    aggregated: Dict[int, List[np.ndarray]] = defaultdict(list)
    for tid, pitch_xy, team, _ in assignments_cam1 + assignments_cam2:
        aggregated[tid].append(pitch_xy)

    fused_positions: Dict[int, np.ndarray] = {}
    for tid, pts in aggregated.items():
        fused_positions[tid] = np.mean(pts, axis=0)

    #  4) DataFrame de salida (una fila por jugador visible en este frame)
    for tid, pitch_xy in fused_positions.items():
        team = player_teams.get(tid, "UNKNOWN")
        posiciones_df.loc[len(posiciones_df)] = [
            frame_idx, tid, float(pitch_xy[0]), float(pitch_xy[1]),
            float(last_ball_position[0]), float(last_ball_position[1]),
            team,
        ]

    #  5) Dibujar radar
    radar_elements = {"points": []}
    for tid, pitch_xy in fused_positions.items():
        color = (0, 0, 0) if player_teams.get(tid) == "equipo_negro" else (255, 255, 255)
        radar_elements["points"].append((pitch_xy[0], pitch_xy[1], color))

    if not (math.isnan(last_ball_position[0]) or math.isnan(last_ball_position[1])):
        radar_elements["points"].append((last_ball_position[0], last_ball_position[1], (0, 255, 255)))

    radar_img = radar_renderer.draw(radar_elements)
    radar_resized = cv2.resize(radar_img, (radar_w, radar_h))

    # Superponer radar en fr1
    overlay = fr1.copy()
    x0, y0 = radar_pos
    alpha = 0.6
    overlay[y0 : y0 + radar_h, x0 : x0 + radar_w] = (
        alpha * radar_resized + (1 - alpha) * overlay[y0 : y0 + radar_h, x0 : x0 + radar_w]
    ).astype(np.uint8)

    frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    vid_writer.write(frame_bgr)

###############################################################################
# 9. Post-proceso y guardado                                                 #
###############################################################################

vid_writer.release()
print(f"[INFO] Video guardado en {TARGET_VIDEO_PATH}")

# Guardar Excel provisional
os.makedirs("codes/data", exist_ok=True)
excel_tmp = "codes/data/Posiciones-jugadores-balon-multicam.xlsx"
posiciones_df.to_excel(excel_tmp, index=False)
print(f"[INFO] Posiciones exportadas a {excel_tmp}")

# Mantener el post-proceso existente (posesión, pases, etc.)
from postprocess.postprocess import process_file

process_file(
    file_path=excel_tmp,
    cleaned_output_path="codes/data/limpieza.xlsx",
    output_possession_path="codes/data/posesion.xlsx",
    output_passes_path="codes/data/pases.xlsx",
    output_team_passes_path="codes/data/passes_by_{team}.xlsx",
)

print("[INFO] ¡Pipeline multicámara finalizado con éxito! ✓")

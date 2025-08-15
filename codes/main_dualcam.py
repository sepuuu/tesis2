import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"

import math
import torchreid
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from configs.soccer import SoccerPitchConfiguration
from decord import VideoReader, cpu
from torchvision import transforms
import supervision as sv
import numpy as np
import cv2
import torch
from configs.drawing import PitchRenderer
from configs.view_transformer import ViewTransformer
from utils.drawing_utils import draw_player_box, draw_box
from utils.ball_setup import callback, ball_tracker
from utils.tracking import SimpleTracker   # puedes usar el tuyo o el mejorado

# =========================
# AJUSTA TUS PATHS
# =========================
VIDEO_PATH_CAM1 = "codes/inputs/video_blancos_corto.mp4"
VIDEO_PATH_CAM2 = "codes/inputs/video_negros_corto.mp4"
TARGET_VIDEO_CAM1 = "codes/outputs/Pruebas-output-cam1.mov"
TARGET_VIDEO_CAM2 = "codes/outputs/Pruebas-output-cam2.mov"
TARGET_VIDEO_SIDE = "codes/outputs/Pruebas-output-sidebyside.mov"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODELOS
# =========================
reid_model = torchreid.models.build_model(name='resnet50', num_classes=12, pretrained=False)
checkpoint = torch.load('codes/models/model.pth.tar-300', map_location=device)
reid_model.load_state_dict(checkpoint['state_dict'])
reid_model = reid_model.to(device).eval()

PLAYER_DETECTION_MODEL = YOLO("codes/models/players.onnx")  # puedes pasar task="detect" si quieres silenciar el warning

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CONFIG = SoccerPitchConfiguration()

# =========================
# HOMOGRAFÍAS
# =========================
# CAM1
points_img1 = np.array([
    [912, 203], [1003, 213], [1063, 221], [1180, 237], [1237, 243], [1358, 265],
    [965, 220], [1216, 253], [898, 270], [42, 768], [149, 369], [604, 499]
], dtype=np.float32)
points_pitch1 = np.array([
    [0, 0], [0, 500], [0, 800], [0, 1200], [0, 1500], [0, 2000],
    [500, 500], [500, 1500], [2000, 1000], [4000, 1500], [3500, 500], [3500, 1500]
], dtype=np.float32)
view_transformer1 = ViewTransformer(source=points_img1, target=points_pitch1)

# CAM2
points_img2 = np.array([
    [256, 829], [811, 596], [451, 506], [1066, 413], [1497, 360], [1368, 359],
    [1312, 359], [1206, 360], [1164, 360], [1082, 359], [1348, 371], [1120, 370]
], dtype=np.float32)
points_pitch2 = np.array([
    [0, 500], [500, 500], [500, 1500], [2000, 1000], [4000, 0], [4000, 500],
    [4000, 800], [4000, 1200], [4000, 1500], [4000, 2000], [3500, 500], [3500, 1500]
], dtype=np.float32)
view_transformer2 = ViewTransformer(source=points_img2, target=points_pitch2)

# =========================
# AUXILIARES
# =========================
def extract_embedding(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = reid_model(crop).cpu().numpy()[0]
    return emb.astype(np.float32)

def l2norm(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n

def reid_dist(a, b):
    return np.linalg.norm(l2norm(a) - l2norm(b))

# =========================
# PARÁMETROS DE FUSIÓN Y ROSTER
# =========================
# Fusión de detecciones entre cámaras (reduce duplicados)
FUSE_POS_THR = 60.0     # cm en cancha para considerar que dos detecciones son el mismo jugador
FUSE_REID_THR = 0.65    # distancia reid para considerar “muy similares”

# Roster (identidades visibles estables 1..N)
ROSTER_SIZE = 11    # máximo número de jugadores a mostrar
N_INIT = 3              # frames necesarios para confirmar una identidad (tentativa -> confirmada)
LOST_TOL = 45           # frames de tolerancia para reciclar un roster slot si no se ve
ALPHA_ROSTER = 0.6      # mezcla de costo: alpha*reid + (1-alpha)*(dist_pos/pos_thr)
POS_THR_ROSTER = 250.0  # normalizador de distancia en cancha para costo

# =========================
# TRACKER GLOBAL (puedes usar el tuyo actual)
# =========================
tracker = SimpleTracker(
    reid_weight=0.7, pos_weight=0.3,
    reid_threshold=0.8, pos_threshold=220.0  # si usas el mejorado, puedes agregar max_age, etc.
)

# =========================
# PROCESO MULTICÁMARA
# =========================
def process_dual_camera(VIDEO_PATH_CAM1, VIDEO_PATH_CAM2,
                        TARGET_VIDEO_CAM1, TARGET_VIDEO_CAM2,
                        vt1, vt2):
    vr1 = VideoReader(VIDEO_PATH_CAM1, ctx=cpu(0))
    vr2 = VideoReader(VIDEO_PATH_CAM2, ctx=cpu(0))
    total_frames = min(len(vr1), len(vr2))
    fps = vr1.get_avg_fps()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # para .mov OpenCV puede cambiar a 'jpeg' internamente
    h1, w1 = vr1[0].shape[0], vr1[0].shape[1]
    h2, w2 = vr2[0].shape[0], vr2[0].shape[1]
    out1 = cv2.VideoWriter(TARGET_VIDEO_CAM1, fourcc, fps, (w1, h1))
    out2 = cv2.VideoWriter(TARGET_VIDEO_CAM2, fourcc, fps, (w2, h2))

    pitch_renderer = PitchRenderer(
        config=CONFIG,
        scale=0.1,
        padding=50,
        background_color=sv.Color(34, 139, 34),
        line_color=sv.Color.WHITE
    )
    radar_width = int(w1 * 0.4)
    radar_height = int(h1 * 0.3)
    radar_position = (int((w1 - radar_width) / 2), int(h1 - radar_height - 20))

    # Export
    posiciones_df = pd.DataFrame(columns=['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y'])
    frames_out_cam1, frames_out_cam2 = [], []
    last_ball_position = [math.nan, math.nan]

    slicer_ball = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,
        overlap_wh=(0, 0),
        overlap_filter=sv.OverlapFilter.NONE
    )

    # -------------------------
    # Estado del ROSTER (cap en 12 IDs visibles)
    # -------------------------
    # (1..ROSTER_SIZE) son IDs visibles estables
    roster_emb = {rid: None for rid in range(1, ROSTER_SIZE+1)}     # embedding prototipo
    roster_pos = {rid: None for rid in range(1, ROSTER_SIZE+1)}     # posición cancha
    roster_last_seen = {rid: -10**9 for rid in range(1, ROSTER_SIZE+1)}  # frame idx
    roster_confirm_hits = {rid: 0 for rid in range(1, ROSTER_SIZE+1)}    # hits para N_INIT
    # mapeo dinámico track_id -> roster_id (por estabilidad de dibujo)
    track2roster = {}  # track_id actual -> roster_id

    def roster_cost(emb, pos, rid):
        # costo mixto para asignar un track a un slot de roster
        if roster_emb[rid] is None or roster_pos[rid] is None:
            # slot vacío: bajo costo para ocuparlo
            return 0.0
        d_reid = reid_dist(emb, roster_emb[rid])
        d_pos = np.linalg.norm(pos - roster_pos[rid]) / POS_THR_ROSTER
        return ALPHA_ROSTER * d_reid + (1 - ALPHA_ROSTER) * d_pos

    def assign_roster(frame_idx, t_id, emb, pos):
        # ya mapeado: actualiza y devuelve
        if t_id in track2roster:
            rid = track2roster[t_id]
            roster_emb[rid] = emb if roster_emb[rid] is None else (0.8*roster_emb[rid] + 0.2*emb)
            roster_pos[rid] = pos if roster_pos[rid] is None else (0.6*roster_pos[rid] + 0.4*pos)
            roster_last_seen[rid] = frame_idx
            roster_confirm_hits[rid] = min(N_INIT, roster_confirm_hits[rid] + 1)
            return rid, True  # True: confirmado o en proceso

        # buscar mejor slot por costo
        best_rid, best_cost = None, float("inf")
        for rid in range(1, ROSTER_SIZE+1):
            c = roster_cost(emb, pos, rid)
            if c < best_cost:
                best_cost, best_rid = c, rid

        # política de ocupación:
        # - si el mejor slot está vacío o stale (no visto hace >= LOST_TOL), úsalo
        # - si todos están ocupados recientes, NO creamos identidad nueva (queda tentativo hasta acumular N_INIT Y liberar alguno)
        stale = (frame_idx - roster_last_seen[best_rid]) >= LOST_TOL
        if roster_emb[best_rid] is None or stale:
            # ocupar / reciclar slot
            track2roster[t_id] = best_rid
            roster_emb[best_rid] = emb.copy()
            roster_pos[best_rid] = pos.copy()
            roster_last_seen[best_rid] = frame_idx
            roster_confirm_hits[best_rid] = 1  # empieza como tentativo
            return best_rid, False  # False: tentativo
        else:
            # no hay hueco aún: intentamos “match suave” si costo muy bajo
            if best_cost < 0.25:  # umbral pequeño: muy probable que sea el mismo
                track2roster[t_id] = best_rid
                roster_emb[best_rid] = 0.8*roster_emb[best_rid] + 0.2*emb
                roster_pos[best_rid] = 0.6*roster_pos[best_rid] + 0.4*pos
                roster_last_seen[best_rid] = frame_idx
                roster_confirm_hits[best_rid] = min(N_INIT, roster_confirm_hits[best_rid] + 1)
                return best_rid, roster_confirm_hits[best_rid] >= N_INIT
            # si no, no asignamos (no dibujamos) hasta que haya hueco o acumule evidencia
            return None, False

    def cleanup_roster(frame_idx):
        # si un slot está perdido hace mucho, lo marcamos como libre (para poder reciclar)
        for rid in range(1, ROSTER_SIZE+1):
            if (frame_idx - roster_last_seen[rid]) >= (LOST_TOL * 2):
                roster_emb[rid] = None
                roster_pos[rid] = None
                roster_confirm_hits[rid] = 0
        # limpiar mapeos de tracks que ya no existen: lo hace implícitamente
        # (no sabemos los activos desde aquí, pero si no se ven por mucho, el slot se recicla)

    # --------- LOOP PRINCIPAL ---------
    for i in tqdm(range(total_frames)):
        fr1 = vr1[i].asnumpy()
        fr2 = vr2[i].asnumpy()
        rgb1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB)

        # ===== BALÓN: elegimos mejor cámara por confianza =====
        det_ball_1 = slicer_ball(rgb1).with_nms(threshold=0.05)
        det_ball_1 = ball_tracker.filter_detections(det_ball_1)
        balls1 = det_ball_1[det_ball_1.class_id == 0]

        det_ball_2 = slicer_ball(rgb2).with_nms(threshold=0.05)
        det_ball_2 = ball_tracker.filter_detections(det_ball_2)
        balls2 = det_ball_2[det_ball_2.class_id == 0]

        ball_xy_pitch, ball_bbox_cam = [math.nan, math.nan], None
        conf1 = float(balls1.confidence.max()) if len(balls1.confidence) > 0 else 0.0
        conf2 = float(balls2.confidence.max()) if len(balls2.confidence) > 0 else 0.0

        if conf1 == 0.0 and conf2 == 0.0:
            pass
        else:
            if conf1 >= conf2:
                idx = int(np.argmax(balls1.confidence))
                balls1.xyxy = balls1.xyxy[idx:idx+1]
                p1 = view_transformer1.transform_points(balls1.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
                ball_xy_pitch = [float(p1[0,0]), float(p1[0,1])]
                ball_bbox_cam = ('cam1', balls1.xyxy[0])
            else:
                idx = int(np.argmax(balls2.confidence))
                balls2.xyxy = balls2.xyxy[idx:idx+1]
                p2 = view_transformer2.transform_points(balls2.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
                ball_xy_pitch = [float(p2[0,0]), float(p2[0,1])]
                ball_bbox_cam = ('cam2', balls2.xyxy[0])

        last_ball_position = ball_xy_pitch
        ball_x, ball_y = last_ball_position

        # ===== JUGADORES: detectamos en ambas cámaras =====
        res1 = PLAYER_DETECTION_MODEL.predict(fr1, imgsz=1792, iou=0.7)[0]
        det1 = sv.Detections.from_ultralytics(res1)
        ply1 = det1[det1.class_id == 1]

        res2 = PLAYER_DETECTION_MODEL.predict(fr2, imgsz=1792, iou=0.7)[0]
        det2 = sv.Detections.from_ultralytics(res2)
        ply2 = det2[det2.class_id == 1]

        # Convertimos a lista con meta por cámara
        cams = []
        if len(ply1.xyxy) > 0:
            xy1 = view_transformer1.transform_points(
                np.array(ply1.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
            )
            for bbox, conf, pxy in zip(ply1.xyxy, ply1.confidence, xy1):
                emb = extract_embedding(fr1, bbox)
                if emb is None: 
                    continue
                h_rel = float(bbox[3] - bbox[1]) / h1
                score = float(conf) * (h_rel ** 2)
                cams.append(dict(cam='cam1', bbox=bbox, conf=float(conf), pos=pxy.astype(np.float32),
                                 emb=emb, h_rel=h_rel, score=score))

        if len(ply2.xyxy) > 0:
            xy2 = view_transformer2.transform_points(
                np.array(ply2.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
            )
            for bbox, conf, pxy in zip(ply2.xyxy, ply2.confidence, xy2):
                emb = extract_embedding(fr2, bbox)
                if emb is None: 
                    continue
                h_rel = float(bbox[3] - bbox[1]) / h2
                score = float(conf) * (h_rel ** 2)
                cams.append(dict(cam='cam2', bbox=bbox, conf=float(conf), pos=pxy.astype(np.float32),
                                 emb=emb, h_rel=h_rel, score=score))

        # ===== FUSIÓN ENTRE CÁMARAS (evita duplicados de un mismo jugador) =====
        # Greedy: comparamos detecciones y si están muy cerca en cancha y similares en reid, nos quedamos con la de mejor score
        cams.sort(key=lambda d: d['score'], reverse=True)  # primero las mejores
        fused = []
        used = [False]*len(cams)

        for a in range(len(cams)):
            if used[a]: 
                continue
            base = cams[a]
            for b in range(a+1, len(cams)):
                if used[b]:
                    continue
                other = cams[b]
                if np.linalg.norm(base['pos'] - other['pos']) <= FUSE_POS_THR and reid_dist(base['emb'], other['emb']) <= FUSE_REID_THR:
                    # ya que base tiene >= score, nos quedamos con base; podemos refinar promediando
                    base['pos'] = (0.7*base['pos'] + 0.3*other['pos']).astype(np.float32)
                    base['emb'] = 0.7*base['emb'] + 0.3*other['emb']
                    used[b] = True
            fused.append(base)
            used[a] = True

        # ===== TRACKER GLOBAL (posición+reid) =====
        det_list = [(d['pos'], d['emb']) for d in fused]
        assigns = tracker.update(det_list)  # [(track_id, pos, emb)] en el MISMO orden que det_list

        # ===== ROSTER: fija el número máximo de IDs visibles (1..ROSTER_SIZE) =====
        # Por cada track asignado, adjudicamos/reciclamos un roster_id estable.
        # Si no hay slot disponible y no es muy similar a alguien, NO se dibuja ni se exporta.
        draw_pack = []  # (roster_id, cam, bbox, pos)
        for (t_id, pos, emb), det in zip(assigns, fused):
            rid, confirmed = assign_roster(i, t_id, emb, pos)
            if rid is None:
                continue  # sin slot (aún)
            if confirmed or roster_confirm_hits[rid] >= N_INIT:
                # Dibuja en la mejor cámara (la de 'fused' ya es la mejor por score)
                draw_pack.append((rid, det['cam'], det['bbox'], pos))

        # limpieza de roster (recicla slots perdidos hace mucho)
        cleanup_roster(i)

        # ===== DIBUJO / EXPORT =====
        # balón en la cámara que ganó
        if ball_bbox_cam is not None:
            camb, bb = ball_bbox_cam
            if camb == 'cam1':
                draw_box(fr1, bb, "Ball", color=(0, 255, 255))
            else:
                draw_box(fr2, bb, "Ball", color=(0, 255, 255))

        # jugadores (IDs visibles = roster_id)
        players_positions = {}
        for rid, cam, bbox, pos in draw_pack:
            players_positions[rid] = pos
            if cam == 'cam1':
                draw_player_box(fr1, bbox, rid, (0, 255, 255))
            else:
                draw_player_box(fr2, bbox, rid, (0, 255, 255))

            posiciones_df.loc[len(posiciones_df)] = [
                i, rid, float(pos[0]), float(pos[1]),
                float(ball_x), float(ball_y)
            ]

        # Radar (sobre cam1)
        elements = {'points': [], 'paths': []}
        for _, pxy in players_positions.items():
            elements['points'].append((pxy[0], pxy[1], (255, 255, 255)))
        if not (math.isnan(ball_x) or math.isnan(ball_y)):
            elements['points'].append((ball_x, ball_y, (0, 255, 255)))

        radar_image = pitch_renderer.draw(elements)
        radar_resized = cv2.resize(radar_image, (radar_width, radar_height))
        x, y = radar_position
        alpha = 0.6
        fr1_overlay = fr1.copy()
        for c in range(3):
            fr1_overlay[y:y+radar_height, x:x+radar_width, c] = (
                alpha * radar_resized[:, :, c] +
                (1 - alpha) * fr1_overlay[y:y+radar_height, x:x+radar_width, c]
            )
        fr1_bgr = cv2.cvtColor(fr1_overlay, cv2.COLOR_RGB2BGR)
        fr2_bgr = cv2.cvtColor(fr2, cv2.COLOR_RGB2BGR)

        out1.write(fr1_bgr)
        out2.write(fr2_bgr)
        frames_out_cam1.append(fr1_bgr)
        frames_out_cam2.append(fr2_bgr)

    out1.release()
    out2.release()

    os.makedirs("codes/data", exist_ok=True)
    posiciones_df.to_excel("codes/data/Posiciones-jugadores-balon-multicam.xlsx", index=False)

    return frames_out_cam1, frames_out_cam2

# =========================
# EJECUCIÓN
# =========================
frames_cam1, frames_cam2 = process_dual_camera(
    VIDEO_PATH_CAM1, VIDEO_PATH_CAM2,
    TARGET_VIDEO_CAM1, TARGET_VIDEO_CAM2,
    view_transformer1, view_transformer2
)

# SIDE-BY-SIDE
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
h, w, _ = frames_cam1[0].shape
video_side = cv2.VideoWriter(TARGET_VIDEO_SIDE, fourcc, 30, (w * 2, h))

for f1, f2 in tqdm(zip(frames_cam1, frames_cam2), total=min(len(frames_cam1), len(frames_cam2))):
    combined = np.hstack((f1, f2))
    video_side.write(combined)
video_side.release()

print("¡Listo! Revisa:")
print("-", TARGET_VIDEO_CAM1)
print("-", TARGET_VIDEO_CAM2)
print("-", TARGET_VIDEO_SIDE)

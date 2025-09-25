import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"

import math
import time
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
TARGET_VIDEO_OUTPUT = "codes/outputs/Pruebas-output-final.mp4"  # MP4 único

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
FUSE_POS_THR = 60.0     # cm
FUSE_REID_THR = 0.65    # distancia reid

ROSTER_SIZE = 11
N_INIT = 10
LOST_TOL = 45
ALPHA_ROSTER = 0.6
POS_THR_ROSTER = 250.0

# =========================
# ⚠️ ASIGNACIÓN MANUAL DE EQUIPOS POR ID (roster_id)
# =========================
TEAM_BLACK_IDS = {3, 11, 10, 9, 8}
TEAM_WHITE_IDS = {1, 2, 5, 4, 6, 7}

COLOR_BLACK = (0, 0, 0)           # BGR
COLOR_WHITE = (255, 255, 255)     # BGR
COLOR_UNKNOWN = (127, 127, 127)   # BGR (fallback)

# Para Voronoi (relleno translúcido)
VOR_ALPHA = 0.25
VOR_COLOR_BLACK = (20, 20, 20)    # más visible que (0,0,0) al mezclar
VOR_COLOR_WHITE = (235, 235, 235)

# =========================
# TRACKER GLOBAL
# =========================
tracker = SimpleTracker(
    reid_weight=0.7, pos_weight=0.3,
    reid_threshold=0.8, pos_threshold=220.0
)

# =========================
# PROCESO MULTICÁMARA
# =========================
def process_dual_camera(VIDEO_PATH_CAM1, VIDEO_PATH_CAM2, vt1, vt2):
    vr1 = VideoReader(VIDEO_PATH_CAM1, ctx=cpu(0))
    vr2 = VideoReader(VIDEO_PATH_CAM2, ctx=cpu(0))
    total_frames = min(len(vr1), len(vr2))
    fps = vr1.get_avg_fps()

    # ===== anti-parpadeo: tiempo de “pegado” (≈1s) =====
    STICKY_MISS_TOL = max(1, int(round(fps * 1.0)))  # ~1 segundo
    last_pos_cache = {}  # rid -> (np.array([x,y],float32), last_seen_frame)

    # ===== MÉTRICAS =====
    os.makedirs("codes/data", exist_ok=True)
    metrics_rows = []  # por-frame
    id_switches_cum = 0
    roster_last_track = {}  # rid -> track_id con el que estaba (para detectar switches)
    DUP_POS_THR = 40.0  # cm: dos jugadores demasiado cerca (potencial duplicado residual)
    t_start_total = time.time()

    # Preparamos dimensiones para salida side-by-side
    h1, w1 = vr1[0].shape[0], vr1[0].shape[1]
    h2, w2 = vr2[0].shape[0], vr2[0].shape[1]
    out_h = max(h1, h2)
    out_w = w1 + w2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
    video_out = cv2.VideoWriter(TARGET_VIDEO_OUTPUT, fourcc, fps, (out_w, out_h))

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
    last_ball_position = [math.nan, math.nan]

    slicer_ball = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,
        overlap_wh=(0, 0),
        overlap_filter=sv.OverlapFilter.NONE
    )

    # -------------------------
    # Estado del ROSTER
    # -------------------------
    roster_emb = {rid: None for rid in range(1, ROSTER_SIZE+1)}
    roster_pos = {rid: None for rid in range(1, ROSTER_SIZE+1)}
    roster_last_seen = {rid: -10**9 for rid in range(1, ROSTER_SIZE+1)}
    roster_confirm_hits = {rid: 0 for rid in range(1, ROSTER_SIZE+1)}
    track2roster = {}

    def roster_cost(emb, pos, rid):
        if roster_emb[rid] is None or roster_pos[rid] is None:
            return 0.0
        d_reid = reid_dist(emb, roster_emb[rid])
        d_pos = np.linalg.norm(pos - roster_pos[rid]) / POS_THR_ROSTER
        return ALPHA_ROSTER * d_reid + (1 - ALPHA_ROSTER) * d_pos

    def assign_roster(frame_idx, t_id, emb, pos):
        if t_id in track2roster:
            rid = track2roster[t_id]
            roster_emb[rid] = emb if roster_emb[rid] is None else (0.8*roster_emb[rid] + 0.2*emb)
            roster_pos[rid] = pos if roster_pos[rid] is None else (0.6*roster_pos[rid] + 0.4*pos)
            roster_last_seen[rid] = frame_idx
            roster_confirm_hits[rid] = min(N_INIT, roster_confirm_hits[rid] + 1)
            return rid, True

        best_rid, best_cost = None, float("inf")
        for rid in range(1, ROSTER_SIZE+1):
            c = roster_cost(emb, pos, rid)
            if c < best_cost:
                best_cost, best_rid = c, rid

        stale = (frame_idx - roster_last_seen[best_rid]) >= LOST_TOL
        if roster_emb[best_rid] is None or stale:
            track2roster[t_id] = best_rid
            roster_emb[best_rid] = emb.copy()
            roster_pos[best_rid] = pos.copy()
            roster_last_seen[best_rid] = frame_idx
            roster_confirm_hits[best_rid] = 1
            return best_rid, False
        else:
            if best_cost < 0.25:
                track2roster[t_id] = best_rid
                roster_emb[best_rid] = 0.8*roster_emb[best_rid] + 0.2*emb
                roster_pos[best_rid] = 0.6*roster_pos[best_rid] + 0.4*pos
                roster_last_seen[best_rid] = frame_idx   # corregido
                roster_confirm_hits[best_rid] = min(N_INIT, roster_confirm_hits[best_rid] + 1)
                return best_rid, roster_confirm_hits[best_rid] >= N_INIT
            return None, False

    def cleanup_roster(frame_idx):
        for rid in range(1, ROSTER_SIZE+1):
            if (frame_idx - roster_last_seen[rid]) >= (LOST_TOL * 2):
                roster_emb[rid] = None
                roster_pos[rid] = None
                roster_confirm_hits[rid] = 0

    # --------- LOOP PRINCIPAL ---------
    for i in tqdm(range(total_frames)):
        t0 = time.time()

        fr1 = vr1[i].asnumpy()
        fr2 = vr2[i].asnumpy()
        rgb1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB)

        # ===== BALÓN =====
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

        # ===== JUGADORES =====
        res1 = PLAYER_DETECTION_MODEL.predict(fr1, imgsz=1792, iou=0.7)[0]
        det1 = sv.Detections.from_ultralytics(res1)
        ply1 = det1[det1.class_id == 1]

        res2 = PLAYER_DETECTION_MODEL.predict(fr2, imgsz=1792, iou=0.7)[0]
        det2 = sv.Detections.from_ultralytics(res2)
        ply2 = det2[det2.class_id == 1]

        n_dets = len(ply1.xyxy) + len(ply2.xyxy)

        # Construimos meta por cámara
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

        # ===== FUSIÓN MULTICÁMARA =====
        cams.sort(key=lambda d: d['score'], reverse=True)
        fused, used = [], [False]*len(cams)
        for a in range(len(cams)):
            if used[a]:
                continue
            base = cams[a]
            for b in range(a+1, len(cams)):
                if used[b]:
                    continue
                other = cams[b]
                if np.linalg.norm(base['pos'] - other['pos']) <= FUSE_POS_THR and reid_dist(base['emb'], other['emb']) <= FUSE_REID_THR:
                    base['pos'] = (0.7*base['pos'] + 0.3*other['pos']).astype(np.float32)
                    base['emb'] = 0.7*base['emb'] + 0.3*other['emb']
                    used[b] = True
            fused.append(base)
            used[a] = True

        n_fused = len(fused)

        # ===== TRACKER GLOBAL =====
        det_list = [(d['pos'], d['emb']) for d in fused]
        assigns = tracker.update(det_list)  # [(track_id, pos, emb)] en el MISMO orden que det_list
        n_tracks = len(assigns)

        # ===== ROSTER =====
        draw_pack = []  # (roster_id, cam, bbox, pos)
        current_rids = set()
        # detectar id switches: si un roster_id confirmado cambia de track_id
        id_switches_frame = 0

        for (t_id, pos, emb), det in zip(assigns, fused):
            rid, confirmed = assign_roster(i, t_id, emb, pos)
            if rid is None:
                continue
            if confirmed or roster_confirm_hits[rid] >= N_INIT:
                # id switch si ese rid ya tenía otro track_id distinto
                last_tid = roster_last_track.get(rid)
                if last_tid is not None and last_tid != t_id:
                    id_switches_cum += 1
                    id_switches_frame += 1
                roster_last_track[rid] = t_id

                draw_pack.append((rid, det['cam'], det['bbox'], pos))
                current_rids.add(rid)

        cleanup_roster(i)

        # ===== DIBUJO / EXPORT =====
        # balón
        if ball_bbox_cam is not None:
            camb, bb = ball_bbox_cam
            if camb == 'cam1':
                draw_box(fr1, bb, "Ball", color=(0, 255, 255))
            else:
                draw_box(fr2, bb, "Ball", color=(0, 255, 255))

        # jugadores (IDs visibles = roster_id) con color según equipo
        players_positions = {}
        for rid, cam, bbox, pos in draw_pack:
            players_positions[rid] = pos

            if rid in TEAM_BLACK_IDS:
                team_color = COLOR_BLACK
            elif rid in TEAM_WHITE_IDS:
                team_color = COLOR_WHITE
            else:
                team_color = COLOR_UNKNOWN

            if cam == 'cam1':
                draw_player_box(fr1, bbox, rid, team_color)
            else:
                draw_player_box(fr2, bbox, rid, team_color)

            posiciones_df.loc[len(posiciones_df)] = [
                i, rid, float(pos[0]), float(pos[1]),
                float(ball_x), float(ball_y)
            ]

        # ---------- RADAR con Voronoi estabilizado ----------
        # 1) cache anti-parpadeo (actualizo posiciones de rids presentes)
        for rid, pos in players_positions.items():
            last_pos_cache[rid] = (pos.astype(np.float32), i)

        # 2) set de puntos para radar/Voronoi (un punto por rid)
        pts_pitch = {}  # rid -> (x,y)
        hold_events = 0
        for rid in range(1, ROSTER_SIZE+1):
            if rid in players_positions:
                pts_pitch[rid] = players_positions[rid]
            else:
                if rid in last_pos_cache:
                    pos_cache, last_seen = last_pos_cache[rid]
                    if (i - last_seen) <= STICKY_MISS_TOL:
                        pts_pitch[rid] = pos_cache  # “pegado”
                        hold_events += 1

        # 3) Render base radar con puntos de color por equipo
        elements = {'points': [], 'paths': []}
        for rid, pxy in pts_pitch.items():
            if rid in TEAM_BLACK_IDS:
                c = (0, 0, 0)
            elif rid in TEAM_WHITE_IDS:
                c = (255, 255, 255)
            else:
                c = (127, 127, 127)
            elements['points'].append((pxy[0], pxy[1], c))
        if not (math.isnan(ball_x) or math.isnan(ball_y)):
            elements['points'].append((ball_x, ball_y, (0, 255, 255)))

        radar_image = pitch_renderer.draw(elements)
        orig_h, orig_w = radar_image.shape[:2]

        # 4) Mapeo cancha->pixeles radar
        scale = 0.1
        pad = 50
        pts_pix = []
        for rid, pxy in pts_pitch.items():
            x, y = float(pxy[0]), float(pxy[1])
            px = int(pad + scale * x)
            py = int(pad + scale * y)
            pts_pix.append((px, py, rid))

        # redimension radar
        radar_resized = cv2.resize(radar_image, (radar_width, radar_height))
        sx = radar_width / float(orig_w)
        sy = radar_height / float(orig_h)
        pts_viz = [(int(px * sx), int(py * sy), rid) for (px, py, rid) in pts_pix]

        # 5) Voronoi con contornos y sitios destacados
        radar_voronoi = radar_resized.copy()
        if len(pts_viz) >= 2:
            rect = (0, 0, radar_width, radar_height)
            subdiv = cv2.Subdiv2D(rect)

            pts_only = []
            for (px, py, rid) in pts_viz:
                px = min(max(px, 1), radar_width - 2)
                py = min(max(py, 1), radar_height - 2)
                subdiv.insert((px, py))
                pts_only.append((px, py, rid))

            facets, centers = subdiv.getVoronoiFacetList([])

            overlay = radar_voronoi.copy()

            def nearest_rid(cx, cy):
                best_d, best_rid = 1e9, None
                for (px, py, rid) in pts_only:
                    d = (px - cx)**2 + (py - cy)**2
                    if d < best_d:
                        best_d, best_rid = d, rid
                return best_rid

            # Relleno translúcido
            for f, c in zip(facets, centers):
                if f is None or len(f) == 0 or c is None:
                    continue
                poly = np.array(f, dtype=np.int32)
                cx, cy = int(c[0]), int(c[1])
                rid = nearest_rid(cx, cy)
                if rid is None:
                    continue
                if rid in TEAM_BLACK_IDS:
                    fill_color = VOR_COLOR_BLACK
                elif rid in TEAM_WHITE_IDS:
                    fill_color = VOR_COLOR_WHITE
                else:
                    fill_color = (160, 160, 160)
                cv2.fillPoly(overlay, [poly], fill_color)

            radar_voronoi = cv2.addWeighted(overlay, VOR_ALPHA, radar_voronoi, 1.0 - VOR_ALPHA, 0)

            # Contornos de celdas
            for f in facets:
                if f is None or len(f) == 0:
                    continue
                poly = np.array(f, dtype=np.int32)
                cv2.polylines(radar_voronoi, [poly], isClosed=True, color=(40, 40, 40), thickness=1)

            # Sitios con anillo
            for (px, py, rid) in pts_only:
                if rid in TEAM_BLACK_IDS:
                    inner = (0, 0, 0)
                    outer = (255, 255, 255)
                elif rid in TEAM_WHITE_IDS:
                    inner = (255, 255, 255)
                    outer = (0, 0, 0)
                else:
                    inner = (127, 127, 127)
                    outer = (60, 60, 60)
                cv2.circle(radar_voronoi, (px, py), 4, outer, thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(radar_voronoi, (px, py), 2, inner, thickness=-1, lineType=cv2.LINE_AA)

        # 6) Pegamos radar con voronoi sobre la cam1
        x, y = radar_position
        alpha = 0.6
        fr1_overlay = fr1.copy()
        for c in range(3):
            fr1_overlay[y:y+radar_height, x:x+radar_width, c] = (
                alpha * radar_voronoi[:, :, c] +
                (1 - alpha) * fr1_overlay[y:y+radar_height, x:x+radar_width, c]
            )

        fr1_bgr = cv2.cvtColor(fr1_overlay, cv2.COLOR_RGB2BGR)
        fr2_bgr = cv2.cvtColor(fr2, cv2.COLOR_RGB2BGR)

        # frame side-by-side en un único MP4
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h1, :w1] = fr1_bgr
        canvas[:h2, w1:w1+w2] = fr2_bgr
        video_out.write(canvas)

        # ======== MÉTRICAS (fin de frame) ========
        # duplicados residuales: pares de rids distintos demasiado cerca
        dup_pairs = 0
        rids_list = list(pts_pitch.keys())
        for a in range(len(rids_list)):
            ra = rids_list[a]
            pa = np.array(pts_pitch[ra], dtype=np.float32)
            for b in range(a+1, len(rids_list)):
                rb = rids_list[b]
                pb = np.array(pts_pitch[rb], dtype=np.float32)
                if np.linalg.norm(pa - pb) < DUP_POS_THR:
                    dup_pairs += 1

        n_drawn = len(players_positions)
        n_voronoi_points = len(pts_pitch)
        fps_frame = 0.0
        dt = time.time() - t0
        if dt > 1e-6:
            fps_frame = 1.0 / dt

        metrics_rows.append({
            "frame": i,
            "fps_frame": round(fps_frame, 3),
            "n_dets": int(n_dets),
            "n_fused": int(n_fused),
            "n_tracks": int(n_tracks),
            "n_drawn": int(n_drawn),
            "n_voronoi_points": int(n_voronoi_points),
            "n_hold_points": int(hold_events),
            "dup_pairs": int(dup_pairs),
            "id_switches_cum": int(id_switches_cum),
        })

    video_out.release()

    # ===== GUARDAR MÉTRICAS =====
    dfm = pd.DataFrame(metrics_rows)
    dfm.to_csv("codes/data/METRICS.csv", index=False)

    # Resumen
    if len(dfm) > 0:
        fps_avg = dfm["fps_frame"].mean()
        dets_avg = dfm["n_dets"].mean()
        fused_avg = dfm["n_fused"].mean()
        tracks_avg = dfm["n_tracks"].mean()
        drawn_avg = dfm["n_drawn"].mean()
        hold_avg = dfm["n_hold_points"].mean()
        dup_avg = dfm["dup_pairs"].mean()
        id_switches_total = int(dfm["id_switches_cum"].iloc[-1])
    else:
        fps_avg = dets_avg = fused_avg = tracks_avg = drawn_avg = hold_avg = dup_avg = 0.0
        id_switches_total = 0

    summary = []
    summary.append(f"Frames: {len(dfm)}")
    summary.append(f"FPS promedio: {fps_avg:.3f}")
    summary.append(f"Detecciones promedio por frame: {dets_avg:.2f}")
    summary.append(f"Fused promedio por frame: {fused_avg:.2f}")
    summary.append(f"Tracks promedio por frame: {tracks_avg:.2f}")
    summary.append(f"Dibujados (roster confirmados) promedio: {drawn_avg:.2f}")
    summary.append(f"Puntos Voronoi promedio: {int(dfm['n_voronoi_points'].mean() if len(dfm)>0 else 0)}")
    summary.append(f"Puntos ‘pegados’ (anti-parpadeo) promedio: {hold_avg:.2f}")
    summary.append(f"Duplicados residuales promedio (pares cercanos): {dup_avg:.2f}")
    summary.append(f"ID switches totales: {id_switches_total}")

    with open("codes/data/METRICS_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    # Export posiciones
    posiciones_df.to_excel("codes/data/Posiciones-jugadores-balon-multicam.xlsx", index=False)

# =========================
# EJECUCIÓN
# =========================
process_dual_camera(
    VIDEO_PATH_CAM1, VIDEO_PATH_CAM2,
    view_transformer1, view_transformer2
)

print("¡Listo! Video único con radar+Voronoi y métricas:")
print("-", TARGET_VIDEO_OUTPUT)
print("Excel de posiciones:", "codes/data/Posiciones-jugadores-balon-multicam.xlsx")
print("Métricas por frame:", "codes/data/METRICS.csv")
print("Resumen de métricas:", "codes/data/METRICS_SUMMARY.txt")

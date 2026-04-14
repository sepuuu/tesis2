# -*- coding: utf-8 -*-
import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"

import math
import time
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchreid
import pandas as pd
import supervision as sv
from decord import VideoReader, cpu
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

import config
from configs.drawing import PitchRenderer
from configs.soccer import SoccerPitchConfiguration
from configs.view_transformer import ViewTransformer
from stable_id import StableIDAssigner
from utils.ball_setup import callback
from utils.drawing_utils import draw_box, draw_player_box

_ACTIVE_RUN = None  # usado para dumpear artefactos incluso si hay excepción


def _codes_data_path(filename: str) -> str:
    data_dir = str(getattr(config, "CODES_DATA_DIR", os.path.join("codes", "data")))
    return os.path.join(data_dir, filename)


def _fmt_dur(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =========================
# CONFIGURACIONES CENTRALES
# =========================
PATHS_CFG = config.PATHS
DETECTION_CFG = config.DETECTION_CONFIG

CONFIG = SoccerPitchConfiguration()
LOGGER.setLevel("ERROR")

VIDEO_PATH_CAM1 = PATHS_CFG.video_cam1
VIDEO_PATH_CAM2 = PATHS_CFG.video_cam2
TARGET_VIDEO_OUTPUT = PATHS_CFG.video_output


# =========================
# MODELOS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

reid_model = torchreid.models.build_model(
    name=PATHS_CFG.reid_model_name,
    num_classes=12,
    pretrained=False,
)
checkpoint = torch.load(PATHS_CFG.reid_checkpoint, map_location=device)
reid_model.load_state_dict(checkpoint["state_dict"], strict=False)
reid_model = reid_model.to(device).eval()

PLAYER_DETECTION_MODEL = YOLO(PATHS_CFG.player_detector_path, task="detect")

preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# =========================
# HOMOGRAFÍAS (hardcoded)
# =========================
points_img1 = np.array(
    [
        [912, 203],
        [1003, 213],
        [1063, 221],
        [1180, 237],
        [1237, 243],
        [1358, 265],
        [965, 220],
        [1216, 253],
        [898, 270],
        [42, 768],
        [149, 369],
        [604, 499],
    ],
    dtype=np.float32,
)
points_pitch1 = np.array(
    [
        [0, 0],
        [0, 500],
        [0, 800],
        [0, 1200],
        [0, 1500],
        [0, 2000],
        [500, 500],
        [500, 1500],
        [2000, 1000],
        [4000, 1500],
        [3500, 500],
        [3500, 1500],
    ],
    dtype=np.float32,
)
view_transformer1 = ViewTransformer(source=points_img1, target=points_pitch1)

points_img2 = np.array(
    [
        [256, 829],
        [811, 596],
        [451, 506],
        [1066, 413],
        [1497, 360],
        [1368, 359],
        [1312, 359],
        [1206, 360],
        [1164, 360],
        [1082, 359],
        [1348, 371],
        [1120, 370],
    ],
    dtype=np.float32,
)
points_pitch2 = np.array(
    [
        [0, 500],
        [500, 500],
        [500, 1500],
        [2000, 1000],
        [4000, 0],
        [4000, 500],
        [4000, 800],
        [4000, 1200],
        [4000, 1500],
        [4000, 2000],
        [3500, 500],
        [3500, 1500],
    ],
    dtype=np.float32,
)
view_transformer2 = ViewTransformer(source=points_img2, target=points_pitch2)


# =========================
# AUX
# =========================
def extract_embedding(frame_bgr, bbox, pad=0.12, min_hw=10):
    # Usa configuración para permitir ajustes rápidos sin tocar llamadas
    pad = float(getattr(config, "EMB_PAD", pad))
    min_hw = int(getattr(config, "EMB_MIN_HW", min_hw))
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    x1 -= int(pad * bw)
    x2 += int(pad * bw)
    y1 -= int(pad * bh)
    y2 += int(pad * bh)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if (x2 - x1) < min_hw or (y2 - y1) < min_hw:
        return None

    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_t = preprocess(crop_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = reid_model(crop_t).cpu().numpy()[0].astype(np.float32)
    return emb


def l2norm(e, eps=1e-12):
    e = e.astype(np.float32)
    return e / (float(np.sqrt((e * e).sum())) + eps)


def xyxy_to_ltwh(b):
    x1, y1, x2, y2 = map(float, b)
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_xyxy_to_pitch_pos(vt: ViewTransformer, bbox_xyxy):
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    mode = str(getattr(config, "PITCH_ANCHOR_MODE", "pseudo_foot")).lower()
    if mode == "center":
        pt = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
    elif mode == "bc":
        pt = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
    else:  # pseudo_foot
        frac = float(getattr(config, "PSEUDO_FOOT_FRAC", 0.08))
        h = (y2 - y1)
        pt = np.array([[(x1 + x2) / 2.0, y2 - frac * h]], dtype=np.float32)
    p = vt.transform_points(pt)[0].astype(np.float32)
    return p


# ============================================================
# Anti-teleport: suavizar punto en IMAGEN (bottom-center) por stable_id
# y congelar si detecta "tipo-1" (pitch grande con img chica).
# ============================================================
_pos_state_cam1 = {}
_pos_state_cam2 = {}


def _bc_from_bbox(bbox_xyxy):
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return np.array([((x1 + x2) / 2.0), y2], dtype=np.float32)  # (bcx, bcy)


def _smooth_project_pos(state: dict, sid: int, bbox_xyxy, vt: ViewTransformer, frame_idx: int,
                        beta: float, spike_pitch: float, spike_img: float, ttl: int):
    bc = _bc_from_bbox(bbox_xyxy)

    st = state.get(sid)
    if st is None:
        p = vt.transform_points(bc.reshape(1, 2).astype(np.float32))[0].astype(np.float32)
        state[sid] = {"bc": bc, "p": p, "t": frame_idx}
        return p

    # limpiar estado viejo (si el track desapareció mucho rato)
    if frame_idx - st["t"] > ttl:
        p = vt.transform_points(bc.reshape(1, 2).astype(np.float32))[0].astype(np.float32)
        state[sid] = {"bc": bc, "p": p, "t": frame_idx}
        return p

    bc_prev = st["bc"]
    # EMA en imagen
    bc_sm = (1.0 - beta) * bc_prev + beta * bc

    img_delta = float(np.linalg.norm(bc_sm - bc_prev))

    p_new = vt.transform_points(bc_sm.reshape(1, 2).astype(np.float32))[0].astype(np.float32)
    pitch_delta = float(np.linalg.norm(p_new - st["p"]))

    # guard tipo-1: si la homografía amplifica un jitter pequeño
    if pitch_delta > spike_pitch and img_delta < spike_img:
        p_new = st["p"]
        bc_sm = bc_prev

    st["bc"] = bc_sm
    st["p"] = p_new
    st["t"] = frame_idx
    return p_new


def _create_ultralytics_tracker(*, frame_rate: float):
    tracker_yaml = getattr(config, "ULTRALYTICS_TRACKER_YAML", "botsort.yaml")
    tracker_path = check_yaml(str(tracker_yaml))
    cfg = IterableSimpleNamespace(**yaml_load(tracker_path))
    tracker_type = str(getattr(cfg, "tracker_type", "botsort")).strip().lower()
    tracker_cls = {"botsort": BOTSORT, "bytetrack": BYTETracker}.get(tracker_type)
    if tracker_cls is None:
        raise ValueError(f"tracker_type no soportado: {tracker_type!r} (usa 'botsort' o 'bytetrack').")
    return tracker_cls(args=cfg, frame_rate=float(frame_rate))


def _load_offline_gid_map(path: str) -> dict:
    """
    Espera un JSON con mapping estable por cámara, por ejemplo:
      { "cam1": { "7": 3 }, "cam2": { "3": 3 } }
    Retorna: {"cam1": {7: 3, ...}, "cam2": {...}}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    maps = {"cam1": {}, "cam2": {}}

    def _ingest(cam_label, obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            for sid, gid in obj.items():
                maps[cam_label][int(sid)] = int(gid)
            return
        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                sid = item.get("stable_id", item.get("sid"))
                gid = item.get("gid", item.get("global_id", item.get("id")))
                if sid is None or gid is None:
                    continue
                maps[cam_label][int(sid)] = int(gid)

    if isinstance(data, dict):
        for cam_label in ("cam1", "cam2"):
            if cam_label in data:
                _ingest(cam_label, data.get(cam_label))
            elif f"{cam_label}_map" in data:
                _ingest(cam_label, data.get(f"{cam_label}_map"))

        # soporte alternativo: lista plana bajo "entries"/"mapping"
        if not maps["cam1"] and not maps["cam2"]:
            for key in ("entries", "mapping", "map"):
                if key not in data:
                    continue
                obj = data.get(key)
                if not isinstance(obj, list):
                    continue
                for item in obj:
                    if not isinstance(item, dict):
                        continue
                    cam_label = item.get("cam")
                    if cam_label not in ("cam1", "cam2"):
                        continue
                    sid = item.get("stable_id", item.get("sid"))
                    gid = item.get("gid", item.get("global_id", item.get("id")))
                    if sid is None or gid is None:
                        continue
                    maps[cam_label][int(sid)] = int(gid)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            cam_label = item.get("cam")
            if cam_label not in ("cam1", "cam2"):
                continue
            sid = item.get("stable_id", item.get("sid"))
            gid = item.get("gid", item.get("global_id", item.get("id")))
            if sid is None or gid is None:
                continue
            maps[cam_label][int(sid)] = int(gid)

    return maps


def _dump_tracklet_artifacts(run_dir: str, prefix: str, stable: StableIDAssigner) -> None:
    os.makedirs(run_dir, exist_ok=True)

    stable_ids_with_emb = sorted(int(s) for s in stable.stable_emb.keys())
    if stable_ids_with_emb:
        emb_mat = np.stack(
            [stable.stable_emb[sid].astype(np.float32) for sid in stable_ids_with_emb], axis=0
        )
    else:
        emb_mat = np.zeros((0, 0), dtype=np.float32)

    np.savez_compressed(
        os.path.join(run_dir, f"{prefix}_embeddings.npz"),
        stable_ids=np.asarray(stable_ids_with_emb, dtype=np.int32),
        embeddings=emb_mat,
    )

    stable_ids_all = set()
    stable_ids_all.update(int(s) for s in stable.stable_first_seen.keys())
    stable_ids_all.update(int(s) for s in stable.stable_last_seen.keys())
    stable_ids_all.update(int(s) for s in stable.stable_obs_count.keys())
    stable_ids_all.update(int(s) for s in stable.stable_emb_obs_count.keys())
    stable_ids_all.update(int(s) for s in stable.stable_last_bbox.keys())
    stable_ids_all.update(int(s) for s in stable.stable_emb.keys())
    stable_ids_all = sorted(stable_ids_all)

    rows = []
    for sid in stable_ids_all:
        x1, y1, x2, y2 = stable.stable_last_bbox.get(sid, (math.nan, math.nan, math.nan, math.nan))
        rows.append(
            dict(
                stable_id=int(sid),
                first_frame=int(stable.stable_first_seen.get(sid, -1)),
                last_frame=int(stable.stable_last_seen.get(sid, -1)),
                n_obs=int(stable.stable_obs_count.get(sid, 0)),
                n_obs_emb=int(stable.stable_emb_obs_count.get(sid, 0)),
                has_emb=bool(sid in stable.stable_emb),
                bbox_x1=float(x1),
                bbox_y1=float(y1),
                bbox_x2=float(x2),
                bbox_y2=float(y2),
            )
        )

    pd.DataFrame(rows).to_csv(os.path.join(run_dir, f"{prefix}_tracklet_summary.csv"), index=False)


def _compute_tracklet_metrics(df_tracks: pd.DataFrame, *, prefix: str) -> dict:
    if df_tracks is None or df_tracks.empty:
        return {
            f"{prefix}_n_tracklets": 0,
            f"{prefix}_tracklet_len_median_frames": math.nan,
        }

    if "stable_id" not in df_tracks.columns or "frame" not in df_tracks.columns:
        return {
            f"{prefix}_n_tracklets": math.nan,
            f"{prefix}_tracklet_len_median_frames": math.nan,
        }

    grp = (
        df_tracks.groupby("stable_id", sort=False)["frame"]
        .agg(first_frame="min", last_frame="max", n_obs="count")
        .reset_index(drop=False)
    )
    if grp.empty:
        return {
            f"{prefix}_n_tracklets": 0,
            f"{prefix}_tracklet_len_median_frames": math.nan,
        }

    durations = (grp["last_frame"] - grp["first_frame"] + 1).astype(np.int64)
    return {
        f"{prefix}_n_tracklets": int(len(grp)),
        f"{prefix}_tracklet_len_median_frames": float(durations.median()) if len(durations) else math.nan,
    }


def _compute_id_churn_near_ball(
    df_tracks: pd.DataFrame,
    *,
    id_col: str = "stable_id",
    lookahead_frames: int = 2,
) -> dict:
    """
    "Churn" barato sin GT: cuenta cambios del jugador más cercano al balón
    que se revierten rápido (A->B->A o A->B->B->A) en frames consecutivos.
    """
    if df_tracks is None or df_tracks.empty:
        return {"n_frames": 0, "switches_adjacent": 0, "churn_events": 0}

    required = {"frame", id_col, "pos_x", "pos_y", "ball_x", "ball_y"}
    if not required.issubset(set(df_tracks.columns)):
        return {"n_frames": math.nan, "switches_adjacent": math.nan, "churn_events": math.nan}

    df = df_tracks[list(required) + (["conf"] if "conf" in df_tracks.columns else [])].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["frame", id_col, "pos_x", "pos_y", "ball_x", "ball_y"])
    if df.empty:
        return {"n_frames": 0, "switches_adjacent": 0, "churn_events": 0}

    df["frame"] = df["frame"].astype(np.int64)
    df[id_col] = df[id_col].astype(np.int64)
    conf = df["conf"] if "conf" in df.columns else 0.0
    df["_conf"] = pd.to_numeric(conf, errors="coerce").fillna(-1.0).astype(np.float32)
    df["_dist"] = np.hypot(
        (df["pos_x"].astype(np.float32) - df["ball_x"].astype(np.float32)),
        (df["pos_y"].astype(np.float32) - df["ball_y"].astype(np.float32)),
    )

    # nearest per frame (tie-break by conf)
    df = df.sort_values(["frame", "_dist", "_conf"], ascending=[True, True, False], kind="mergesort")
    nearest = df.drop_duplicates(subset=["frame"], keep="first")[["frame", id_col]].rename(
        columns={id_col: "nearest_id"}
    )
    nearest = nearest.sort_values(["frame"], ascending=True)

    frames = nearest["frame"].to_numpy(dtype=np.int64, copy=False)
    ids = nearest["nearest_id"].to_numpy(dtype=np.int64, copy=False)
    n = int(len(frames))
    if n <= 1:
        return {"n_frames": n, "switches_adjacent": 0, "churn_events": 0}

    switches = 0
    for j in range(1, n):
        if frames[j] == frames[j - 1] + 1 and ids[j] != ids[j - 1]:
            switches += 1

    churn = 0
    for j in range(1, n - 1):
        if frames[j] != frames[j - 1] + 1:
            continue
        if ids[j] == ids[j - 1]:
            continue

        # A-B-A
        if frames[j + 1] == frames[j] + 1 and ids[j + 1] == ids[j - 1]:
            churn += 1
            continue

        if lookahead_frames >= 2 and j + 2 < n:
            # A-B-B-A
            if (
                frames[j + 1] == frames[j] + 1
                and frames[j + 2] == frames[j] + 2
                and ids[j + 1] == ids[j]
                and ids[j + 2] == ids[j - 1]
            ):
                churn += 1

    return {"n_frames": n, "switches_adjacent": int(switches), "churn_events": int(churn)}


def _compute_crosscam_consistency(
    df_cam1: pd.DataFrame,
    df_cam2: pd.DataFrame,
    *,
    gid_col: str = "gid",
    pos_thr: float = 80.0,
) -> dict:
    if (
        df_cam1 is None
        or df_cam2 is None
        or df_cam1.empty
        or df_cam2.empty
        or gid_col not in df_cam1.columns
        or gid_col not in df_cam2.columns
    ):
        return {"n_pairs": 0, "consistency_pct": math.nan, "dist_median": math.nan, "dist_p80": math.nan, "dist_p95": math.nan}

    df1 = df_cam1.copy()
    df2 = df_cam2.copy()
    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna(subset=["frame", gid_col, "pos_x", "pos_y"])
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna(subset=["frame", gid_col, "pos_x", "pos_y"])
    if df1.empty or df2.empty:
        return {"n_pairs": 0, "consistency_pct": math.nan, "dist_median": math.nan, "dist_p80": math.nan, "dist_p95": math.nan}

    for df in (df1, df2):
        df["frame"] = df["frame"].astype(np.int64)
        df[gid_col] = df[gid_col].astype(np.int64)
        if "conf" in df.columns:
            df["_conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0).astype(np.float32)
        else:
            df["_conf"] = 0.0

    # 1 row per (frame,gid): keep best conf
    df1 = df1.sort_values(["frame", gid_col, "_conf"], ascending=[True, True, False], kind="mergesort")
    df2 = df2.sort_values(["frame", gid_col, "_conf"], ascending=[True, True, False], kind="mergesort")
    df1 = df1.drop_duplicates(subset=["frame", gid_col], keep="first")
    df2 = df2.drop_duplicates(subset=["frame", gid_col], keep="first")

    merged = df1.merge(df2, on=["frame", gid_col], suffixes=("_c1", "_c2"))
    if merged.empty:
        return {"n_pairs": 0, "consistency_pct": math.nan, "dist_median": math.nan, "dist_p80": math.nan, "dist_p95": math.nan}

    dist = np.hypot(
        (merged["pos_x_c1"].astype(np.float32) - merged["pos_x_c2"].astype(np.float32)),
        (merged["pos_y_c1"].astype(np.float32) - merged["pos_y_c2"].astype(np.float32)),
    )
    dist = dist.astype(np.float32)
    ok = dist <= float(pos_thr)
    dist_p80 = float(np.percentile(dist, 80)) if len(dist) else math.nan
    dist_p95 = float(np.percentile(dist, 95)) if len(dist) else math.nan
    return {
        "n_pairs": int(len(merged)),
        "consistency_pct": float(ok.mean() * 100.0) if len(ok) else math.nan,
        "dist_median": float(np.median(dist)) if len(dist) else math.nan,
        "dist_p80": dist_p80,
        "dist_p95": dist_p95,
    }


def _compute_dashboard_row(
    *,
    run_dir: str,
    pipeline_stage: str | None,
    fps: float | None,
    total_frames: int | None,
    frames_processed: int | None,
    df_cam1: pd.DataFrame,
    df_cam2: pd.DataFrame,
    df_ball: pd.DataFrame,
) -> dict:
    row: dict = dict(
        run_dir=str(run_dir),
        pipeline_stage=(str(pipeline_stage) if pipeline_stage is not None else None),
        fps=(float(fps) if fps is not None else math.nan),
        total_frames=(int(total_frames) if total_frames is not None else math.nan),
        frames_processed=(int(frames_processed) if frames_processed is not None else math.nan),
    )

    try:
        row["ball_visible_frames"] = int(
            df_ball.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["ball_x", "ball_y"])
            .shape[0]
        )
    except Exception:
        row["ball_visible_frames"] = math.nan

    row.update(_compute_tracklet_metrics(df_cam1, prefix="cam1"))
    row.update(_compute_tracklet_metrics(df_cam2, prefix="cam2"))

    churn1 = _compute_id_churn_near_ball(df_cam1, id_col="stable_id")
    churn2 = _compute_id_churn_near_ball(df_cam2, id_col="stable_id")
    row.update({f"cam1_ball_nearest_{k}": v for k, v in churn1.items()})
    row.update({f"cam2_ball_nearest_{k}": v for k, v in churn2.items()})

    default_thr = float(getattr(getattr(config, "FUSION_CONFIG", object()), "strict_pos_thr", 80.0))
    cross_thr = float(getattr(config, "DASHBOARD_CROSSCAM_POS_THR", default_thr))
    if not (isinstance(cross_thr, (int, float)) and cross_thr > 0):
        cross_thr = default_thr
    cross = _compute_crosscam_consistency(df_cam1, df_cam2, pos_thr=cross_thr)
    row["crosscam_pos_thr"] = float(cross_thr)
    row.update({f"crosscam_{k}": v for k, v in cross.items()})
    return row


def _write_dashboard_csv(*, run_dir: str, row: dict, also_write_codes_data: bool) -> None:
    dash_df = pd.DataFrame([row])
    dash_df.to_csv(os.path.join(run_dir, "DASHBOARD.csv"), index=False)
    if also_write_codes_data:
        dash_df.to_csv(_codes_data_path("DASHBOARD.csv"), index=False)


def _finalize_active_run(*, aborted: bool, error_repr: str | None = None) -> None:
    global _ACTIVE_RUN
    state = _ACTIVE_RUN
    if not isinstance(state, dict) or state.get("finalized"):
        return
    state["finalized"] = True

    run_dir = state.get("run_dir")
    if not run_dir:
        _ACTIVE_RUN = None
        return

    also_write_codes_data = bool(state.get("also_write_codes_data", False))
    live_window = state.get("live_window")

    for writer_key in ("video_out", "video_out_local", "points_writer"):
        writer = state.get(writer_key)
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

    try:
        if live_window:
            cv2.destroyWindow(live_window)
    except Exception:
        pass

    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception:
        pass

    metrics_rows = state.get("metrics_rows") or []
    track_rows_cam1 = state.get("track_rows_cam1") or []
    track_rows_cam2 = state.get("track_rows_cam2") or []
    ball_rows = state.get("ball_rows") or []

    try:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(os.path.join(run_dir, "METRICS.csv"), index=False)
        if also_write_codes_data:
            metrics_df.to_csv(_codes_data_path("METRICS.csv"), index=False)
    except Exception as dump_exc:
        print(f"[WARN] No se pudo escribir METRICS.csv: {dump_exc!r}")

    df_cam1 = pd.DataFrame(track_rows_cam1)
    df_cam2 = pd.DataFrame(track_rows_cam2)
    df_ball = pd.DataFrame(ball_rows)

    try:
        df_cam1.to_csv(os.path.join(run_dir, "c1_tracks.csv"), index=False)
        df_cam2.to_csv(os.path.join(run_dir, "c2_tracks.csv"), index=False)
        df_ball.to_csv(os.path.join(run_dir, "ball.csv"), index=False)
    except Exception as dump_exc:
        print(f"[WARN] No se pudo escribir tracks/ball CSV: {dump_exc!r}")

    try:
        df_all = pd.concat([df_cam1, df_cam2], ignore_index=True)
        excel_path = os.path.join(run_dir, "Posiciones-jugadores-balon-multicam.xlsx")
        df_all.to_excel(excel_path, index=False)
        if also_write_codes_data:
            df_all.to_excel(_codes_data_path("Posiciones-jugadores-balon-multicam.xlsx"), index=False)
    except Exception as dump_exc:
        print(f"[WARN] No se pudo escribir Excel combinado: {dump_exc!r}")

    try:
        row = _compute_dashboard_row(
            run_dir=str(run_dir),
            pipeline_stage=state.get("pipeline_stage"),
            fps=state.get("fps"),
            total_frames=state.get("total_frames"),
            frames_processed=int(len(metrics_rows)),
            df_cam1=df_cam1,
            df_cam2=df_cam2,
            df_ball=df_ball,
        )
        _write_dashboard_csv(run_dir=str(run_dir), row=row, also_write_codes_data=also_write_codes_data)
    except Exception as dump_exc:
        print(f"[WARN] No se pudo escribir DASHBOARD.csv: {dump_exc!r}")

    try:
        meta = dict(
            fps=state.get("fps"),
            total_frames=state.get("total_frames"),
            frames_processed=int(len(metrics_rows)),
            aborted=bool(aborted),
            error=(error_repr if aborted else None),
            video_cam1=state.get("video_cam1"),
            video_cam2=state.get("video_cam2"),
            pipeline_stage=state.get("pipeline_stage"),
            run_dir=str(run_dir),
            offline_map_path=state.get("offline_map_path"),
            output_video=state.get("output_video"),
            output_video_local_ids=state.get("output_video_local_ids"),
            output_video_points=state.get("output_video_points"),
        )
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as dump_exc:
        print(f"[WARN] No se pudo escribir meta.json: {dump_exc!r}")

    try:
        stable_cam1 = state.get("stable_cam1")
        stable_cam2 = state.get("stable_cam2")
        if stable_cam1 is not None:
            _dump_tracklet_artifacts(run_dir, "c1", stable_cam1)
        if stable_cam2 is not None:
            _dump_tracklet_artifacts(run_dir, "c2", stable_cam2)
    except Exception as dump_exc:
        print(f"[WARN] No se pudo dumpear embeddings/summary: {dump_exc!r}")
    finally:
        _ACTIVE_RUN = None


def process_dual_camera(video_cam1, video_cam2, vt1, vt2):
    vr1 = VideoReader(video_cam1, ctx=cpu(0))
    vr2 = VideoReader(video_cam2, ctx=cpu(0))
    total_frames = min(len(vr1), len(vr2))
    fps = float(vr1.get_avg_fps())

    run_dir = str(getattr(config, "RUN_DIR", "runs/default"))
    pipeline_stage = str(getattr(config, "PIPELINE_STAGE", "export")).strip().lower()
    offline_map_path = str(
        getattr(config, "OFFLINE_MAP_PATH", os.path.join(run_dir, "crosscam_map.json"))
    )
    also_write_codes_data = bool(getattr(config, "ALSO_WRITE_CODES_DATA", False))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "debug"), exist_ok=True)
    if also_write_codes_data:
        os.makedirs(str(getattr(config, "CODES_DATA_DIR", os.path.join("codes", "data"))), exist_ok=True)

    gid_map = {"cam1": {}, "cam2": {}}
    if pipeline_stage == "render":
        if not os.path.exists(offline_map_path):
            raise FileNotFoundError(
                f"PIPELINE_STAGE='render' pero no existe OFFLINE_MAP_PATH: {offline_map_path}"
            )
        gid_map = _load_offline_gid_map(offline_map_path)
        print(f"[STAGE] render | offline map: {offline_map_path}")
    else:
        if pipeline_stage != "export":
            print(f"[WARN] PIPELINE_STAGE desconocido '{pipeline_stage}', usando 'export'.")
            pipeline_stage = "export"
        print(f"[STAGE] export | run dir: {run_dir}")

    render_offline = pipeline_stage == "render" and bool(getattr(config, "RENDER_USE_STITCHED", False))

    tracker_cam1 = tracker_cam2 = None
    stable_cam1 = stable_cam2 = None
    slicer_ball = None
    offline_tracks = {"cam1": {}, "cam2": {}}
    ball_idx = {}

    def _normalize_manual_overrides(obj: dict) -> dict:
        """
        Formato único esperado:
          { "players": { "1": { "cam1": [1,38], "cam2": [1] }, "2": {...} } }
        Retorna dict con cam1_merges/cam2_merges/crosscam_pairs (listas).
        """
        out = {"cam1_merges": [], "cam2_merges": [], "crosscam_pairs": []}
        if not isinstance(obj, dict):
            return out
        players = obj.get("players")
        if not isinstance(players, dict):
            return out
        cam1_seen = set()
        cam2_seen = set()
        cross_seen = set()
        for gid_key, data in players.items():
            try:
                gid = int(gid_key)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            cam1_ids = data.get("cam1") or data.get("cam1_ids") or []
            cam2_ids = data.get("cam2") or data.get("cam2_ids") or []
            for sid in cam1_ids:
                try:
                    sid_int = int(sid)
                except Exception:
                    continue
                if (sid_int, gid) not in cam1_seen:
                    out["cam1_merges"].append([sid_int, gid])
                    cam1_seen.add((sid_int, gid))
            for sid in cam2_ids:
                try:
                    sid_int = int(sid)
                except Exception:
                    continue
                if (sid_int, gid) not in cam2_seen:
                    out["cam2_merges"].append([sid_int, gid])
                    cam2_seen.add((sid_int, gid))
            if gid not in cross_seen:
                out["crosscam_pairs"].append([gid, gid])
                cross_seen.add(gid)
        return out

    # manual overrides (merges intra-cámara) y crosscam_pairs (para siguiente fase)
    manual_overrides = {"cam1_merges": [], "cam2_merges": [], "crosscam_pairs": []}
    manual_path = os.path.join(run_dir, "manual_overrides.json")
    if pipeline_stage == "render" and os.path.exists(manual_path):
        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                manual_overrides_raw = json.load(f)
            manual_overrides = _normalize_manual_overrides(manual_overrides_raw)
        except Exception as exc:
            print(f"[WARN] No se pudo leer manual_overrides.json: {exc!r}")

    def remap_sid(cam_key, sid):
        merges = manual_overrides.get(f"{cam_key}_merges", []) if isinstance(manual_overrides, dict) else []
        m = {int(a): int(b) for a, b in merges}
        sid = int(sid)
        seen = set()
        while sid in m and sid not in seen:
            seen.add(sid)
            sid = int(m[sid])
        return sid

    if not render_offline:
        tracker_cam1 = _create_ultralytics_tracker(frame_rate=fps)
        tracker_cam2 = _create_ultralytics_tracker(frame_rate=fps)
        ds_dead_after = int(getattr(config, "STABLEID_DS_DEAD_AFTER", 0))
        stable_cam1 = StableIDAssigner(max_gap_frames=300, reattach_cos_thr=0.25, ds_dead_after=ds_dead_after)
        stable_cam2 = StableIDAssigner(max_gap_frames=300, reattach_cos_thr=0.25, ds_dead_after=ds_dead_after)

        slicer_ball = sv.InferenceSlicer(
            callback=callback,
            slice_wh=(640, 640),
            overlap_ratio_wh=None,
            overlap_wh=(0, 0),
            overlap_filter=sv.OverlapFilter.NONE,
        )
    else:
        # Cargar tracks stitched para render offline sin retrack
        for cam in ("cam1", "cam2"):
            path = os.path.join(run_dir, f"{'c1' if cam=='cam1' else 'c2'}_tracks_stitched.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"[RENDER_OFFLINE] Falta {path}")
            df = pd.read_csv(path)
            df["frame"] = df["frame"].astype(int)
            offline_tracks[cam] = {int(f): g for f, g in df.groupby("frame")}
        ball_path = os.path.join(run_dir, "ball.csv")
        if os.path.exists(ball_path):
            df_ball_off = pd.read_csv(ball_path)
            if "frame" in df_ball_off.columns:
                df_ball_off["frame"] = df_ball_off["frame"].astype(int)
                ball_idx = {int(f): g.iloc[0] for f, g in df_ball_off.groupby("frame")}
        print("[STAGE] render offline desde CSV stitched")
        # Limitar total_frames al máximo frame disponible en los CSV stitched (evita loops largos sin datos)
        max_frame_tracks = max(
            [
                max(offline_tracks["cam1"].keys(), default=-1),
                max(offline_tracks["cam2"].keys(), default=-1),
            ]
        )
        max_frame_ball = max(ball_idx.keys(), default=-1)
        max_frame = max(max_frame_tracks, max_frame_ball)
        if max_frame >= 0:
            total_frames = min(total_frames, max_frame + 1)

    # outputs
    local_ids_output = os.path.splitext(TARGET_VIDEO_OUTPUT)[0] + "_local_ids.mp4"
    points_video_path = os.path.splitext(TARGET_VIDEO_OUTPUT)[0] + "_points_only.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_dir = os.path.dirname(TARGET_VIDEO_OUTPUT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    h1, w1 = vr1[0].shape[0], vr1[0].shape[1]
    h2, w2 = vr2[0].shape[0], vr2[0].shape[1]
    out_h = max(h1, h2)
    out_w = w1 + w2

    video_out = cv2.VideoWriter(TARGET_VIDEO_OUTPUT, fourcc, fps, (out_w, out_h))
    video_out_local = cv2.VideoWriter(local_ids_output, fourcc, fps, (out_w, out_h))

    pitch_renderer = PitchRenderer(
        config=CONFIG,
        scale=0.1,
        padding=50,
        background_color=sv.Color(34, 139, 34),
        line_color=sv.Color.WHITE,
    )

    radar_width = int(w1 * 0.4)
    radar_height = int(h1 * 0.3)
    radar_position = (int((w1 - radar_width) / 2), int(h1 - radar_height - 20))
    points_writer = cv2.VideoWriter(points_video_path, fourcc, fps, (radar_width, radar_height))

    # metrics / export
    metrics_rows = []
    track_rows_cam1 = []
    track_rows_cam2 = []
    ball_rows = []
    last_ball_pos = None
    last_ball_frame = None
    cum_ball_ms = 0.0
    cum_frame_ms = 0.0
    n_frames_proc = 0
    cum_ball_ms = 0.0
    cum_frame_ms = 0.0
    n_frames_proc = 0

    last_pos_cache = {}  # uid -> (pos, last_seen_frame)
    last_cam_cache = {}  # uid -> "cam1"|"cam2"
    last_mapped_cache = {}  # uid -> bool (solo útil en render)
    override_sticky = getattr(config, "STICKY_MISS_TOL_FRAMES", None)
    if override_sticky is not None:
        STICKY_MISS_TOL = int(max(0, override_sticky))
    else:
        STICKY_MISS_TOL = max(1, int(round(fps * float(getattr(config, "STABILITY_CONFIG").sticky_miss_tol_sec))))

    CAM2_ID_OFFSET = 100000
    live_window = "Live (S/ESC para detener y guardar)"
    stop_button_rect = (10, 10, 220, 55)  # x1,y1,x2,y2
    stop_requested = False

    def _on_mouse_live(event, x, y, flags, param):
        nonlocal stop_requested
        if event == cv2.EVENT_LBUTTONUP:
            x1, y1, x2, y2 = stop_button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                stop_requested = True

    global _ACTIVE_RUN
    _ACTIVE_RUN = dict(
        finalized=False,
        run_dir=str(run_dir),
        also_write_codes_data=bool(also_write_codes_data),
        live_window=str(live_window),
        video_out=video_out,
        video_out_local=video_out_local,
        points_writer=points_writer,
        fps=float(fps),
        total_frames=int(total_frames),
        video_cam1=str(video_cam1),
        video_cam2=str(video_cam2),
        pipeline_stage=str(pipeline_stage),
        offline_map_path=str(offline_map_path),
        output_video=str(TARGET_VIDEO_OUTPUT),
        output_video_local_ids=str(local_ids_output),
        output_video_points=str(points_video_path),
        metrics_rows=metrics_rows,
        track_rows_cam1=track_rows_cam1,
        track_rows_cam2=track_rows_cam2,
        ball_rows=ball_rows,
        stable_cam1=stable_cam1,
        stable_cam2=stable_cam2,
    )

    cv2.namedWindow(live_window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(live_window, _on_mouse_live)

    clip_enable = bool(getattr(config, "CLIP_ENABLE", False))
    clip_start = int(getattr(config, "CLIP_START_FRAME", 0))
    clip_end = int(getattr(config, "CLIP_END_FRAME", 0))
    for i in range(total_frames):
        if clip_enable:
            if i < clip_start:
                continue
            if clip_end > 0 and i > clip_end:
                break
        t0 = time.time()
        t_frame_start = time.time()
        chosen = None

        fr1_rgb = vr1[i].asnumpy()
        fr2_rgb = vr2[i].asnumpy()
        fr1_bgr = cv2.cvtColor(fr1_rgb, cv2.COLOR_RGB2BGR)
        fr2_bgr = cv2.cvtColor(fr2_rgb, cv2.COLOR_RGB2BGR)

        fr1 = fr1_bgr.copy()
        fr2 = fr2_bgr.copy()
        fr1_local = fr1_bgr.copy()
        fr2_local = fr2_bgr.copy()

        # Ball (selección simple + filtro de outliers)
        t_ball_start = time.time()
        ball_xy_pitch = [math.nan, math.nan]
        ball_bbox_cam = None

        def _best_ball(balls, vt, cam_label, shape):
            if balls is None or len(balls.xyxy) == 0:
                return None
            idx = int(np.argmax(balls.confidence)) if len(balls.confidence) > 0 else -1
            if idx < 0:
                return None
            bbox = balls.xyxy[idx].astype(np.float32)
            conf = float(balls.confidence[idx]) if len(balls.confidence) > idx else 0.0
            x1, y1, x2, y2 = bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))
            h, w = shape[:2]
            border = min(x1, y1, max(0.0, w - x2), max(0.0, h - y2))
            if conf < float(getattr(config, "BALL_MIN_CONF", 0.2)):
                return None
            if area < float(getattr(config, "BALL_MIN_AREA", 0.0)) or area > float(
                getattr(config, "BALL_MAX_AREA", 1e9)
            ):
                return None
            if border < float(getattr(config, "BALL_BORDER_MIN", 0.0)):
                return None
            anchor = balls.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[idx]
            pt = vt.transform_points(anchor.reshape(1, 2))[0].astype(np.float32)
            return dict(cam=cam_label, bbox=bbox, conf=conf, pos=pt, area=area, border=border)

        ball_conf_cam1 = math.nan
        ball_conf_cam2 = math.nan
        if render_offline:
            row_ball = ball_idx.get(i)
            if row_ball is not None:
                ball_xy_pitch = [float(row_ball.get("ball_x", math.nan)), float(row_ball.get("ball_y", math.nan))]
                if {"ball_bbox_x1", "ball_bbox_y1", "ball_bbox_x2", "ball_bbox_y2"}.issubset(row_ball.index):
                    bb = (
                        float(row_ball["ball_bbox_x1"]),
                        float(row_ball["ball_bbox_y1"]),
                        float(row_ball["ball_bbox_x2"]),
                        float(row_ball["ball_bbox_y2"]),
                    )
                    ball_bbox_cam = (row_ball.get("cam_source"), bb)
                else:
                    ball_bbox_cam = (row_ball.get("cam_source"), None)
                ball_conf_cam1 = float(row_ball.get("ball_conf_cam1", math.nan))
                ball_conf_cam2 = float(row_ball.get("ball_conf_cam2", math.nan))
        else:
            cand = []
            b1 = _best_ball(slicer_ball(fr1_rgb), vt1, "cam1", fr1_rgb.shape)
            b2 = _best_ball(slicer_ball(fr2_rgb), vt2, "cam2", fr2_rgb.shape)
            if b1 is not None:
                cand.append(b1)
                ball_conf_cam1 = b1["conf"]
            if b2 is not None:
                cand.append(b2)
                ball_conf_cam2 = b2["conf"]
            if cand:
                # escoger la de mayor confianza (si hay empate, la primera)
                cand.sort(key=lambda x: x["conf"], reverse=True)
                chosen = cand[0]
                # filtro de outliers por salto grande
                max_jump = float(getattr(config, "BALL_MAX_JUMP_PITCH", 0.0))
                if (
                    max_jump > 0
                    and last_ball_pos is not None
                    and last_ball_frame is not None
                    and (np.linalg.norm(chosen["pos"] - last_ball_pos) > max_jump * max(1, i - last_ball_frame))
                ):
                    chosen = None
                if chosen is not None:
                    ball_xy_pitch = [float(chosen["pos"][0]), float(chosen["pos"][1])]
                    ball_bbox_cam = (chosen["cam"], chosen["bbox"])
                    last_ball_pos = chosen["pos"]
                    last_ball_frame = i

        ball_x, ball_y = ball_xy_pitch
        ball_row = dict(
            frame=int(i),
            ball_x=float(ball_x),
            ball_y=float(ball_y),
            cam_source=(ball_bbox_cam[0] if ball_bbox_cam is not None else None),
            ball_conf=(float(chosen["conf"]) if "chosen" in locals() and chosen is not None else float("nan")),
            ball_conf_cam1=float(ball_conf_cam1),
            ball_conf_cam2=float(ball_conf_cam2),
        )
        if ball_bbox_cam is not None and ball_bbox_cam[1] is not None:
            bb = ball_bbox_cam[1]
            try:
                ball_row.update(
                    dict(
                        ball_bbox_x1=float(bb[0]),
                        ball_bbox_y1=float(bb[1]),
                        ball_bbox_x2=float(bb[2]),
                        ball_bbox_y2=float(bb[3]),
                    )
                )
            except Exception:
                pass
        ball_rows.append(ball_row)
        cum_ball_ms += (time.time() - t_ball_start) * 1000.0
        cum_frame_ms += (time.time() - t_frame_start) * 1000.0
        n_frames_proc += 1

        if render_offline:
            n_emb_none_cam1 = 0
            n_emb_none_cam2 = 0
            n_dets = 0
            observed1 = []
            observed2 = []
            rows1 = offline_tracks.get("cam1", {}).get(i)
            rows2 = offline_tracks.get("cam2", {}).get(i)
            if rows1 is not None:
                for _, r in rows1.iterrows():
                    bbox = (
                        float(r["bbox_x1"]) if "bbox_x1" in r else 0.0,
                        float(r["bbox_y1"]) if "bbox_y1" in r else 0.0,
                        float(r["bbox_x2"]) if "bbox_x2" in r else 0.0,
                        float(r["bbox_y2"]) if "bbox_y2" in r else 0.0,
                    )
                    pos = np.array([float(r["pos_x"]), float(r["pos_y"])], dtype=np.float32)
                    observed1.append(
                        dict(
                            ds_id=int(r.get("ds_id", r["stable_id"])),
                            stable_id=remap_sid("cam1", r["stable_id"]),
                            bbox=bbox,
                            emb=None,
                            has_emb=int(r.get("has_emb", 0)),
                            pos=pos,
                            conf=float(r.get("conf", 1.0)),
                        )
                    )
                    n_dets += 1
            if rows2 is not None:
                for _, r in rows2.iterrows():
                    bbox = (
                        float(r["bbox_x1"]) if "bbox_x1" in r else 0.0,
                        float(r["bbox_y1"]) if "bbox_y1" in r else 0.0,
                        float(r["bbox_x2"]) if "bbox_x2" in r else 0.0,
                        float(r["bbox_y2"]) if "bbox_y2" in r else 0.0,
                    )
                    pos = np.array([float(r["pos_x"]), float(r["pos_y"])], dtype=np.float32)
                    observed2.append(
                        dict(
                            ds_id=int(r.get("ds_id", r["stable_id"])),
                            stable_id=remap_sid("cam2", r["stable_id"]),
                            bbox=bbox,
                            emb=None,
                            has_emb=int(r.get("has_emb", 0)),
                            pos=pos,
                            conf=float(r.get("conf", 1.0)),
                        )
                    )
                    n_dets += 1
        else:
            # Players detections + tracking monocámara (BoT-SORT/ByteTrack via Ultralytics)
            player_cls = int(getattr(config, "PLAYER_CLASS_ID", 1))
            det_conf = float(getattr(config, "TRACKING_DET_CONF", 0.1))

            res1 = PLAYER_DETECTION_MODEL.predict(
                fr1_bgr,
                imgsz=DETECTION_CFG.imgsz,
                iou=DETECTION_CFG.iou,
                conf=det_conf,
                classes=[player_cls],
            )[0]
            res2 = PLAYER_DETECTION_MODEL.predict(
                fr2_bgr,
                imgsz=DETECTION_CFG.imgsz,
                iou=DETECTION_CFG.iou,
                conf=det_conf,
                classes=[player_cls],
            )[0]

            boxes1 = res1.boxes
            boxes2 = res2.boxes

            def _filter_boxes_boxes(boxes, w, h):
                if boxes is None or boxes.data is None or len(boxes) == 0:
                    return None
                data = boxes.data
                min_h = int(getattr(config, "TRACK_MIN_BBOX_H", 0))
                min_area = int(getattr(config, "TRACK_MIN_BBOX_AREA", 0))
                max_y_top = float(getattr(config, "TRACK_MAX_Y_TOP", 1.0))

                x1 = data[:, 0]
                y1 = data[:, 1]
                x2 = data[:, 2]
                y2 = data[:, 3]

                bh = (y2 - y1)
                bw = (x2 - x1)
                area = bw * bh

                keep = (bh >= min_h) & (area >= min_area) & (y1 <= max_y_top * h)
                if not bool(keep.any()):
                    return None
                return boxes.__class__(data[keep].cpu(), boxes.orig_shape)

            boxes1_f = _filter_boxes_boxes(boxes1, w1, h1)
            boxes2_f = _filter_boxes_boxes(boxes2, w2, h2)

            t1 = tracker_cam1.update(boxes1_f, fr1_bgr) if boxes1_f is not None else np.empty((0, 8), dtype=np.float32)
            t2 = tracker_cam2.update(boxes2_f, fr2_bgr) if boxes2_f is not None else np.empty((0, 8), dtype=np.float32)

            t1 = np.asarray(t1, dtype=np.float32)
            t2 = np.asarray(t2, dtype=np.float32)
            if t1.size == 0:
                t1 = t1.reshape(0, 8)
            if t2.size == 0:
                t2 = t2.reshape(0, 8)

            n_dets = (len(boxes1_f) if boxes1_f is not None else 0) + (len(boxes2_f) if boxes2_f is not None else 0)

            # Observed-only + StableID
            # alive_ds_ids controla cuándo un ds_id "muere" y se manda al bank (para reattach).
            alive_mode = str(getattr(config, "STABLEID_ALIVE_MODE", "tracker")).strip().lower()
            if alive_mode == "observed":
                alive1 = {int(tid) for _x1, _y1, _x2, _y2, tid, _score, cls, _idx in t1 if int(cls) == player_cls}
                alive2 = {int(tid) for _x1, _y1, _x2, _y2, tid, _score, cls, _idx in t2 if int(cls) == player_cls}
            else:
                if alive_mode != "tracker":
                    print(f"[WARN] STABLEID_ALIVE_MODE='{alive_mode}' desconocido, usando 'tracker'.")
                alive1 = {int(t.track_id) for t in list(getattr(tracker_cam1, "tracked_stracks", []))} | {
                    int(t.track_id) for t in list(getattr(tracker_cam1, "lost_stracks", []))
                }
                alive2 = {int(t.track_id) for t in list(getattr(tracker_cam2, "tracked_stracks", []))} | {
                    int(t.track_id) for t in list(getattr(tracker_cam2, "lost_stracks", []))
                }

            n_emb_none_cam1 = 0
            n_emb_none_cam2 = 0

            observed1 = []
            for x1, y1, x2, y2, tid, score, cls, _idx in t1:
                if int(cls) != player_cls:
                    continue
                bbox = (float(x1), float(y1), float(x2), float(y2))
                emb = extract_embedding(fr1_bgr, bbox)
                if emb is None:
                    n_emb_none_cam1 += 1
                emb_out = np.asarray(emb, dtype=np.float32) if emb is not None else None
                pos = bbox_xyxy_to_pitch_pos(vt1, bbox)
                observed1.append(
                    {
                        "ds_id": int(tid),
                        "bbox": bbox,
                        "emb": emb_out,
                        "has_emb": int(emb_out is not None),
                        "pos": pos,
                        "conf": float(score),
                    }
                )

            observed2 = []
            for x1, y1, x2, y2, tid, score, cls, _idx in t2:
                if int(cls) != player_cls:
                    continue
                bbox = (float(x1), float(y1), float(x2), float(y2))
                emb = extract_embedding(fr2_bgr, bbox)
                if emb is None:
                    n_emb_none_cam2 += 1
                emb_out = np.asarray(emb, dtype=np.float32) if emb is not None else None
                pos = bbox_xyxy_to_pitch_pos(vt2, bbox)
                observed2.append(
                    {
                        "ds_id": int(tid),
                        "bbox": bbox,
                        "emb": emb_out,
                        "has_emb": int(emb_out is not None),
                        "pos": pos,
                        "conf": float(score),
                    }
                )

            # === NO CONFUNDIR: desactiva StableID y usa ds_id directo ===
            if bool(getattr(config, "STABLEID_ENABLE", True)):
                observed1 = stable_cam1.update(i, alive1, observed1, frame_wh=(w1, h1))
                observed2 = stable_cam2.update(i, alive2, observed2, frame_wh=(w2, h2))
            else:
                # stable_id = ds_id (identidad del tracker puro)
                for o in observed1:
                    o["stable_id"] = int(o["ds_id"])
                for o in observed2:
                    o["stable_id"] = int(o["ds_id"])

        # --- Anti-teleport: recalcular o["pos"] usando stable_id (después del update) ---
        if getattr(config, "POS_SMOOTH_ENABLE", True) and bool(getattr(config, "STABLEID_ENABLE", True)):
            beta = float(getattr(config, "POS_EMA_BETA", 0.20))
            sp_pitch = float(getattr(config, "POS_SPIKE_PITCH_THR", 150.0))
            sp_img = float(getattr(config, "POS_SPIKE_IMG_THR", 15.0))
            ttl = int(getattr(config, "POS_STATE_TTL_FRAMES", 240))

            for o in observed1:
                sid = int(o["stable_id"])
                o["pos_raw"] = o["pos"]  # guarda el original
                o["pos"] = _smooth_project_pos(
                    _pos_state_cam1, sid, o["bbox"], vt1, i, beta=beta, spike_pitch=sp_pitch, spike_img=sp_img, ttl=ttl
                )

            for o in observed2:
                sid = int(o["stable_id"])
                o["pos_raw"] = o["pos"]
                o["pos"] = _smooth_project_pos(
                    _pos_state_cam2, sid, o["bbox"], vt2, i, beta=beta, spike_pitch=sp_pitch, spike_img=sp_img, ttl=ttl
                )

        # Dedup por gid (preferir cam1) solo si no estamos en modo raw
        if pipeline_stage == "render" and not getattr(config, "RAW_POS_MODE", False):
            gid_first = {}
            for o in list(observed1) + list(observed2):
                gid_o = o.get("gid")
                if gid_o is None:
                    continue
                gid_o = int(gid_o)
                if gid_o in gid_first:
                    continue
                gid_first[gid_o] = o.get("cam", "cam1")
            observed1 = [
                o for o in observed1 if o.get("gid") is None or gid_first.get(int(o["gid"]), "cam1") == "cam1"
            ]
            observed2 = [
                o for o in observed2 if o.get("gid") is None or gid_first.get(int(o["gid"]), "cam1") == "cam2"
            ]

        # Draw local IDs video + main video (IDs = uid)
        best_obs_by_id = {}  # draw_id -> {"pos":..., "conf":..., "cam":..., "mapped":..., "cam_idx":...}

        # gids compartidos (presentes en ambos mapas)
        common_gids = set(gid_map.get("cam1", {}).values()) & set(gid_map.get("cam2", {}).values()) if pipeline_stage == "render" else set()

        def _consider(draw_id, cam_label, pos, conf_val, mapped: bool = True):
            prev = best_obs_by_id.get(draw_id)
            cand = dict(
                pos=pos,
                conf=float(conf_val) if conf_val is not None else -1.0,
                cam=cam_label,
                mapped=bool(mapped),
                cam_idx=(1 if cam_label == "cam1" else 2),
            )
            # preferir cam1 cuando el gid existe en ambas cámaras (evita saltos por desalineación)
            if pipeline_stage == "render" and draw_id in common_gids:
                if prev is not None and prev.get("cam_idx") == 1 and cand["cam_idx"] == 2:
                    return
            if prev is None:
                best_obs_by_id[draw_id] = cand
                return
            if cand["conf"] > prev["conf"] + 1e-6:
                best_obs_by_id[draw_id] = cand
                return
            if abs(cand["conf"] - prev["conf"]) <= 1e-6 and cam_label == "cam1" and prev["cam"] != "cam1":
                best_obs_by_id[draw_id] = cand

        for o in observed1:
            sid = int(o["stable_id"])
            gid = None
            if pipeline_stage == "render":
                gid = gid_map.get("cam1", {}).get(sid)
                gid = int(gid) if gid is not None else None
            uid = int(gid) if gid is not None else sid
            bbox = o["bbox"]
            pos = o["pos"]
            conf = o.get("conf")
            if pipeline_stage == "render":
                if bool(getattr(config, "RENDER_MAPPED_ONLY", False)) and gid is None:
                    continue
                if not getattr(config, "RAW_POS_MODE", False):
                    if not (-200.0 <= pos[0] <= 4200.0 and -200.0 <= pos[1] <= 2200.0):
                        continue
            _consider(uid, "cam1", pos, conf, mapped=(gid is not None or pipeline_stage != "render"))

            track_rows_cam1.append(
                dict(
                    frame=int(i),
                    cam="cam1",
                    stable_id=int(sid),
                    ds_id=int(o.get("ds_id", -1)),
                    gid=(int(gid) if gid is not None else None),
                    has_emb=int(o.get("has_emb", 0)),
                    pos_x=float(pos[0]),
                    pos_y=float(pos[1]),
                    pos_raw_x=float(o.get("pos_raw", pos)[0]),
                    pos_raw_y=float(o.get("pos_raw", pos)[1]),
                    bbox_x1=float(bbox[0]),
                    bbox_y1=float(bbox[1]),
                    bbox_x2=float(bbox[2]),
                    bbox_y2=float(bbox[3]),
                    conf=(float(conf) if conf is not None else math.nan),
                    ball_x=float(ball_x),
                    ball_y=float(ball_y),
                )
            )

            draw_player_box(fr1_local, bbox, sid, (0, 255, 255), track_id=f"cam1:{sid}", global_id=None)
            track_id_txt = (
                f"cam1:sid={sid} UNMAPPED" if (pipeline_stage == "render" and gid is None) else f"cam1:sid={sid}"
            )
            if gid is not None:
                track_id_txt = f"cam1:sid={sid} gid={gid}"
            draw_player_box(fr1, bbox, uid, (0, 255, 255), track_id=track_id_txt, global_id=None)

        for o in observed2:
            sid = int(o["stable_id"])
            gid = None
            if pipeline_stage == "render":
                gid = gid_map.get("cam2", {}).get(sid)
                gid = int(gid) if gid is not None else None
            uid = int(gid) if gid is not None else (CAM2_ID_OFFSET + sid)
            bbox = o["bbox"]
            pos = o["pos"]
            conf = o.get("conf")
            if pipeline_stage == "render":
                if bool(getattr(config, "RENDER_MAPPED_ONLY", False)) and gid is None:
                    continue
                if not getattr(config, "RAW_POS_MODE", False):
                    if not (-200.0 <= pos[0] <= 4200.0 and -200.0 <= pos[1] <= 2200.0):
                        continue
            _consider(uid, "cam2", pos, conf, mapped=(gid is not None or pipeline_stage != "render"))

            track_rows_cam2.append(
                dict(
                    frame=int(i),
                    cam="cam2",
                    stable_id=int(sid),
                    ds_id=int(o.get("ds_id", -1)),
                    gid=(int(gid) if gid is not None else None),
                    has_emb=int(o.get("has_emb", 0)),
                    pos_x=float(pos[0]),
                    pos_y=float(pos[1]),
                    pos_raw_x=float(o.get("pos_raw", pos)[0]),
                    pos_raw_y=float(o.get("pos_raw", pos)[1]),
                    bbox_x1=float(bbox[0]),
                    bbox_y1=float(bbox[1]),
                    bbox_x2=float(bbox[2]),
                    bbox_y2=float(bbox[3]),
                    conf=(float(conf) if conf is not None else math.nan),
                    ball_x=float(ball_x),
                    ball_y=float(ball_y),
                )
            )

            draw_player_box(fr2_local, bbox, sid, (255, 255, 0), track_id=f"cam2:{sid}", global_id=None)
            track_id_txt = (
                f"cam2:sid={sid} UNMAPPED" if (pipeline_stage == "render" and gid is None) else f"cam2:sid={sid}"
            )
            if gid is not None:
                track_id_txt = f"cam2:sid={sid} gid={gid}"
            draw_player_box(fr2, bbox, uid, (255, 255, 0), track_id=track_id_txt, global_id=None)

        drawn_gids = set()
        players_positions = {}
        for pid, v in best_obs_by_id.items():
            gid = int(pid)
            if gid in drawn_gids:
                continue
            drawn_gids.add(gid)
            players_positions[pid] = v["pos"]
        players_cam = {pid: v["cam"] for pid, v in best_obs_by_id.items()}
        players_mapped = {pid: bool(v.get("mapped", True)) for pid, v in best_obs_by_id.items()}

        if ball_bbox_cam is not None:
            cam_b, bb = ball_bbox_cam
            if bb is not None:
                if cam_b == "cam1":
                    draw_box(fr1, bb, "Ball", color=(0, 255, 255))
                elif cam_b == "cam2":
                    draw_box(fr2, bb, "Ball", color=(0, 255, 255))

        # Radar/Voronoi
        for uid, pos in players_positions.items():
            last_pos_cache[uid] = (pos.astype(np.float32), i)
        for uid, cam_label in players_cam.items():
            last_cam_cache[uid] = cam_label
        for uid, mapped in players_mapped.items():
            last_mapped_cache[uid] = bool(mapped)

        pts_pitch = {}
        pts_cam = {}
        pts_mapped = {}
        hold_events = 0
        for uid in set(list(players_positions.keys()) + list(last_pos_cache.keys())):
            if uid in players_positions:
                pts_pitch[uid] = players_positions[uid]
                pts_cam[uid] = players_cam.get(uid, last_cam_cache.get(uid, "cam1"))
                pts_mapped[uid] = players_mapped.get(uid, last_mapped_cache.get(uid, True))
            else:
                if STICKY_MISS_TOL > 0:
                    pos_cache, last_seen = last_pos_cache.get(uid, (None, None))
                    if pos_cache is not None and last_seen is not None and (i - last_seen) <= STICKY_MISS_TOL:
                        pts_pitch[uid] = pos_cache
                        pts_cam[uid] = last_cam_cache.get(uid, "cam1")
                        pts_mapped[uid] = last_mapped_cache.get(uid, True)
                        hold_events += 1

        elements = {"points": [], "paths": []}
        for uid, pxy in pts_pitch.items():
            cam_label = pts_cam.get(uid, "cam1")
            mapped = bool(pts_mapped.get(uid, True))
            if pipeline_stage == "render" and not mapped:
                color = (128, 128, 128)  # unmapped -> gris
            else:
                color = (0, 0, 255) if cam_label == "cam1" else (255, 0, 0)  # BGR: rojo cam1, azul cam2
            elements["points"].append((float(pxy[0]), float(pxy[1]), color))
        if not (math.isnan(ball_x) or math.isnan(ball_y)):
            elements["points"].append((float(ball_x), float(ball_y), (0, 255, 255)))

        radar_image = pitch_renderer.draw(elements)
        radar_resized = cv2.resize(radar_image, (radar_width, radar_height))

        rx, ry = radar_position
        fr1[ry : ry + radar_height, rx : rx + radar_width] = radar_resized
        points_writer.write(radar_resized)

        # side-by-side composition
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[: h1, :w1] = fr1
        canvas[: h2, w1 : w1 + w2] = fr2
        canvas_local = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas_local[: h1, :w1] = fr1_local
        canvas_local[: h2, w1 : w1 + w2] = fr2_local

        # UI live: botón STOP + overlay simple
        x1, y1, x2, y2 = stop_button_rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.putText(canvas, "STOP (S/ESC)", (x1 + 10, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Watermark de stage/fuente/anchor/smoothing/clip
        src_txt = "stitched" if render_offline else "tracks"
        smooth_txt = str(bool(getattr(config, "POS_SMOOTH_ENABLE", False)))
        clip_txt = (
            f"{clip_start}-{clip_end}" if clip_enable and clip_end > 0 else (f"from {clip_start}" if clip_enable else "off")
        )
        wm = f"stage={pipeline_stage} source={src_txt} anchor=BC smooth={smooth_txt} clip={clip_txt}"
        cv2.putText(canvas, wm, (10, out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        video_out.write(canvas)
        video_out_local.write(canvas_local)

        fps_frame = 1.0 / max(1e-6, (time.time() - t0))
        metrics_rows.append(
            dict(
                frame=i,
                fps_frame=round(fps_frame, 3),
                n_dets=int(n_dets),
                n_tracks=int(len(observed1) + len(observed2)),
                n_emb_none_cam1=int(n_emb_none_cam1),
                n_emb_none_cam2=int(n_emb_none_cam2),
                n_drawn=int(len(players_positions)),
                n_voronoi_points=int(len(pts_pitch)),
                n_hold_points=int(hold_events),
            )
        )
        cum_frame_ms += (time.time() - t_frame_start) * 1000.0
        n_frames_proc += 1

        if i % 30 == 0:
            pct = (i + 1) * 100.0 / total_frames
            print(f"[LOOP] Frame {i+1}/{total_frames} ({pct:.1f}%) | fps_frame={fps_frame:.2f} dets={n_dets}")

        # Vista en vivo + stop
        cv2.imshow(live_window, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("s"), ord("S")):
            stop_requested = True
        if stop_requested:
            break

    video_out.release()
    video_out_local.release()
    points_writer.release()
    try:
        cv2.destroyWindow(live_window)
    except Exception:
        pass

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(run_dir, "METRICS.csv"), index=False)
    if also_write_codes_data:
        metrics_df.to_csv(_codes_data_path("METRICS.csv"), index=False)

    df_cam1 = pd.DataFrame(track_rows_cam1)
    df_cam2 = pd.DataFrame(track_rows_cam2)
    df_ball = pd.DataFrame(ball_rows)

    df_cam1.to_csv(os.path.join(run_dir, "c1_tracks.csv"), index=False)
    df_cam2.to_csv(os.path.join(run_dir, "c2_tracks.csv"), index=False)
    df_ball.to_csv(os.path.join(run_dir, "ball.csv"), index=False)

    if n_frames_proc > 0:
        avg_ball_ms = cum_ball_ms / n_frames_proc
        avg_frame_ms = cum_frame_ms / n_frames_proc
        print(
            f"[TIME] frames={n_frames_proc} avg_ball_ms={avg_ball_ms:.2f} avg_frame_ms={avg_frame_ms:.2f} | approx_total={_fmt_dur(cum_frame_ms/1000.0)}"
        )

    # Excel combinado (humano): útil para inspección rápida
    df_all = pd.concat([df_cam1, df_cam2], ignore_index=True)
    excel_path = os.path.join(run_dir, "Posiciones-jugadores-balon-multicam.xlsx")
    df_all.to_excel(excel_path, index=False)
    if also_write_codes_data:
        df_all.to_excel(_codes_data_path("Posiciones-jugadores-balon-multicam.xlsx"), index=False)

    meta = dict(
        fps=float(fps),
        total_frames=int(total_frames),
        video_cam1=str(video_cam1),
        video_cam2=str(video_cam2),
        pipeline_stage=str(pipeline_stage),
        run_dir=str(run_dir),
        offline_map_path=str(offline_map_path),
        output_video=str(TARGET_VIDEO_OUTPUT),
        output_video_local_ids=str(local_ids_output),
        output_video_points=str(points_video_path),
    )
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if stable_cam1 is not None:
        _dump_tracklet_artifacts(run_dir, "c1", stable_cam1)
    if stable_cam2 is not None:
        _dump_tracklet_artifacts(run_dir, "c2", stable_cam2)

    try:
        row = _compute_dashboard_row(
            run_dir=str(run_dir),
            pipeline_stage=str(pipeline_stage),
            fps=float(fps),
            total_frames=int(total_frames),
            frames_processed=int(len(metrics_rows)),
            df_cam1=df_cam1,
            df_cam2=df_cam2,
            df_ball=df_ball,
        )
        _write_dashboard_csv(run_dir=str(run_dir), row=row, also_write_codes_data=also_write_codes_data)
    except Exception as dump_exc:
        print(f"[WARN] No se pudo escribir DASHBOARD.csv: {dump_exc!r}")

    print("Listo!")
    print("-", TARGET_VIDEO_OUTPUT)
    print("-", local_ids_output)
    print("-", points_video_path)
    _ACTIVE_RUN = None


if __name__ == "__main__":
    try:
        process_dual_camera(VIDEO_PATH_CAM1, VIDEO_PATH_CAM2, view_transformer1, view_transformer2)
    except BaseException as exc:
        _finalize_active_run(aborted=True, error_repr=repr(exc))
        raise

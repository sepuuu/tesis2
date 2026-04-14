"""
Genera videos de diagnóstico por etapa (export, stitched, render con mapping) y métricas por frame.
Se ejecuta sin argumentos: usa config.RUN_DIR y config.DEBUG_*.
Salidas en RUN_DIR/debug_viz/.
"""
import os
import json
import time
from typing import Dict, Tuple, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

import config
from configs.drawing import PitchRenderer
from configs.soccer import SoccerPitchConfiguration

# Default team assignment (final GIDs); override via config if present
TEAM_WHITE_GIDS = set(getattr(config, "TEAM_WHITE_GIDS", {1, 2, 3, 4, 9, 10}))
TEAM_BLACK_GIDS = set(getattr(config, "TEAM_BLACK_GIDS", {5, 6, 7, 8, 11, 12}))
PASS_MAP_RIGHT_GOALKEEPER_GID = int(getattr(config, "PASS_MAP_RIGHT_GOALKEEPER_GID", 1))
PASS_MAP_LEFT_GOALKEEPER_GID = int(getattr(config, "PASS_MAP_LEFT_GOALKEEPER_GID", 8))

def _load_meta_info(run_dir: str) -> dict:
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_meta(run_dir: str, meta: Optional[dict] = None) -> float:
    if meta is None:
        meta = _load_meta_info(run_dir)
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return 24.0
    try:
        return float(meta.get("fps", 24.0))
    except Exception:
        return 24.0


def _load_tracks(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in ("frame", "stable_id"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def _load_ball(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["frame", "ball_x", "ball_y", "cam_source"])
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df["frame"] = df["frame"].astype(int)
    return df


def _clean_ball(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["frame", "ball_x", "ball_y", "cam_source", "is_clean"])
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("frame")
    min_conf = float(getattr(config, "BALL_MIN_CONF", 0.0))
    min_area = float(getattr(config, "BALL_MIN_AREA", 0.0))
    max_area = float(getattr(config, "BALL_MAX_AREA", 1e9))
    max_jump = float(getattr(config, "BALL_MAX_JUMP_PITCH", 0.0))
    max_jump_strict = float(getattr(config, "BALL_MAX_JUMP_PITCH_STRICT", max_jump if max_jump > 0 else 0.0))
    max_jump_long = float(getattr(config, "BALL_MAX_JUMP_LONG", max_jump if max_jump > 0 else 1e9))
    neighbor_thr = float(getattr(config, "BALL_NEIGHBOR_JUMP_THR", max_jump_strict if max_jump_strict > 0 else 300.0))

    last_pos = None
    last_frame = None
    is_clean = []
    # precompute neighbor jumps for two-sided outlier check
    df["_idx"] = range(len(df))
    df_valid = df.dropna(subset=["ball_x", "ball_y"])
    df_valid = df_valid.sort_values("frame")
    # map frame -> idx for nearest prev/next
    frames = df_valid["frame"].to_numpy()
    xs = df_valid["ball_x"].to_numpy(dtype=float)
    ys = df_valid["ball_y"].to_numpy(dtype=float)
    neighbor_flag = {idx: False for idx in df["_idx"]}
    if len(frames) >= 3:
        # compute prev/next distances
        for j in range(1, len(frames) - 1):
            idx_center = int(df_valid.iloc[j]["_idx"])
            gap_prev = int(frames[j] - frames[j - 1])
            gap_next = int(frames[j + 1] - frames[j])
            if gap_prev <= 0 or gap_next <= 0:
                continue
            dx_prev = xs[j] - xs[j - 1]
            dy_prev = ys[j] - ys[j - 1]
            dist_prev = float(np.hypot(dx_prev, dy_prev))
            dx_next = xs[j + 1] - xs[j]
            dy_next = ys[j + 1] - ys[j]
            dist_next = float(np.hypot(dx_next, dy_next))
            rate_prev = dist_prev / max(1, gap_prev)
            rate_next = dist_next / max(1, gap_next)
            if rate_prev > neighbor_thr and rate_next > neighbor_thr:
                neighbor_flag[idx_center] = True
    for _, r in df.iterrows():
        bx = r.get("ball_x")
        by = r.get("ball_y")
        conf = r.get("ball_conf")
        cam_src = r.get("cam_source")
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = float("nan")
        ok = pd.notna(bx) and pd.notna(by)
        if not ok:
            is_clean.append(False)
            continue
        ok = ok and isinstance(cam_src, str) and len(str(cam_src).strip()) > 0
        if pd.isna(conf_val) or conf_val < min_conf:
            ok = False
        # area check if bbox present
        if ok and {"ball_bbox_x1", "ball_bbox_y1", "ball_bbox_x2", "ball_bbox_y2"}.issubset(r.index):
            try:
                area = max(
                    0.0,
                    (float(r["ball_bbox_x2"]) - float(r["ball_bbox_x1"]))
                    * (float(r["ball_bbox_y2"]) - float(r["ball_bbox_y1"])),
                )
                if area < min_area or area > max_area:
                    ok = False
            except Exception:
                pass
        if ok and max_jump > 0 and last_pos is not None and last_frame is not None:
            try:
                pos = np.array([float(bx), float(by)], dtype=np.float32)
                dist = float(np.linalg.norm(pos - last_pos))
                frame_gap = max(1, int(r.get("frame", last_frame + 1) - last_frame))
                limit = max_jump * frame_gap
                if frame_gap <= 1 and max_jump_strict > 0:
                    limit = min(limit, max_jump_strict)
                if max_jump_long > 0:
                    limit = min(limit, max_jump_long)
                if dist > limit:
                    ok = False
            except Exception:
                pass
        # two-sided jump check (prev and next)
        if ok and neighbor_flag.get(int(r["_idx"]), False):
            ok = False
        if ok:
            try:
                last_pos = np.array([float(bx), float(by)], dtype=np.float32)
                last_frame = int(r.get("frame", 0))
            except Exception:
                pass
        is_clean.append(bool(ok))
    df["is_clean"] = is_clean
    df = df.drop(columns=["_idx"], errors="ignore")
    return df


def _fmt_dur(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_map(run_dir: str) -> Dict[str, Dict[int, int]]:
    map_path = getattr(config, "OFFLINE_MAP_PATH", os.path.join(run_dir, "crosscam_map.json"))
    if not os.path.isabs(map_path):
        map_path = os.path.join(run_dir, map_path)
    if not os.path.exists(map_path):
        return {"cam1": {}, "cam2": {}}
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "cam1": {int(k): int(v) for k, v in data.get("cam1", {}).items()},
        "cam2": {int(k): int(v) for k, v in data.get("cam2", {}).items()},
    }


def _load_manual_override_map(run_dir: str) -> Tuple[Dict[str, Dict[int, int]], Dict[str, List[dict]]]:
    path = os.path.join(run_dir, "manual_overrides.json")
    out_map = {"cam1": {}, "cam2": {}}
    segments: Dict[str, List[dict]] = {"cam1": [], "cam2": []}
    if not os.path.exists(path):
        return out_map, segments
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        players = data.get("players")
        if not isinstance(players, dict):
            return out_map, segments
        for gid_key, info in players.items():
            try:
                gid = int(gid_key)
            except Exception:
                continue
            if not isinstance(info, dict):
                continue
            for cam_key in ("cam1", "cam2"):
                ids = info.get(cam_key) or info.get(f"{cam_key}_ids") or []
                for sid in ids:
                    try:
                        out_map[cam_key][int(sid)] = gid
                    except Exception:
                        continue
        # Casos especiales con rango de frames (opcional)
        segs = data.get("special_segments") or data.get("segments")
        if isinstance(segs, dict):
            for cam_key in ("cam1", "cam2"):
                lst = segs.get(cam_key, [])
                if not isinstance(lst, list):
                    continue
                for item in lst:
                    if not isinstance(item, dict):
                        continue
                    try:
                        sid = int(item.get("id"))
                        gid = int(item.get("gid"))
                    except Exception:
                        continue
                    try:
                        start_i = int(item.get("start", 0))
                    except Exception:
                        start_i = 0
                    end_val = item.get("end", None)
                    if end_val is None:
                        end_i = None
                    else:
                        try:
                            end_i = int(end_val)
                        except Exception:
                            end_i = None
                    segments[cam_key].append(dict(id=sid, gid=gid, start=start_i, end=end_i))
        return out_map, segments
    except Exception:
        return out_map, segments


def _resolve_gid(
    cam_label: str, sid: int, frame: int, gid_map: Dict[str, Dict[int, int]], segs: Dict[str, List[dict]]
) -> Optional[int]:
    # Prioriza segmentos (rangos de frames) y luego mapping fijo
    for seg in segs.get(cam_label, []):
        if seg.get("id") != sid:
            continue
        start = seg.get("start", 0)
        end = seg.get("end", None)
        if frame >= start and (end is None or frame <= end):
            return int(seg.get("gid"))
    return gid_map.get(cam_label, {}).get(sid)


def _resolve_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    alt = os.path.join(base_dir, path)
    if os.path.exists(alt):
        return alt
    return path


def _hash_color(gid: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(gid)
    c = rng.integers(60, 255, size=3, dtype=np.int64)
    return int(c[2]), int(c[1]), int(c[0])  # BGR


def _get_bounds(df_list: List[pd.DataFrame]) -> Tuple[float, float, float, float]:
    # Si hay bounds fijos en config, úsalos
    fixed = getattr(config, "DEBUG_VIZ_BOUNDS", None)
    if fixed and len(fixed) == 4:
        return tuple(map(float, fixed))
    if getattr(config, "DEBUG_VIZ_USE_AUTO_BOUNDS", False):
        xs = []
        ys = []
        for df in df_list:
            if "pos_x" in df.columns and "pos_y" in df.columns:
                xs.append(df["pos_x"])
                ys.append(df["pos_y"])
        if xs and ys:
            x_all = pd.concat(xs)
            y_all = pd.concat(ys)
            xmin, xmax = float(x_all.min()), float(x_all.max())
            ymin, ymax = float(y_all.min()), float(y_all.max())
            pad = float(getattr(config, "DEBUG_VIZ_PAD_FRAC", 0.0))
            dx = (xmax - xmin) * pad
            dy = (ymax - ymin) * pad
            return xmin - dx, xmax + dx, ymin - dy, ymax + dy
    # fallback a cancha estándar
    return 0.0, 4000.0, 0.0, 2000.0


def _setup_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)


def _frame_window(fps: float, max_frame: int) -> Tuple[int, int]:
    approx_sec = config.DEBUG_APPROX_SECOND
    window = int(getattr(config, "DEBUG_FRAME_WINDOW", 0))
    if approx_sec is None or window <= 0:
        return 0, max_frame
    center = int(float(approx_sec) * fps)
    start = max(0, center - window)
    end = min(max_frame, center + window)
    if getattr(config, "DEBUG_RUN_WINDOW_ONLY", False):
        return start, end
    return 0, max_frame


def _build_cam_players(df: pd.DataFrame, cam_label: str, gid_map: Dict[str, Dict[int, int]], segs: Dict[str, List[dict]]) -> Dict[int, List[dict]]:
    """Precalcula, por frame, los jugadores con su GID final y centro de bbox en píxeles."""
    required = {"frame", "stable_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    if df.empty or not required.issubset(df.columns):
        return {}
    out: Dict[int, List[dict]] = {}
    for frame, rows in df.groupby("frame"):
        lst = []
        for _, r in rows.iterrows():
            gid_val = None
            if "gid" in r and pd.notna(r["gid"]):
                try:
                    gid_val = int(r["gid"])
                except Exception:
                    gid_val = None
            if gid_val is None:
                gid_val = _resolve_gid(cam_label, int(r["stable_id"]), int(frame), gid_map, segs)
            if gid_val is None:
                continue
            try:
                x1 = float(r["bbox_x1"])
                y1 = float(r["bbox_y1"])
                x2 = float(r["bbox_x2"])
                y2 = float(r["bbox_y2"])
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                size = max(abs(x2 - x1), abs(y2 - y1))
            except Exception:
                continue
            lst.append(dict(gid=gid_val, cx=cx, cy=cy, size=size))
        if lst:
            out[int(frame)] = lst
    return out


def _build_gid_positions(df: pd.DataFrame, cam_label: str, gid_map: Dict[str, Dict[int, int]], segs: Dict[str, List[dict]]) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Por frame, devuelve posiciones en cancha por GID (pos_x/pos_y) usando el mapping final."""
    required = {"frame", "stable_id", "pos_x", "pos_y"}
    if df.empty or not required.issubset(df.columns):
        return {}
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for frame, rows in df.groupby("frame"):
        frame_map: Dict[int, Tuple[float, float]] = {}
        for _, r in rows.iterrows():
            gid_val = None
            if "gid" in r and pd.notna(r["gid"]):
                try:
                    gid_val = int(r["gid"])
                except Exception:
                    gid_val = None
            if gid_val is None:
                gid_val = _resolve_gid(cam_label, int(r["stable_id"]), int(frame), gid_map, segs)
            if gid_val is None:
                continue
            try:
                px = float(r["pos_x"])
                py = float(r["pos_y"])
            except Exception:
                continue
            frame_map[gid_val] = (px, py)
        if frame_map:
            out[int(frame)] = frame_map
    return out


def _merge_gid_positions(pos1: Dict[int, Dict[int, Tuple[float, float]]], pos2: Dict[int, Dict[int, Tuple[float, float]]]) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Combina mapas de posiciones por frame sin sobreescribir si ya existe."""
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for src in (pos1, pos2):
        for frame, gmap in src.items():
            dst = out.setdefault(frame, {})
            for gid, xy in gmap.items():
                dst.setdefault(gid, xy)
    return out


def _radar_color_for_gid(
    gid: int,
    *,
    team_mode: bool = False,
    teams: Optional[Dict[str, set]] = None,
) -> Tuple[int, int, int]:
    teams = teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    if team_mode:
        if gid in teams.get("white", set()):
            return (0, 255, 0)
        if gid in teams.get("black", set()):
            return (0, 165, 255)
    return _hash_color(gid)


def _render_pitch_radar(
    pitch_renderer: PitchRenderer,
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    frame_idx: int,
    *,
    ball_row: Optional[pd.Series] = None,
    team_mode: bool = False,
    teams: Optional[Dict[str, set]] = None,
) -> np.ndarray:
    elements = {"points": [], "paths": []}
    for gid, (x, y) in sorted(gid_pos_map.get(frame_idx, {}).items()):
        elements["points"].append((float(x), float(y), _radar_color_for_gid(gid, team_mode=team_mode, teams=teams)))
    if ball_row is not None:
        bx = ball_row.get("ball_x")
        by = ball_row.get("ball_y")
        if pd.notna(bx) and pd.notna(by):
            elements["points"].append((float(bx), float(by), (0, 255, 255)))
    return pitch_renderer.draw(elements)


def _blit_center_bottom(
    canvas: np.ndarray,
    overlay: np.ndarray,
    *,
    margin_bottom: int = 20,
    border_pad: int = 8,
) -> np.ndarray:
    out = canvas.copy()
    oh, ow = overlay.shape[:2]
    ch, cw = out.shape[:2]
    x = max(0, (cw - ow) // 2)
    y = max(0, ch - oh - margin_bottom)
    x2 = min(cw, x + ow)
    y2 = min(ch, y + oh)
    ow_eff = max(0, x2 - x)
    oh_eff = max(0, y2 - y)
    if ow_eff <= 0 or oh_eff <= 0:
        return out

    bg_x1 = max(0, x - border_pad)
    bg_y1 = max(0, y - border_pad)
    bg_x2 = min(cw, x2 + border_pad)
    bg_y2 = min(ch, y2 + border_pad)
    shaded = out.copy()
    cv2.rectangle(shaded, (bg_x1, bg_y1), (bg_x2, bg_y2), (18, 18, 18), -1)
    out = cv2.addWeighted(shaded, 0.35, out, 0.65, 0)
    out[y:y2, x:x2] = overlay[:oh_eff, :ow_eff]
    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (235, 235, 235), 1, lineType=cv2.LINE_AA)
    return out


def _make_pitch_canvas(
    *,
    width: int,
    height: int,
    padding: int = 30,
) -> Tuple[PitchRenderer, np.ndarray, int, int]:
    pitch_cfg = SoccerPitchConfiguration()
    usable_w = max(1, width - 2 * padding)
    usable_h = max(1, height - 2 * padding)
    scale_x = usable_w / float(pitch_cfg.length)
    scale_y = usable_h / float(pitch_cfg.width)
    scale = max(1e-6, min(scale_x, scale_y))
    renderer = PitchRenderer(config=pitch_cfg, scale=scale, padding=padding)
    pitch = renderer.base_pitch.copy()
    ph, pw = pitch.shape[:2]
    canvas = np.full((height, width, 3), renderer.background_color, dtype=np.uint8)
    off_x = max(0, (width - pw) // 2)
    off_y = max(0, (height - ph) // 2)
    canvas[off_y : off_y + ph, off_x : off_x + pw] = pitch
    return renderer, canvas, off_x, off_y


def _pitch_point_to_canvas(
    renderer: PitchRenderer,
    point: Tuple[float, float],
    *,
    off_x: int = 0,
    off_y: int = 0,
) -> Tuple[int, int]:
    px, py = renderer._scale_point([float(point[0]), float(point[1])])
    return int(px + off_x), int(py + off_y)


def _selected_debug_videos() -> set[str]:
    raw = getattr(config, "DEBUG_VIZ_VIDEO_TARGETS", ["04", "09", "10"])
    if isinstance(raw, (str, int)):
        raw = [raw]
    out = set()
    for item in raw:
        try:
            out.add(f"{int(str(item)):02d}")
        except Exception:
            continue
    return out or {"04", "09", "10"}


def _find_gid_pos(pos_map: Dict[int, Dict[int, Tuple[float, float]]], gid: int, frame: int, tol: int = 3) -> Optional[Tuple[float, float]]:
    """Busca la posición de un GID en frame, permitiendo tolerancia de ±tol frames."""
    if frame in pos_map and gid in pos_map[frame]:
        return pos_map[frame][gid]
    for d in range(1, tol + 1):
        for ff in (frame - d, frame + d):
            if ff in pos_map and gid in pos_map[ff]:
                return pos_map[ff][gid]
    return None


def _mean_gid_positions(gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]]) -> Dict[int, Tuple[float, float]]:
    """Promedia las posiciones por GID a partir del mapa frame->gid->(x,y)."""
    accum = {}
    count = {}
    for frame_map in gid_pos_map.values():
        for gid, (x, y) in frame_map.items():
            accum[gid] = accum.get(gid, (0.0, 0.0))
            cx, cy = accum[gid]
            accum[gid] = (cx + float(x), cy + float(y))
            count[gid] = count.get(gid, 0) + 1
    return {gid: (v[0] / count[gid], v[1] / count[gid]) for gid, v in accum.items() if count.get(gid, 0) > 0}


def _team_of_gid(gid: Optional[int], teams: Optional[Dict[str, set]] = None) -> str:
    teams = teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    if gid is None:
        return "unknown"
    if gid in teams.get("white", set()):
        return "white"
    if gid in teams.get("black", set()):
        return "black"
    return "unknown"


def _pass_map_team_sides(teams: Optional[Dict[str, set]] = None) -> Tuple[str, str]:
    teams = teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    right_team = _team_of_gid(PASS_MAP_RIGHT_GOALKEEPER_GID, teams)
    left_team = _team_of_gid(PASS_MAP_LEFT_GOALKEEPER_GID, teams)
    if right_team == "unknown":
        right_team = "white"
    if left_team == "unknown" or left_team == right_team:
        left_team = "black" if right_team == "white" else "white"
    return left_team, right_team


def _infer_pass_map_team_sets(
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    *,
    active_gids: Optional[set[int]] = None,
    fallback_teams: Optional[Dict[str, set]] = None,
) -> Dict[str, set]:
    fallback_teams = fallback_teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    mean_pos = _mean_gid_positions(gid_pos_map)
    if active_gids is None:
        active_gids = set(mean_pos.keys())
    else:
        active_gids = {int(gid) for gid in active_gids}
    if not active_gids:
        return {
            "white": set(fallback_teams.get("white", set())),
            "black": set(fallback_teams.get("black", set())),
        }

    pitch_cfg = SoccerPitchConfiguration()
    left_anchor = mean_pos.get(PASS_MAP_LEFT_GOALKEEPER_GID, (0.11 * pitch_cfg.length, 0.50 * pitch_cfg.width))
    right_anchor = mean_pos.get(PASS_MAP_RIGHT_GOALKEEPER_GID, (0.89 * pitch_cfg.length, 0.50 * pitch_cfg.width))
    inferred = {"white": set(), "black": set()}

    if PASS_MAP_LEFT_GOALKEEPER_GID in active_gids:
        inferred["black"].add(PASS_MAP_LEFT_GOALKEEPER_GID)
    if PASS_MAP_RIGHT_GOALKEEPER_GID in active_gids:
        inferred["white"].add(PASS_MAP_RIGHT_GOALKEEPER_GID)

    for gid in sorted(active_gids):
        if gid in (PASS_MAP_LEFT_GOALKEEPER_GID, PASS_MAP_RIGHT_GOALKEEPER_GID):
            continue
        if gid not in mean_pos:
            if gid in fallback_teams.get("black", set()):
                inferred["black"].add(gid)
            else:
                inferred["white"].add(gid)
            continue
        x = float(mean_pos[gid][0])
        if abs(x - float(left_anchor[0])) <= abs(x - float(right_anchor[0])):
            inferred["black"].add(gid)
        else:
            inferred["white"].add(gid)

    if not inferred["white"] or not inferred["black"]:
        return {
            "white": {int(gid) for gid in fallback_teams.get("white", set()) if gid in active_gids},
            "black": {int(gid) for gid in fallback_teams.get("black", set()) if gid in active_gids},
        }
    return inferred


def _pass_map_formation_slots(pitch_cfg: SoccerPitchConfiguration, side: str) -> Dict[str, Tuple[float, float]]:
    length = float(pitch_cfg.length)
    width = float(pitch_cfg.width)
    if side == "left":
        x_gk = 0.11 * length
        x_def = 0.22 * length
        x_mid = 0.31 * length
        x_fwd = 0.41 * length
    else:
        x_gk = 0.89 * length
        x_def = 0.78 * length
        x_mid = 0.69 * length
        x_fwd = 0.59 * length
    return {
        "gk": (x_gk, 0.50 * width),
        "def_top": (x_def, 0.34 * width),
        "def_bottom": (x_def, 0.66 * width),
        "mid": (x_mid, 0.50 * width),
        "fwd_top": (x_fwd, 0.38 * width),
        "fwd_bottom": (x_fwd, 0.62 * width),
    }


def _build_team_formation_nodes(
    gids: List[int],
    mean_pos: Dict[int, Tuple[float, float]],
    *,
    side: str,
    goalie_gid: Optional[int],
    pitch_cfg: SoccerPitchConfiguration,
) -> Dict[int, Tuple[float, float]]:
    slots = _pass_map_formation_slots(pitch_cfg, side)
    available = [int(gid) for gid in gids]
    if not available:
        return {}

    def _xy(gid: int) -> Tuple[float, float]:
        return mean_pos.get(gid, slots["mid"])

    gk = goalie_gid if goalie_gid in available else None
    if gk is None:
        if side == "right":
            gk = max(available, key=lambda gid: _xy(gid)[0])
        else:
            gk = min(available, key=lambda gid: _xy(gid)[0])

    remaining = [gid for gid in available if gid != gk]
    if side == "right":
        remaining = sorted(remaining, key=lambda gid: (-_xy(gid)[0], _xy(gid)[1], gid))
    else:
        remaining = sorted(remaining, key=lambda gid: (_xy(gid)[0], _xy(gid)[1], gid))

    defenders = sorted(remaining[:2], key=lambda gid: (_xy(gid)[1], gid))
    mids = remaining[2:3]
    forwards = sorted(remaining[3:5], key=lambda gid: (_xy(gid)[1], gid))
    extras = remaining[5:]

    out = {int(gk): slots["gk"]} if gk is not None else {}
    slot_order = [
        ("def_top", defenders[:1]),
        ("def_bottom", defenders[1:2]),
        ("mid", mids[:1]),
        ("fwd_top", forwards[:1]),
        ("fwd_bottom", forwards[1:2]),
    ]
    for slot_name, slot_gids in slot_order:
        if slot_gids:
            out[int(slot_gids[0])] = slots[slot_name]

    extra_y = np.linspace(0.24 * pitch_cfg.width, 0.76 * pitch_cfg.width, num=len(extras)) if extras else []
    for gid, y in zip(extras, extra_y):
        out[int(gid)] = (slots["mid"][0], float(y))
    return out


def _build_pass_map_nodes(
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    teams: Optional[Dict[str, set]] = None,
    *,
    active_gids: Optional[set[int]] = None,
) -> Dict[int, Tuple[float, float]]:
    teams = teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    pitch_cfg = SoccerPitchConfiguration()
    mean_pos = _mean_gid_positions(gid_pos_map)
    team_gids_all = {int(gid) for gid in teams.get("white", set())} | {int(gid) for gid in teams.get("black", set())}
    if active_gids is None:
        active_gids = set(mean_pos.keys()) | team_gids_all
    else:
        active_gids = {int(gid) for gid in active_gids} | team_gids_all

    left_team, right_team = _pass_map_team_sides(teams)
    out: Dict[int, Tuple[float, float]] = {}
    assigned_gids = set()
    side_specs = [
        (left_team, "left", PASS_MAP_LEFT_GOALKEEPER_GID),
        (right_team, "right", PASS_MAP_RIGHT_GOALKEEPER_GID),
    ]
    for team_name, side, goalie_gid in side_specs:
        team_gids = [gid for gid in sorted(teams.get(team_name, set())) if gid in active_gids]
        nodes = _build_team_formation_nodes(team_gids, mean_pos, side=side, goalie_gid=goalie_gid, pitch_cfg=pitch_cfg)
        out.update(nodes)
        assigned_gids.update(nodes.keys())

    leftovers = sorted(gid for gid in active_gids if gid not in assigned_gids)
    if leftovers:
        y_positions = np.linspace(0.25 * pitch_cfg.width, 0.75 * pitch_cfg.width, num=len(leftovers))
        for gid, y in zip(leftovers, y_positions):
            out[int(gid)] = (0.50 * pitch_cfg.length, float(y))
    return out


def _zone_grid(pitch_cfg: SoccerPitchConfiguration, *, cols: int = 6, rows: int = 3) -> List[dict]:
    zone_w = float(pitch_cfg.length) / float(cols)
    zone_h = float(pitch_cfg.width) / float(rows)
    zones = []
    zone_id = 1
    for col in range(cols):
        for row in range(rows):
            x_min = col * zone_w
            x_max = (col + 1) * zone_w
            y_min = row * zone_h
            y_max = (row + 1) * zone_h
            zones.append(
                dict(
                    zone_id=zone_id,
                    row=row + 1,
                    col=col + 1,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    cx=0.5 * (x_min + x_max),
                    cy=0.5 * (y_min + y_max),
                )
            )
            zone_id += 1
    return zones


def _zone_id_for_point(
    point: Tuple[float, float],
    pitch_cfg: SoccerPitchConfiguration,
    *,
    cols: int = 6,
    rows: int = 3,
) -> int:
    x = float(np.clip(point[0], 0.0, float(pitch_cfg.length) - 1e-6))
    y = float(np.clip(point[1], 0.0, float(pitch_cfg.width) - 1e-6))
    col = min(cols - 1, max(0, int(x / max(1e-6, float(pitch_cfg.length) / cols))))
    row = min(rows - 1, max(0, int(y / max(1e-6, float(pitch_cfg.width) / rows))))
    return col * rows + row + 1


def _compute_zone_control(
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    teams: Optional[Dict[str, set]] = None,
    *,
    cols: int = 6,
    rows: int = 3,
) -> pd.DataFrame:
    teams = teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    pitch_cfg = SoccerPitchConfiguration()
    zones = _zone_grid(pitch_cfg, cols=cols, rows=rows)
    total_frames = len(gid_pos_map)
    stats = {
        int(z["zone_id"]): {
            **z,
            "white_players": 0,
            "black_players": 0,
            "white_frames_control": 0,
            "black_frames_control": 0,
            "tie_frames": 0,
            "empty_frames": 0,
            "occupied_frames": 0,
        }
        for z in zones
    }
    for _, frame_map in sorted(gid_pos_map.items()):
        frame_zone_counts = {int(z["zone_id"]): {"white": 0, "black": 0} for z in zones}
        for gid, pos in frame_map.items():
            team = _team_of_gid(int(gid), teams)
            if team not in ("white", "black"):
                continue
            zone_id = _zone_id_for_point(pos, pitch_cfg, cols=cols, rows=rows)
            frame_zone_counts[zone_id][team] += 1
            stats[zone_id][f"{team}_players"] += 1
        for zone_id, team_counts in frame_zone_counts.items():
            white_n = int(team_counts["white"])
            black_n = int(team_counts["black"])
            if white_n == 0 and black_n == 0:
                stats[zone_id]["empty_frames"] += 1
                continue
            stats[zone_id]["occupied_frames"] += 1
            if white_n > black_n:
                stats[zone_id]["white_frames_control"] += 1
            elif black_n > white_n:
                stats[zone_id]["black_frames_control"] += 1
            else:
                stats[zone_id]["tie_frames"] += 1

    rows_out = []
    for zone in zones:
        zone_id = int(zone["zone_id"])
        row = stats[zone_id]
        occupied = int(row["occupied_frames"])
        total_player_events = int(row["white_players"] + row["black_players"])
        white_control_pct = 100.0 * float(row["white_frames_control"]) / max(1, occupied)
        black_control_pct = 100.0 * float(row["black_frames_control"]) / max(1, occupied)
        white_presence_pct = 100.0 * float(row["white_players"]) / max(1, total_player_events)
        black_presence_pct = 100.0 * float(row["black_players"]) / max(1, total_player_events)
        dominant_team = "tie"
        dominant_pct = 0.0
        if row["white_frames_control"] > row["black_frames_control"]:
            dominant_team = "white"
            dominant_pct = white_control_pct
        elif row["black_frames_control"] > row["white_frames_control"]:
            dominant_team = "black"
            dominant_pct = black_control_pct
        elif occupied > 0:
            dominant_pct = max(white_control_pct, black_control_pct)
        rows_out.append(
            dict(
                zone_id=zone_id,
                row=int(zone["row"]),
                col=int(zone["col"]),
                x_min=float(zone["x_min"]),
                x_max=float(zone["x_max"]),
                y_min=float(zone["y_min"]),
                y_max=float(zone["y_max"]),
                total_frames=int(total_frames),
                occupied_frames=occupied,
                empty_frames=int(row["empty_frames"]),
                white_players=int(row["white_players"]),
                black_players=int(row["black_players"]),
                white_frames_control=int(row["white_frames_control"]),
                black_frames_control=int(row["black_frames_control"]),
                tie_frames=int(row["tie_frames"]),
                white_control_pct=float(white_control_pct),
                black_control_pct=float(black_control_pct),
                white_presence_pct=float(white_presence_pct),
                black_presence_pct=float(black_presence_pct),
                dominant_team=dominant_team,
                dominant_control_pct=float(dominant_pct),
            )
        )
    return pd.DataFrame(rows_out).sort_values(["zone_id"]).reset_index(drop=True)


def _render_zone_control_report(
    out_path: str,
    zone_df: pd.DataFrame,
    *,
    width: int = 1200,
    height: int = 700,
) -> None:
    if zone_df is None or zone_df.empty:
        print(f"[SKIP] {out_path}: sin datos de zonas")
        return
    renderer, canvas, off_x, off_y = _make_pitch_canvas(width=width, height=height, padding=36)
    for _, row in zone_df.iterrows():
        p1 = _pitch_point_to_canvas(renderer, (float(row["x_min"]), float(row["y_min"])), off_x=off_x, off_y=off_y)
        p2 = _pitch_point_to_canvas(renderer, (float(row["x_max"]), float(row["y_max"])), off_x=off_x, off_y=off_y)
        dominant = str(row.get("dominant_team", "tie"))
        strength = float(np.clip(float(row.get("dominant_control_pct", 0.0)) / 100.0, 0.0, 1.0))
        if dominant == "white":
            fill = (0, 255, 0)
            label = f"W {row['white_control_pct']:.0f}%"
        elif dominant == "black":
            fill = (0, 165, 255)
            label = f"B {row['black_control_pct']:.0f}%"
        else:
            fill = (170, 170, 170)
            label = "TIE"
        alpha = 0.16 + 0.34 * strength
        shade = canvas.copy()
        cv2.rectangle(shade, p1, p2, fill, -1)
        canvas = cv2.addWeighted(shade, alpha, canvas, 1.0 - alpha, 0)
        cv2.rectangle(canvas, p1, p2, (235, 235, 235), 1, lineType=cv2.LINE_AA)
        tx = int(min(p1[0], p2[0]) + 10)
        ty = int(min(p1[1], p2[1]) + 24)
        cv2.putText(canvas, f"Z{int(row['zone_id']):02d}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Control territorial por zonas (18)", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"[WRITE] {out_path}")


def _render_pitch_voronoi_radar(
    pitch_renderer: PitchRenderer,
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    frame_idx: int,
    *,
    ball_row: Optional[pd.Series] = None,
    teams: Optional[Dict[str, set]] = None,
) -> np.ndarray:
    teams = teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    base = pitch_renderer.base_pitch.copy()
    point_radius = max(6, int(round(28.0 * float(pitch_renderer.scale))))
    point_outline = max(1, int(round(8.0 * float(pitch_renderer.scale))))
    line_thickness = max(1, int(round(10.0 * float(pitch_renderer.scale))))
    font_scale = max(0.45, 3.8 * float(pitch_renderer.scale))
    font_thickness = max(1, int(round(8.0 * float(pitch_renderer.scale))))
    ball_radius = max(5, int(round(22.0 * float(pitch_renderer.scale))))
    frame_map = gid_pos_map.get(frame_idx, {})
    if not frame_map:
        if ball_row is not None:
            return _render_pitch_radar(pitch_renderer, gid_pos_map, frame_idx, ball_row=ball_row, team_mode=True, teams=teams)
        return base

    rect = (
        int(pitch_renderer.padding),
        int(pitch_renderer.padding),
        int(max(1, pitch_renderer.scaled_length)),
        int(max(1, pitch_renderer.scaled_width)),
    )
    subdiv = cv2.Subdiv2D(rect)
    team_colors = {"white": (0, 255, 0), "black": (0, 165, 255)}
    sites: List[Tuple[Tuple[int, int], str, int]] = []
    used_points = set()
    for gid, (x, y) in sorted(frame_map.items()):
        team = _team_of_gid(int(gid), teams)
        if team not in team_colors:
            continue
        px, py = pitch_renderer._scale_point([float(x), float(y)])
        px = int(np.clip(px, rect[0], rect[0] + rect[2] - 1))
        py = int(np.clip(py, rect[1], rect[1] + rect[3] - 1))
        while (px, py) in used_points and py < rect[1] + rect[3] - 1:
            py += 1
        if (px, py) in used_points:
            continue
        used_points.add((px, py))
        subdiv.insert((float(px), float(py)))
        sites.append(((px, py), team, int(gid)))

    if len(sites) < 2 or len({team for _, team, _ in sites}) < 2:
        return _render_pitch_radar(pitch_renderer, gid_pos_map, frame_idx, ball_row=ball_row, team_mode=True, teams=teams)

    overlay = base.copy()
    facets, centers = subdiv.getVoronoiFacetList([])
    if facets is None or centers is None:
        return _render_pitch_radar(pitch_renderer, gid_pos_map, frame_idx, ball_row=ball_row, team_mode=True, teams=teams)

    for facet, center in zip(facets, centers):
        poly = np.array(facet, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        cx, cy = float(center[0]), float(center[1])
        nearest = min(sites, key=lambda item: (item[0][0] - cx) ** 2 + (item[0][1] - cy) ** 2)
        fill = team_colors.get(nearest[1], (150, 150, 150))
        poly[:, 0] = np.clip(poly[:, 0], rect[0], rect[0] + rect[2] - 1)
        poly[:, 1] = np.clip(poly[:, 1], rect[1], rect[1] + rect[3] - 1)
        cv2.fillConvexPoly(overlay, np.round(poly).astype(np.int32), fill, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [np.round(poly).astype(np.int32)], True, (245, 245, 245), line_thickness, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(overlay, 0.34, base, 0.66, 0)
    for (px, py), team, gid in sites:
        color = team_colors.get(team, (255, 255, 255))
        cv2.circle(out, (int(px), int(py)), point_radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(px), int(py)), point_radius, (25, 25, 25), point_outline, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            f"G{gid}",
            (int(px) + point_radius + 4, int(py) + max(4, point_radius // 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )
    if ball_row is not None:
        bx = ball_row.get("ball_x")
        by = ball_row.get("ball_y")
        if pd.notna(bx) and pd.notna(by):
            bp = pitch_renderer._scale_point([float(bx), float(by)])
            cv2.circle(out, bp, ball_radius, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, bp, ball_radius, (25, 25, 25), point_outline, lineType=cv2.LINE_AA)
    return out


def _render_voronoi_radar_video(
    out_path: str,
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    ball_df: Optional[pd.DataFrame],
    fps: float,
    *,
    team_sets: Optional[Dict[str, set]] = None,
) -> None:
    if not gid_pos_map:
        print(f"[SKIP] {out_path}: sin posiciones para Voronoi")
        return
    pitch_cfg = SoccerPitchConfiguration()
    target_w = int(getattr(config, "DEBUG_VORONOI_WIDTH", 1440))
    target_h = int(getattr(config, "DEBUG_VORONOI_HEIGHT", 820))
    padding = int(getattr(config, "DEBUG_VORONOI_PADDING", 64))
    scale_x = max(1e-6, float(target_w - 2 * padding) / float(pitch_cfg.length))
    scale_y = max(1e-6, float(target_h - 2 * padding) / float(pitch_cfg.width))
    pitch_renderer = PitchRenderer(config=pitch_cfg, scale=min(scale_x, scale_y), padding=padding)
    base = pitch_renderer.base_pitch
    size = (base.shape[1], base.shape[0])
    max_frame = max(int(max(gid_pos_map.keys())), int(ball_df["frame"].max()) if ball_df is not None and not ball_df.empty else 0)
    start, end = _frame_window(fps, max_frame)
    writer = _setup_writer(out_path, fps if fps > 0 else 24.0, size)
    for f in range(start, end + 1):
        ball_row = None
        if ball_df is not None and not ball_df.empty:
            br = ball_df[(ball_df["frame"] == f) & (ball_df.get("is_clean", True))]
            if not br.empty:
                ball_row = br.iloc[0]
        frame = _render_pitch_voronoi_radar(
            pitch_renderer,
            gid_pos_map,
            f,
            ball_row=ball_row,
            teams=team_sets,
        )
        cv2.putText(frame, f"f={f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[WRITE] {out_path}")


def _compute_possession_frames(
    ball_df: pd.DataFrame,
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    *,
    radius: float,
    hold_frames: int,
    dispute_frames: int,
    teams: Optional[Dict[str, set]] = None,
) -> pd.DataFrame:
    """Replica la lógica de posesión: distancia a balón + persistencia y resolución de disputas."""
    if ball_df is None or ball_df.empty:
        return pd.DataFrame(columns=["frame", "gid", "team", "pos_x", "pos_y", "ball_x", "ball_y", "dist", "unique_possession"])
    teams = teams or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    records = []
    for _, r in ball_df.iterrows():
        if not bool(r.get("is_clean", True)):
            continue
        if pd.isna(r.get("ball_x")) or pd.isna(r.get("ball_y")):
            continue
        f = int(r["frame"])
        bx = float(r["ball_x"])
        by = float(r["ball_y"])
        if f not in gid_pos_map:
            continue
        for gid, (px, py) in gid_pos_map[f].items():
            dist = float(np.hypot(bx - px, by - py))
            records.append(
                dict(
                    frame=f,
                    gid=int(gid),
                    team=_team_of_gid(int(gid), teams),
                    pos_x=float(px),
                    pos_y=float(py),
                    ball_x=bx,
                    ball_y=by,
                    dist=dist,
                )
            )
    if not records:
        return pd.DataFrame(columns=["frame", "gid", "team", "pos_x", "pos_y", "ball_x", "ball_y", "dist", "unique_possession"])
    df = pd.DataFrame(records)
    df = df.sort_values(["frame", "gid"]).reset_index(drop=True)
    df["initial_possession"] = df["dist"] <= float(radius)
    # Persistencia: rolling por jugador
    df["validated_possession"] = False
    df["roll_valid"] = 0
    for gid, g in df.groupby("gid"):
        roll = g["initial_possession"].rolling(window=max(1, int(hold_frames)), min_periods=1).sum()
        df.loc[g.index, "roll_valid"] = roll
        df.loc[g.index, "validated_possession"] = roll >= hold_frames
    # Disputas: único por frame, si más de uno valido, quedar con los que tienen roll_valid >= dispute_frames
    df["unique_possession"] = False
    for frame, g in df[df["validated_possession"]].groupby("frame"):
        if len(g) == 1:
            df.loc[g.index, "unique_possession"] = True
        else:
            keep_idx = g[g["roll_valid"] >= dispute_frames].index
            if len(keep_idx):
                df.loc[keep_idx, "unique_possession"] = True
    return df


def _detect_passes_sequence(
    possession_df: pd.DataFrame,
    *,
    allow_cross: bool,
) -> pd.DataFrame:
    """Genera secuencia de dueño único por frame y registra cambios como pases."""
    cols = ["id_emisor", "id_receptor", "team_emisor", "team_receptor", "Frame", "frame_end", "X_ball_inicio", "Y_ball_inicio", "X_ball_Final", "Y_ball_final"]
    if possession_df is None or possession_df.empty:
        return pd.DataFrame(columns=cols)
    seq = []
    for frame, g in possession_df[possession_df["unique_possession"]].groupby("frame"):
        if len(g) == 1:
            row = g.iloc[0]
        else:
            # escoger el más cercano al balón
            row = g.loc[g["dist"].idxmin()]
        seq.append(
            dict(
                frame=int(row["frame"]),
                gid=int(row["gid"]),
                team=str(row.get("team", "unknown")),
                ball_x=float(row["ball_x"]),
                ball_y=float(row["ball_y"]),
            )
        )
    passes = []
    for i in range(1, len(seq)):
        prev = seq[i - 1]
        curr = seq[i]
        if prev["gid"] == curr["gid"]:
            continue
        if not allow_cross and prev["team"] != "unknown" and curr["team"] != "unknown" and prev["team"] != curr["team"]:
            continue
        passes.append(
            dict(
                id_emisor=prev["gid"],
                id_receptor=curr["gid"],
                team_emisor=prev["team"],
                team_receptor=curr["team"],
                Frame=prev["frame"],
                frame_end=curr["frame"],
                X_ball_inicio=prev["ball_x"],
                Y_ball_inicio=prev["ball_y"],
                X_ball_Final=curr["ball_x"],
                Y_ball_final=curr["ball_y"],
            )
        )
    return pd.DataFrame(passes, columns=cols)


def _detect_passes_pitch(
    ball_df: pd.DataFrame,
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    team_sets: Optional[Dict[str, set]] = None,
    *,
    pass_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Detecta pases usando posesión en cancha (ball_x/ball_y vs pos_x/pos_y) con histeresis simple.
    """
    if ball_df is None or ball_df.empty:
        return pd.DataFrame(columns=["kick_frame", "recv_frame", "from_gid", "to_gid", "cam_kick", "cam_recv", "duration_frames", "dist_px"])
    teams = team_sets or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    params = pass_params or {}
    r_in = float(params.get("POS_R_IN", getattr(config, "PASS_POS_RADIUS_IN", 600.0)))
    r_out = float(params.get("POS_R_OUT", getattr(config, "PASS_POS_RADIUS_OUT", 900.0)))
    n_hold = int(params.get("POS_HOLD", getattr(config, "PASS_POS_HOLD", 2)))
    n_release = int(params.get("POS_RELEASE", getattr(config, "PASS_POS_RELEASE", 3)))
    min_travel = float(params.get("POS_MIN_TRAVEL", getattr(config, "PASS_POS_MIN_TRAVEL", 80.0)))
    max_gap = int(params.get("POS_MAX_GAP", getattr(config, "PASS_POS_MAX_GAP", 20)))
    allow_cross = bool(getattr(config, "PASS_ALLOW_CROSS_TEAM", False))

    df = ball_df.copy()
    df = df[df.get("is_clean", True)]
    df = df.dropna(subset=["ball_x", "ball_y"])
    if df.empty:
        return pd.DataFrame(columns=["kick_frame", "recv_frame", "from_gid", "to_gid", "cam_kick", "cam_recv", "duration_frames", "dist_px"])
    df = df.sort_values("frame")

    rows = []
    owner = None
    owner_team = None
    owner_start_frame = None
    owner_ball_start = None
    hold = 0
    release = 0
    change_gid = None
    change_hold = 0
    last_frame = None

    def _team_of(gid: Optional[int]) -> Optional[str]:
        if gid is None:
            return None
        if gid in teams.get("white", set()):
            return "white"
        if gid in teams.get("black", set()):
            return "black"
        return None

    for _, r in df.iterrows():
        f = int(r["frame"])
        bx = float(r["ball_x"])
        by = float(r["ball_y"])
        cam_src = str(r.get("cam_source", ""))
        ball_pos = np.array([bx, by], dtype=np.float32)

        if last_frame is not None and (f - last_frame) > max_gap:
            owner = None
            owner_team = None
            owner_start_frame = None
            owner_ball_start = None
            hold = release = 0
            change_gid = None
            change_hold = 0
        last_frame = f

        best_gid = None
        best_dist = None
        if f in gid_pos_map:
            for gid, pos in gid_pos_map[f].items():
                d = float(np.hypot(bx - pos[0], by - pos[1]))
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_gid = gid
        else:
            # tolerancia de ±2 frames para no perder posesión si falta la posición exacta
            for df_tol in (1, 2):
                for ff in (f - df_tol, f + df_tol):
                    if ff not in gid_pos_map:
                        continue
                    for gid, pos in gid_pos_map[ff].items():
                        d = float(np.hypot(bx - pos[0], by - pos[1]))
                        if best_dist is None or d < best_dist:
                            best_dist = d
                            best_gid = gid

        if owner is None:
            if best_gid is not None and best_dist is not None and best_dist <= r_in:
                hold += 1
                if hold >= n_hold:
                    owner = best_gid
                    owner_team = _team_of(owner)
                    owner_start_frame = f
                    owner_ball_start = ball_pos.copy()
                    hold = 0
            else:
                hold = 0
        else:
            if best_gid == owner and best_dist is not None and best_dist <= r_out:
                release = 0
                change_gid = None
                change_hold = 0
            elif best_gid is not None and best_dist is not None and best_dist <= r_in:
                if change_gid == best_gid:
                    change_hold += 1
                else:
                    change_gid = best_gid
                    change_hold = 1
                if change_hold >= n_hold:
                    prev_owner = owner
                    prev_team = owner_team
                    owner = best_gid
                    owner_team = _team_of(owner)
                    change_gid = None
                    change_hold = 0
                    if prev_owner is not None and owner is not None and prev_owner != owner:
                        if allow_cross or (prev_team and owner_team and prev_team == owner_team):
                            travel = float(np.linalg.norm(ball_pos - owner_ball_start)) if owner_ball_start is not None else 0.0
                            if travel >= min_travel:
                                rows.append(
                                    dict(
                                        kick_frame=owner_start_frame if owner_start_frame is not None else f,
                                        recv_frame=f,
                                        from_gid=prev_owner,
                                        to_gid=owner,
                                        cam_kick=cam_src,
                                        cam_recv=cam_src,
                                        duration_frames=f - (owner_start_frame if owner_start_frame is not None else f),
                                        dist_px=travel,
                                    )
                                )
                    owner_start_frame = f
                    owner_ball_start = ball_pos.copy()
                    release = 0
            else:
                release += 1
                if release >= n_release:
                    owner = None
                    owner_team = None
                    owner_start_frame = None
                    owner_ball_start = None
                    release = 0
                    change_gid = None
                    change_hold = 0

    return pd.DataFrame(rows)


def _draw_points(
    canvas: np.ndarray,
    points: List[Tuple[float, float, Tuple[int, int, int], str]],
    bounds: Tuple[float, float, float, float],
    *,
    ball: Optional[Tuple[float, float]] = None,
    ball_clean: bool = True,
):
    h, w = canvas.shape[:2]
    min_x, max_x, min_y, max_y = bounds
    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)
    for x, y, color, label in points:
        px = int(np.clip((x - min_x) / span_x * w, 0, w - 1))
        py = int(np.clip((y - min_y) / span_y * h, 0, h - 1))
        cv2.circle(canvas, (px, py), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if ball is not None:
        bx, by = ball
        px = int(np.clip((bx - min_x) / span_x * w, 0, w - 1))
        py = int(np.clip((by - min_y) / span_y * h, 0, h - 1))
        color = (0, 255, 255) if ball_clean else (0, 165, 255)
        cv2.circle(canvas, (px, py), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "Ball", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _render_stage(
    out_path: str,
    fps: float,
    bounds: Tuple[float, float, float, float],
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    use_map: bool = False,
    gid_map: Optional[Dict[str, Dict[int, int]]] = None,
    segs: Optional[Dict[str, List[dict]]] = None,
    mapped_only: bool = False,
    draw_lines: bool = False,
    metrics_rows: Optional[List[dict]] = None,
    ball_df: Optional[pd.DataFrame] = None,
):
    if "frame" not in df1.columns or "frame" not in df2.columns:
        return
    max_frame = int(max(df1["frame"].max(), df2["frame"].max()))
    start, end = _frame_window(fps, max_frame)
    w_cfg = getattr(config, "DEBUG_VIZ_WIDTH", 960)
    h_cfg = getattr(config, "DEBUG_VIZ_HEIGHT", 540)
    size = (int(w_cfg), int(h_cfg))
    writer = _setup_writer(out_path, fps, size)
    gid_map = gid_map if gid_map is not None else (_load_map(config.RUN_DIR) if use_map else {"cam1": {}, "cam2": {}})
    segs = segs or {"cam1": [], "cam2": []}

    for f in range(start, end + 1):
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        rows1 = df1[df1["frame"] == f]
        rows2 = df2[df2["frame"] == f]

        pts = []
        gid_pos_cam1 = {}
        gid_pos_cam2 = {}
        ball_pt = None
        ball_clean = True
        if ball_df is not None and not ball_df.empty:
            row_ball = ball_df[ball_df["frame"] == f]
            if not row_ball.empty and pd.notna(row_ball.iloc[0].get("ball_x")) and pd.notna(row_ball.iloc[0].get("ball_y")):
                ball_pt = (float(row_ball.iloc[0]["ball_x"]), float(row_ball.iloc[0]["ball_y"]))
                ball_clean = bool(row_ball.iloc[0].get("is_clean", True))
        def _add_rows(rows: pd.DataFrame, cam_label: str):
            for _, r in rows.iterrows():
                sid = int(r["stable_id"])
                gid = _resolve_gid(cam_label, sid, f, gid_map, segs)
                if mapped_only and gid is None:
                    continue
                label = f"{cam_label}:{sid}"
                color = (255, 0, 0) if cam_label == "cam1" else (0, 0, 255)
                if gid is not None:
                    color = _hash_color(gid)
                    label = f"{cam_label}:{sid}->g{gid}"
                x = float(r["pos_x"])
                y = float(r["pos_y"])
                pts.append((x, y, color, label))
                if gid is not None:
                    if cam_label == "cam1":
                        gid_pos_cam1[gid] = (x, y, color)
                    else:
                        gid_pos_cam2[gid] = (x, y, color)
        _add_rows(rows1, "cam1")
        _add_rows(rows2, "cam2")

        _draw_points(canvas, pts, bounds, ball=ball_pt, ball_clean=ball_clean)

        dist_list = []
        if draw_lines:
            shared = set(gid_pos_cam1.keys()) & set(gid_pos_cam2.keys())
            for gid in shared:
                x1, y1, c1 = gid_pos_cam1[gid]
                x2, y2, _ = gid_pos_cam2[gid]
                # same scaling as points
                min_x, max_x, min_y, max_y = bounds
                span_x = max(1e-3, max_x - min_x)
                span_y = max(1e-3, max_y - min_y)
                p1 = (
                    int(np.clip((x1 - min_x) / span_x * size[0], 0, size[0] - 1)),
                    int(np.clip((y1 - min_y) / span_y * size[1], 0, size[1] - 1)),
                )
                p2 = (
                    int(np.clip((x2 - min_x) / span_x * size[0], 0, size[0] - 1)),
                    int(np.clip((y2 - min_y) / span_y * size[1], 0, size[1] - 1)),
                )
                cv2.line(canvas, p1, p2, (0, 255, 255), 1, lineType=cv2.LINE_AA)
                dist_list.append(np.linalg.norm([x1 - x2, y1 - y2]))
        if metrics_rows is not None:
            metrics_rows.append(
                dict(
                    frame=int(f),
                    cam1_pts=int(len(rows1)),
                    cam2_pts=int(len(rows2)),
                    mapped_only=bool(mapped_only),
                    mapped_gids_cam1=int(len(gid_pos_cam1)),
                    mapped_gids_cam2=int(len(gid_pos_cam2)),
                    shared_gids=int(len(set(gid_pos_cam1.keys()) & set(gid_pos_cam2.keys()))),
                    dist_med=float(np.median(dist_list)) if dist_list else np.nan,
                    dist_p80=float(np.percentile(dist_list, 80)) if dist_list else np.nan,
                )
            )

        writer.write(canvas)
    writer.release()


def _render_cam_video(
    out_path: str,
    video_path: Optional[str],
    df: pd.DataFrame,
    cam_label: str,
    gid_map: Optional[Dict[str, Dict[int, int]]],
    segs: Optional[Dict[str, List[dict]]],
    ball_df: Optional[pd.DataFrame],
    fps_hint: float,
):
    if df.empty:
        return
    required_cols = {"frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "stable_id"}
    if not required_cols.issubset(df.columns):
        print(f"[SKIP] {cam_label} video: faltan columnas de bbox en tracks")
        return
    if not video_path or not os.path.exists(video_path):
        print(f"[SKIP] {cam_label} video: no se encontró el archivo de video ({video_path})")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] {cam_label} video: no se pudo abrir {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = float(cap.get(cv2.CAP_PROP_FPS))
    fps_out = fps_video if fps_video > 1e-3 else float(fps_hint or 24.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ret, sample = cap.read()
        if not ret:
            print(f"[SKIP] {cam_label} video: no se pudo leer frame inicial")
            cap.release()
            return
        h, w = sample.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = _setup_writer(out_path, fps_out if fps_out > 0 else 24.0, (w, h))
    frame_groups = {int(k): v for k, v in df.groupby("frame")}
    max_frame_df = int(df["frame"].max())
    max_frame = max_frame_df
    if total_frames > 0:
        max_frame = min(max_frame_df, total_frames - 1)
    start, end = _frame_window(fps_out if fps_out > 0 else fps_hint, max_frame)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    base_color = (255, 0, 0) if cam_label == "cam1" else (0, 0, 255)
    gid_map = gid_map or {"cam1": {}, "cam2": {}}
    segs = segs or {"cam1": [], "cam2": []}

    for f in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        ball_row = None
        if ball_df is not None and not ball_df.empty:
            br = ball_df[(ball_df["frame"] == f) & (ball_df.get("is_clean", True))]
            if not br.empty:
                ball_row = br.iloc[0]
        rows = frame_groups.get(f)
        if rows is not None:
            for _, r in rows.iterrows():
                try:
                    x1 = int(round(float(r["bbox_x1"])))
                    y1 = int(round(float(r["bbox_y1"])))
                    x2 = int(round(float(r["bbox_x2"])))
                    y2 = int(round(float(r["bbox_y2"])))
                except Exception:
                    continue
                x1 = int(np.clip(x1, 0, max(w - 1, 1)))
                y1 = int(np.clip(y1, 0, max(h - 1, 1)))
                x2 = int(np.clip(x2, 0, max(w - 1, 1)))
                y2 = int(np.clip(y2, 0, max(h - 1, 1)))

                gid_val: Optional[int] = None
                if "gid" in r and pd.notna(r["gid"]):
                    try:
                        gid_val = int(r["gid"])
                    except Exception:
                        gid_val = None
                if gid_val is None:
                    gid_val = _resolve_gid(cam_label, int(r["stable_id"]), f, gid_map, segs)

                color = _hash_color(gid_val) if gid_val is not None else base_color
                label = f"{cam_label}:{int(r['stable_id'])}"
                if gid_val is not None:
                    label = f"{label}->G{gid_val}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        if ball_row is not None and str(ball_row.get("cam_source", "")) == cam_label:
            if {"ball_bbox_x1", "ball_bbox_y1", "ball_bbox_x2", "ball_bbox_y2"}.issubset(ball_row.index):
                try:
                    bx1 = int(round(float(ball_row["ball_bbox_x1"])))
                    by1 = int(round(float(ball_row["ball_bbox_y1"])))
                    bx2 = int(round(float(ball_row["ball_bbox_x2"])))
                    by2 = int(round(float(ball_row["ball_bbox_y2"])))
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(frame, "Ball", (bx1, max(12, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
        overlay_y = 20
        cv2.putText(frame, f"f={f}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        overlay_y += 22
        if ball_row is not None:
            cam_src = str(ball_row.get("cam_source", ""))
            bx = ball_row.get("ball_x")
            by = ball_row.get("ball_y")
            if pd.notna(bx) and pd.notna(by):
                txt = f"Ball {cam_src} ({bx:.1f},{by:.1f})"
                cv2.putText(frame, txt, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cap.release()
    print(f"[WRITE] {out_path}")


def _render_combined_video(
    out_path: str,
    video_cam1: Optional[str],
    video_cam2: Optional[str],
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    gid_map: Dict[str, Dict[int, int]],
    segs: Dict[str, List[dict]],
    ball_df: Optional[pd.DataFrame],
    fps_hint: float,
    *,
    team_mode: bool = False,
    team_sets: Optional[Dict[str, set]] = None,
):
    required_cols = {"frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "stable_id"}
    if df1.empty or df2.empty or not required_cols.issubset(df1.columns) or not required_cols.issubset(df2.columns):
        print("[SKIP] video combinado: faltan datos o columnas")
        return
    if not video_cam1 or not os.path.exists(video_cam1) or not video_cam2 or not os.path.exists(video_cam2):
        print(f"[SKIP] video combinado: no se encontraron videos ({video_cam1}, {video_cam2})")
        return

    cap1 = cv2.VideoCapture(video_cam1)
    cap2 = cv2.VideoCapture(video_cam2)
    if not cap1.isOpened() or not cap2.isOpened():
        print(f"[SKIP] video combinado: no se pudieron abrir videos")
        cap1.release()
        cap2.release()
        return

    fps1 = float(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = float(cap2.get(cv2.CAP_PROP_FPS))
    fps_out = fps1 if fps1 > 1e-3 else (fps2 if fps2 > 1e-3 else float(fps_hint or 24.0))

    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        print("[SKIP] video combinado: dimensiones inválidas")
        cap1.release()
        cap2.release()
        return

    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame_df = min(int(df1["frame"].max()), int(df2["frame"].max()))
    max_frame = max_frame_df
    if total1 > 0 and total2 > 0:
        max_frame = min(max_frame_df, total1 - 1, total2 - 1)
    start, end = _frame_window(fps_out, max_frame)
    if start > 0:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, start)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_groups1 = {int(k): v for k, v in df1.groupby("frame")}
    frame_groups2 = {int(k): v for k, v in df2.groupby("frame")}
    size_out = (w1 + w2, max(h1, h2))
    writer = _setup_writer(out_path, fps_out if fps_out > 0 else 24.0, size_out)

    teams = team_sets or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    pitch_cfg = SoccerPitchConfiguration()
    pitch_renderer = PitchRenderer(config=pitch_cfg, scale=0.1, padding=50)
    gid_pos_cam1 = _build_gid_positions(df1, "cam1", gid_map, segs)
    gid_pos_cam2 = _build_gid_positions(df2, "cam2", gid_map, segs)
    # Prioridad explícita: usar cam2 en el radar y caer a cam1 solo cuando falte el GID.
    radar_gid_pos_map = _merge_gid_positions(gid_pos_cam2, gid_pos_cam1)
    radar_base = pitch_renderer.base_pitch
    radar_base_h, radar_base_w = radar_base.shape[:2]
    radar_target_h = max(120, min(radar_base_h, int(size_out[1] * 0.24)))
    radar_target_w = max(1, int(round(radar_target_h * (radar_base_w / max(1, radar_base_h)))))

    def _team_style(gid: int) -> Tuple[Optional[str], Tuple[int, int, int]]:
        if gid in teams.get("white", set()):
            return "W", (0, 255, 0)  # verde
        if gid in teams.get("black", set()):
            return "B", (0, 165, 255)  # naranjo
        return None, _hash_color(gid)

    def _draw_rows(frame, rows, cam_label, frame_idx: int):
        base_color = (255, 0, 0) if cam_label == "cam1" else (0, 0, 255)
        for _, r in rows.iterrows():
            gid_val = None
            if "gid" in r and pd.notna(r["gid"]):
                try:
                    gid_val = int(r["gid"])
                except Exception:
                    gid_val = None
            if gid_val is None:
                gid_val = _resolve_gid(cam_label, int(r["stable_id"]), frame_idx, gid_map, segs)
            if gid_val is None:
                continue  # solo queremos la ID final unificada
            try:
                x1 = int(round(float(r["bbox_x1"])))
                y1 = int(round(float(r["bbox_y1"])))
                x2 = int(round(float(r["bbox_x2"])))
                y2 = int(round(float(r["bbox_y2"])))
            except Exception:
                continue
            x1 = int(np.clip(x1, 0, max(frame.shape[1] - 1, 1)))
            y1 = int(np.clip(y1, 0, max(frame.shape[0] - 1, 1)))
            x2 = int(np.clip(x2, 0, max(frame.shape[1] - 1, 1)))
            y2 = int(np.clip(y2, 0, max(frame.shape[0] - 1, 1)))
            if team_mode:
                prefix, color = _team_style(gid_val)
                label = f"{prefix}{gid_val}" if prefix else f"G{gid_val}"
            else:
                color = _hash_color(gid_val)
                label = f"G{gid_val}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    for f in range(start, end + 1):
        ret1, fr1 = cap1.read()
        ret2, fr2 = cap2.read()
        if not ret1 or not ret2:
            break
        ball_row = None
        if ball_df is not None and not ball_df.empty:
            br = ball_df[(ball_df["frame"] == f) & (ball_df.get("is_clean", True))]
            if not br.empty:
                ball_row = br.iloc[0]
        rows1 = frame_groups1.get(f)
        rows2 = frame_groups2.get(f)
        if rows1 is not None:
            _draw_rows(fr1, rows1, "cam1", f)
        if rows2 is not None:
            _draw_rows(fr2, rows2, "cam2", f)
        if ball_row is not None and {"ball_bbox_x1", "ball_bbox_y1", "ball_bbox_x2", "ball_bbox_y2"}.issubset(ball_row.index):
            try:
                bx1 = int(round(float(ball_row["ball_bbox_x1"])))
                by1 = int(round(float(ball_row["ball_bbox_y1"])))
                bx2 = int(round(float(ball_row["ball_bbox_x2"])))
                by2 = int(round(float(ball_row["ball_bbox_y2"])))
                cam_src = str(ball_row.get("cam_source", ""))
                if cam_src == "cam1":
                    cv2.rectangle(fr1, (bx1, by1), (bx2, by2), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(fr1, "Ball", (bx1, max(12, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                elif cam_src == "cam2":
                    cv2.rectangle(fr2, (bx1, by1), (bx2, by2), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(fr2, "Ball", (bx1, max(12, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass
        canvas = np.zeros((size_out[1], size_out[0], 3), dtype=np.uint8)
        canvas[: h1, :w1] = fr1
        canvas[: h2, w1 : w1 + w2] = fr2
        radar_img = _render_pitch_radar(
            pitch_renderer,
            radar_gid_pos_map,
            f,
            ball_row=ball_row,
            team_mode=team_mode,
            teams=teams,
        )
        radar_resized = cv2.resize(radar_img, (radar_target_w, radar_target_h), interpolation=cv2.INTER_AREA)
        canvas = _blit_center_bottom(canvas, radar_resized)
        if ball_row is not None:
            bx = ball_row.get("ball_x")
            by = ball_row.get("ball_y")
            cam_src = str(ball_row.get("cam_source", ""))
            if pd.notna(bx) and pd.notna(by):
                clean = bool(ball_row.get("is_clean", True))
                txt = f"Ball {cam_src} ({bx:.1f},{by:.1f}){' [outlier]' if not clean else ''}"
                cv2.putText(canvas, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    cap1.release()
    cap2.release()
    print(f"[WRITE] {out_path}")


def _render_pass_network(
    out_path: str,
    events_df: pd.DataFrame,
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    bounds: Tuple[float, float, float, float],
    team_sets: Optional[Dict[str, set]] = None,
):
    """Renderiza un grafo de pases en el plano de la cancha."""
    if events_df is None or events_df.empty:
        print(f"[SKIP] {out_path}: sin eventos de pase")
        return
    # compatibilidad con distintos nombres de columnas
    fg_col = "from_gid" if "from_gid" in events_df.columns else "id_emisor"
    tg_col = "to_gid" if "to_gid" in events_df.columns else "id_receptor"
    required = {fg_col, tg_col}
    if not required.issubset(events_df.columns):
        print(f"[SKIP] {out_path}: faltan columnas requeridas")
        return
    teams = team_sets or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    events = []
    for _, r in events_df.iterrows():
        try:
            events.append((int(r[fg_col]), int(r[tg_col])))
        except Exception:
            continue
    if not events:
        print(f"[SKIP] {out_path}: no hay eventos válidos")
        return

    active_gids = {gid for edge in events for gid in edge}
    pass_map_teams = _infer_pass_map_team_sets(gid_pos_map, active_gids=active_gids, fallback_teams=teams)
    node_layout = _build_pass_map_nodes(gid_pos_map, pass_map_teams, active_gids=active_gids)
    if not node_layout:
        print(f"[SKIP] {out_path}: no hay layout de jugadores")
        return

    edge_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    passes_given: Dict[int, int] = {}
    for fg, tg in events:
        edge = edge_stats.setdefault((fg, tg), {"count": 0})
        edge["count"] += 1
        passes_given[fg] = passes_given.get(fg, 0) + 1

    w = int(getattr(config, "DEBUG_VIZ_WIDTH", 960))
    h = int(getattr(config, "DEBUG_VIZ_HEIGHT", 540))
    renderer, canvas, off_x, off_y = _make_pitch_canvas(width=w, height=h)

    def _to_px(pt: Tuple[float, float]) -> Tuple[int, int]:
        px, py = _pitch_point_to_canvas(renderer, pt, off_x=off_x, off_y=off_y)
        return int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))

    def _edge_color(gid: int) -> Tuple[int, int, int]:
        if gid in pass_map_teams.get("white", set()):
            return (0, 255, 0)
        if gid in pass_map_teams.get("black", set()):
            return (0, 165, 255)
        return (255, 255, 255)

    # dibujar edges
    for (fg, tg), st in edge_stats.items():
        count = st["count"]
        pf = node_layout.get(fg)
        pt = node_layout.get(tg)
        if pf is None or pt is None:
            continue
        p1 = _to_px(pf)
        p2 = _to_px(pt)
        thickness = max(1, int(round(1 + (count - 1) ** 0.5)))
        color = _edge_color(fg)
        cv2.arrowedLine(canvas, p1, p2, color, thickness, cv2.LINE_AA, tipLength=0.12)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(canvas, str(count), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # dibujar nodos
    for gid, pos in node_layout.items():
        p = _to_px(pos)
        color = _edge_color(gid)
        radius = 10 + min(10, 2 * int(passes_given.get(gid, 0)))
        cv2.circle(canvas, p, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p, radius, (25, 25, 25), 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"G{gid}", (p[0] + 12, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"[WRITE] {out_path}")


def _render_pass_map_matplotlib(
    out_path: str,
    events_df: pd.DataFrame,
    gid_pos_map: Dict[int, Dict[int, Tuple[float, float]]],
    team_sets: Optional[Dict[str, set]] = None,
):
    """Render estilo grafo con matplotlib, grosor por cantidad (similar al script de referencia)."""
    teams = team_sets or {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    if events_df is None or events_df.empty:
        print(f"[SKIP] {out_path}: sin eventos")
        return
    fg_col = "from_gid" if "from_gid" in events_df.columns else ("id_emisor" if "id_emisor" in events_df.columns else None)
    tg_col = "to_gid" if "to_gid" in events_df.columns else ("id_receptor" if "id_receptor" in events_df.columns else None)
    if fg_col is None or tg_col is None:
        print(f"[SKIP] {out_path}: columnas de emisor/receptor faltantes")
        return
    # conteos por nodo y edge
    events_df = events_df.copy()
    events_df["from_gid"] = events_df[fg_col]
    events_df["to_gid"] = events_df[tg_col]
    edge_counts = events_df.groupby(["from_gid", "to_gid"]).size()
    passes_given = events_df.groupby("from_gid").size()
    active_gids = {int(gid) for gid in teams.get("white", set())} | {int(gid) for gid in teams.get("black", set())}
    gid_nodes = _build_pass_map_nodes(gid_pos_map, teams, active_gids=active_gids)
    if not gid_nodes:
        print(f"[SKIP] {out_path}: no hay layout de jugadores")
        return

    def _team_color(team: str) -> str:
        if team == "black":
            return "#000000"
        if team == "white":
            return "#FFFFFF"
        return "#888888"

    renderer, pitch_canvas, off_x, off_y = _make_pitch_canvas(width=1200, height=700, padding=36)
    pitch_rgb = cv2.cvtColor(pitch_canvas, cv2.COLOR_BGR2RGB)

    def _to_canvas(pt: Tuple[float, float]) -> Tuple[int, int]:
        return _pitch_point_to_canvas(renderer, pt, off_x=off_x, off_y=off_y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pitch_rgb, zorder=0)
    # nodos
    for gid, pos in gid_nodes.items():
        team = _team_of_gid(gid, teams)
        size = 20 + 8 * float(passes_given.get(gid, 0))
        px, py = _to_canvas(pos)
        label_color = "black" if team == "white" else "white"
        ax.scatter(px, py, s=size, c=_team_color(team), edgecolors="k", zorder=3)
        ax.text(px + 14, py, f"G{gid}", fontsize=8, color=label_color, zorder=4, va="center")
    # edges
    if len(edge_counts):
        max_c = float(edge_counts.max()) if len(edge_counts) else 1.0
        for (fg, tg), c in edge_counts.items():
            if fg not in gid_nodes or tg not in gid_nodes:
                continue
            x1, y1 = _to_canvas(gid_nodes[fg])
            x2, y2 = _to_canvas(gid_nodes[tg])
            lw = 0.5 + 3.5 * (c / max(1.0, max_c))
            arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, linewidth=lw, color="#444444", alpha=0.45, zorder=2)
            ax.add_patch(arr)
    ax.set_title("Grafo de pases (grosor = cantidad)")
    ax.set_xlim(0, pitch_canvas.shape[1])
    ax.set_ylim(pitch_canvas.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out_path}")


def main() -> None:
    run_dir = getattr(config, "RUN_DIR", "runs/default")
    out_dir = os.path.join(run_dir, "debug_viz")
    teams = {"white": TEAM_WHITE_GIDS, "black": TEAM_BLACK_GIDS}
    meta = _load_meta_info(run_dir)
    fps = _load_meta(run_dir, meta)
    t0 = time.time()
    manual_gid_map, manual_segments = _load_manual_override_map(run_dir)
    manual_map_available = bool(manual_gid_map.get("cam1") or manual_gid_map.get("cam2") or manual_segments.get("cam1") or manual_segments.get("cam2"))
    gid_map = manual_gid_map if manual_map_available else _load_map(config.RUN_DIR)
    segs = manual_segments if manual_map_available else {"cam1": [], "cam2": []}

    df_exp1 = _load_tracks(os.path.join(run_dir, "c1_tracks.csv"))
    df_exp2 = _load_tracks(os.path.join(run_dir, "c2_tracks.csv"))
    df_st1 = _load_tracks(os.path.join(run_dir, "c1_tracks_stitched.csv"))
    df_st2 = _load_tracks(os.path.join(run_dir, "c2_tracks_stitched.csv"))
    df_ball_raw = _load_ball(os.path.join(run_dir, "ball.csv"))
    df_ball = _clean_ball(df_ball_raw)
    try:
        df_ball.to_csv(os.path.join(run_dir, "ball_clean.csv"), index=False)
        print(f"[WRITE] {os.path.join(run_dir, 'ball_clean.csv')}")
    except Exception as ball_write_exc:
        print(f"[WARN] No se pudo escribir ball_clean.csv: {ball_write_exc!r}")

    bounds = _get_bounds([df_exp1, df_exp2, df_st1, df_st2])

    gid_pos1 = _build_gid_positions(df_st1, "cam1", gid_map, segs)
    gid_pos2 = _build_gid_positions(df_st2, "cam2", gid_map, segs)
    gid_pos_map = _merge_gid_positions(gid_pos1, gid_pos2)
    gid_pos_map_spatial = _merge_gid_positions(gid_pos2, gid_pos1)

    pass_events = pd.DataFrame()
    try:
        pass_radius = float(getattr(config, "PASS_POS_RADIUS", getattr(config, "PASS_POS_RADIUS_IN", 600.0)))
        pass_hold = int(getattr(config, "PASS_POS_HOLD", 4))
        pass_dispute = int(getattr(config, "PASS_POS_DISPUTE_FRAMES", getattr(config, "PASS_POS_DISPUTE", 2)))
        allow_cross = bool(getattr(config, "PASS_ALLOW_CROSS_TEAM", False))
        poss_df = _compute_possession_frames(
            df_ball,
            gid_pos_map,
            radius=pass_radius,
            hold_frames=pass_hold,
            dispute_frames=pass_dispute,
            teams=teams,
        )
        try:
            os.makedirs(out_dir, exist_ok=True)
            poss_df.to_csv(os.path.join(out_dir, "pass_possession.csv"), index=False)
            print(f"[WRITE] {os.path.join(out_dir, 'pass_possession.csv')} ({len(poss_df)} rows)")
        except Exception as poss_exc:
            print(f"[WARN] No se pudo escribir pass_possession.csv: {poss_exc!r}")
        pass_events = _detect_passes_sequence(poss_df, allow_cross=allow_cross)
        if not pass_events.empty:
            pass_events["from_gid"] = pass_events["id_emisor"]
            pass_events["to_gid"] = pass_events["id_receptor"]
            pass_events["kick_frame"] = pass_events["Frame"]
            pass_events["recv_frame"] = pass_events["frame_end"]
        os.makedirs(out_dir, exist_ok=True)
        pass_events.to_csv(os.path.join(out_dir, "pass_events.csv"), index=False)
        print(f"[WRITE] {os.path.join(out_dir, 'pass_events.csv')} ({len(pass_events)} eventos) [mode=possession]")
    except Exception as pass_exc:
        print(f"[WARN] No se pudo detectar/escribir pases: {pass_exc!r}")

    try:
        zone_cols = int(getattr(config, "ZONE_CONTROL_COLS", 6))
        zone_rows = int(getattr(config, "ZONE_CONTROL_ROWS", 3))
        zone_control_df = _compute_zone_control(gid_pos_map_spatial, teams, cols=zone_cols, rows=zone_rows)
        if not zone_control_df.empty:
            os.makedirs(out_dir, exist_ok=True)
            zone_csv = os.path.join(out_dir, "zone_control_18zones.csv")
            zone_png = os.path.join(out_dir, "zone_control_18zones.png")
            zone_control_df.to_csv(zone_csv, index=False)
            print(f"[WRITE] {zone_csv} ({len(zone_control_df)} zonas)")
            _render_zone_control_report(zone_png, zone_control_df)
    except Exception as zone_exc:
        print(f"[WARN] No se pudo generar control territorial por zonas: {zone_exc!r}")

    metrics_rows: List[dict] = []
    video_targets = _selected_debug_videos()

    if "01" in video_targets:
        _render_stage(
            os.path.join(out_dir, "01_export_radar_all.mp4"),
            fps,
            bounds,
            df_exp1,
            df_exp2,
            use_map=False,
            ball_df=df_ball,
            mapped_only=False,
            draw_lines=False,
        )
    if "02" in video_targets:
        _render_stage(
            os.path.join(out_dir, "02_stitched_radar_all.mp4"),
            fps,
            bounds,
            df_st1,
            df_st2,
            use_map=False,
            ball_df=df_ball,
            mapped_only=False,
            draw_lines=False,
        )
    if "03" in video_targets:
        _render_stage(
            os.path.join(out_dir, "03_render_radar_mapped_only.mp4"),
            fps,
            bounds,
            df_st1,
            df_st2,
            use_map=True,
            gid_map=gid_map,
            segs=segs,
            ball_df=df_ball,
            mapped_only=True,
            draw_lines=True,
            metrics_rows=metrics_rows,
        )
    if "04" in video_targets:
        _render_stage(
            os.path.join(out_dir, "04_render_radar_all_with_gid.mp4"),
            fps,
            bounds,
            df_st1,
            df_st2,
            use_map=True,
            gid_map=gid_map,
            segs=segs,
            ball_df=df_ball,
            mapped_only=False,
            draw_lines=False,
        )
    if "05" in video_targets:
        _render_stage(
            os.path.join(out_dir, "05_crosscam_pair_error_lines.mp4"),
            fps,
            bounds,
            df_st1,
            df_st2,
            use_map=True,
            gid_map=gid_map,
            segs=segs,
            ball_df=df_ball,
            mapped_only=True,
            draw_lines=True,
        )

    video_cam1 = _resolve_path(run_dir, meta.get("video_cam1") if meta else None) or _resolve_path(
        run_dir, getattr(getattr(config, "PATHS", None), "video_cam1", None)
    )
    video_cam2 = _resolve_path(run_dir, meta.get("video_cam2") if meta else None) or _resolve_path(
        run_dir, getattr(getattr(config, "PATHS", None), "video_cam2", None)
    )
    if "06" in video_targets:
        _render_cam_video(
            os.path.join(out_dir, "06_cam1_original_ids.mp4"),
            video_cam1,
            df_st1,
            "cam1",
            gid_map,
            segs,
            df_ball,
            fps,
        )
    if "07" in video_targets:
        _render_cam_video(
            os.path.join(out_dir, "07_cam2_original_ids.mp4"),
            video_cam2,
            df_st2,
            "cam2",
            gid_map,
            segs,
            df_ball,
            fps,
        )
    if "08" in video_targets:
        _render_combined_video(
            os.path.join(out_dir, "08_combined_cams_gid.mp4"),
            video_cam1,
            video_cam2,
            df_st1,
            df_st2,
            gid_map,
            segs,
            df_ball,
            fps,
        )
    if "09" in video_targets:
        _render_combined_video(
            os.path.join(out_dir, "09_combined_teams_gid.mp4"),
            video_cam1,
            video_cam2,
            df_st1,
            df_st2,
            gid_map,
            segs,
            df_ball,
            fps,
            team_mode=True,
            team_sets=teams,
        )
    if "10" in video_targets:
        _render_voronoi_radar_video(
            os.path.join(out_dir, "10_voronoi_radar_teams.mp4"),
            gid_pos_map_spatial,
            df_ball,
            fps,
            team_sets=teams,
        )
    try:
        _render_pass_network(
            os.path.join(out_dir, "pass_network.png"),
            pass_events,
            gid_pos_map,
            bounds,
            team_sets=teams,
        )
        _render_pass_map_matplotlib(
            os.path.join(out_dir, "pass_map_count.png"),
            pass_events,
            gid_pos_map,
            teams,
        )
    except Exception as passnet_exc:
        print(f"[WARN] No se pudo renderizar pass network: {passnet_exc!r}")

    if metrics_rows:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "stage_metrics_by_frame.csv"), index=False)
        print(f"[WRITE] {os.path.join(out_dir, 'stage_metrics_by_frame.csv')}")
    # trayectorias de balón
    try:
        traj_dir = out_dir
        bounds_traj = bounds
        def _render_ball_traj(out_path: str, df: pd.DataFrame, cam_filter: Optional[str], connect: bool = True):
            df = df.copy()
            if cam_filter:
                df = df[df["cam_source"] == cam_filter]
            df = df[df["is_clean"]]
            df = df.dropna(subset=["ball_x", "ball_y"])
            if df.empty:
                print(f"[SKIP] traj {out_path}: no hay puntos")
                return
            df = df.sort_values("frame")
            w = int(getattr(config, "DEBUG_VIZ_WIDTH", 960))
            h = int(getattr(config, "DEBUG_VIZ_HEIGHT", 540))
            renderer, canvas, off_x, off_y = _make_pitch_canvas(width=w, height=h)
            pts_px = []
            for _, r in df.iterrows():
                px, py = _pitch_point_to_canvas(renderer, (float(r["ball_x"]), float(r["ball_y"])), off_x=off_x, off_y=off_y)
                pts_px.append((int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))))
            if connect and len(pts_px) >= 2:
                cv2.polylines(canvas, [np.array(pts_px, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
            for p in pts_px:
                cv2.circle(canvas, p, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, canvas)
            print(f"[WRITE] {out_path}")

        _render_ball_traj(os.path.join(traj_dir, "ball_traj_cam1.png"), df_ball, "cam1", connect=False)
        _render_ball_traj(os.path.join(traj_dir, "ball_traj_cam2.png"), df_ball, "cam2", connect=False)
        _render_ball_traj(os.path.join(traj_dir, "ball_traj_clean_path.png"), df_ball, None, connect=True)
    except Exception as traj_exc:
        print(f"[WARN] No se pudo renderizar trayectorias de balón: {traj_exc!r}")

    dt = time.time() - t0
    print(f"[DONE] debug viz -> {out_dir} | dur={_fmt_dur(dt)} ({dt:.2f}s)")


if __name__ == "__main__":
    main()

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

import config


@dataclass
class Tracklet:
    sid: int
    frames: np.ndarray
    pos: np.ndarray
    conf: np.ndarray
    area: np.ndarray
    t0: int
    t1: int
    mean_conf: float
    mean_area: float
    emb_mean: Optional[np.ndarray]
    emb_ratio: float
    reid_disabled: bool
    team: Optional[str]
    team_conf: float


def load_meta(run_dir: str) -> dict:
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tracks_stitched(run_dir: str, cam: str) -> pd.DataFrame:
    path = os.path.join(run_dir, f"{cam}_tracks_stitched.csv")
    fallback_path = os.path.join(run_dir, f"{cam}_tracks.csv")
    if not os.path.exists(path):
        if os.path.exists(fallback_path):
            print(f"[WARN] stitched tracks missing, usando fallback {fallback_path}")
            path = fallback_path
        else:
            raise FileNotFoundError(f"Missing tracks stitched file: {path}")
    df = pd.read_csv(path)
    required = {"frame", "stable_id", "pos_x", "pos_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {required}")
    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["stable_id"] = df["stable_id"].astype(int)
    n_rows = len(df)
    n_ids = df["stable_id"].nunique()
    frame_min = int(df["frame"].min()) if n_rows else -1
    frame_max = int(df["frame"].max()) if n_rows else -1
    nan_pos = df[["pos_x", "pos_y"]].isna().any(axis=1).mean() * 100.0 if n_rows else 0.0
    print(f"[LOAD] {cam}: rows={n_rows} ids={n_ids} frame_range=[{frame_min},{frame_max}] pos_nan={nan_pos:.2f}%")
    return df


def load_tracklet_summary_stitched(run_dir: str, cam: str) -> Optional[pd.DataFrame]:
    path = os.path.join(run_dir, f"{cam}_tracklet_summary_stitched.csv")
    fallback_path = os.path.join(run_dir, f"{cam}_tracklet_summary.csv")
    if not os.path.exists(path):
        if os.path.exists(fallback_path):
            print(f"[WARN] stitched summary missing, usando fallback {fallback_path}")
            path = fallback_path
        else:
            return None
    try:
        df = pd.read_csv(path)
        if "stitched_id" in df.columns:
            df["stable_id"] = df["stitched_id"].astype(int)
        elif "stable_id" in df.columns:
            df["stable_id"] = df["stable_id"].astype(int)
        return df
    except Exception as exc:
        print(f"[WARN] could not load summary {path}: {exc!r}")
        return None


def load_stitched_embeddings(run_dir: str, cam: str) -> Optional[Dict[int, np.ndarray]]:
    path = os.path.join(run_dir, f"{cam}_stitched_embeddings.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    ids = data.get("stitched_ids")
    embs = data.get("embeddings")
    if ids is None or embs is None:
        return None
    out = {}
    for sid, emb in zip(ids.tolist(), embs):
        if emb is None:
            continue
        out[int(sid)] = _l2norm(np.asarray(emb, dtype=np.float32))
    return out


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x.astype(np.float32) / (np.linalg.norm(x) + eps)


def load_map_stitch(run_dir: str, cam: str) -> dict:
    path = os.path.join(run_dir, f"{cam}_map_stitch.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_orig_embeddings(run_dir: str, cam: str) -> Dict[int, np.ndarray]:
    path = os.path.join(run_dir, f"{cam}_embeddings.npz")
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    ids = data.get("stable_ids")
    embs = data.get("embeddings")
    if ids is None or embs is None:
        return {}
    out = {}
    for sid, emb in zip(ids.tolist(), embs):
        if emb is None:
            continue
        out[int(sid)] = _l2norm(np.asarray(emb, dtype=np.float32))
    return out


def load_orig_summary(run_dir: str, cam: str) -> Dict[int, dict]:
    path = os.path.join(run_dir, f"{cam}_tracklet_summary.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        sid = int(row.get("stable_id", row.get("stable_id", -1)))
        out[sid] = {
            "n_obs": int(row.get("n_obs", 0)),
            "n_obs_emb": int(row.get("n_obs_emb", 0)),
        }
    return out


def _apply_transform(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if mat is None or pts is None or len(pts) == 0:
        return pts
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    out = pts_h @ mat.T
    return out.astype(np.float32)


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    # src -> dst
    if src.shape[0] < 2:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    var_src = np.sum(src_c ** 2)
    if var_src <= 1e-12:
        return None
    cov = src_c.T @ dst_c / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = np.trace(np.diag(S)) / var_src
    t = dst_mean - scale * (R @ src_mean)
    mat = np.zeros((2, 3), dtype=np.float32)
    mat[:, :2] = scale * R
    mat[:, 2] = t
    return mat


def _estimate_affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    if src.shape[0] < 3:
        return None
    A = np.hstack([src, np.ones((len(src), 1), dtype=np.float32)])
    try:
        X, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
    except Exception:
        return None
    mat = X.T  # 2x3
    return mat.astype(np.float32)


def _ransac_align(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    model: str = "similarity",
    iters: int = 1000,
    inlier_thr: float = 150.0,
    min_pairs: int = 2,
    min_inliers: int = 50,
) -> Tuple[Optional[np.ndarray], int, float]:
    n = len(src)
    if n < max(min_pairs, min_inliers, 1):
        return None, 0, math.nan
    rng = np.random.default_rng(12345)
    best_mat = None
    best_inliers = 0
    best_med = math.inf
    sample_size = 2 if model == "similarity" else 3
    for _ in range(int(iters)):
        if n < sample_size:
            break
        idx = rng.choice(n, size=sample_size, replace=False)
        src_s = src[idx]
        dst_s = dst[idx]
        if model == "affine":
            mat = _estimate_affine(src_s, dst_s)
        else:
            mat = _estimate_similarity(src_s, dst_s)
        if mat is None:
            continue
        pred = _apply_transform(mat, src)
        err = np.linalg.norm(dst - pred, axis=1)
        inliers = err <= inlier_thr
        n_in = int(inliers.sum())
        if n_in > best_inliers or (n_in == best_inliers and np.median(err[inliers]) < best_med if n_in > 0 else False):
            best_inliers = n_in
            best_mat = mat
            best_med = float(np.median(err[inliers])) if n_in > 0 else math.inf
            best_inlier_mask = inliers.copy()
    if best_mat is None:
        return None, 0, math.nan
    if best_inliers < min_inliers:
        return None, int(best_inliers), best_med
    # refinar con inliers
    src_in = src[best_inlier_mask]
    dst_in = dst[best_inlier_mask]
    if model == "affine":
        mat_ref = _estimate_affine(src_in, dst_in)
    else:
        mat_ref = _estimate_similarity(src_in, dst_in)
    if mat_ref is None:
        mat_ref = best_mat
    return mat_ref, int(best_inliers), float(best_med)

def reconstruct_stitched_embeddings(run_dir: str, cam: str) -> Optional[Dict[int, np.ndarray]]:
    # Plan B: usa map_stitch + embeddings originales
    stitch_map = load_map_stitch(run_dir, cam)
    orig_to_stitched = stitch_map.get("orig_to_stitched", {})
    if not orig_to_stitched:
        return None
    emb_orig = load_orig_embeddings(run_dir, cam)
    summary_orig = load_orig_summary(run_dir, cam)
    buckets: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    for orig_sid_str, stitched_list in orig_to_stitched.items():
        orig_sid = int(orig_sid_str)
        emb = emb_orig.get(orig_sid)
        if emb is None:
            continue
        stats = summary_orig.get(orig_sid, {})
        w = int(stats.get("n_obs_emb", stats.get("n_obs", 1))) or 1
        for stitched_sid in stitched_list:
            buckets.setdefault(int(stitched_sid), []).append((emb, w))
    if not buckets:
        return None
    out = {}
    for stitched_sid, pairs in buckets.items():
        total = float(sum(w for _, w in pairs))
        emb = sum((vec * float(w) for vec, w in pairs)) / max(total, 1e-6)
        out[int(stitched_sid)] = _l2norm(np.asarray(emb, dtype=np.float32))
    return out


def build_tracklets(
    df_tracks: pd.DataFrame,
    df_summary: Optional[pd.DataFrame],
    emb_dict: Optional[Dict[int, np.ndarray]],
    *,
    min_len: int,
    min_emb_ratio: float,
) -> Dict[int, Tracklet]:
    summary_lookup = {}
    if df_summary is not None and "stable_id" in df_summary.columns:
        for _, row in df_summary.iterrows():
            sid = int(row["stable_id"])
            n_total = int(row.get("n_obs_total", row.get("n_obs", 0)))
            n_emb = int(row.get("n_obs_emb_total", row.get("n_obs_emb", 0)))
            emb_ratio = float(row.get("emb_obs_ratio_total", row.get("emb_obs_ratio", 0.0)))
            if n_total > 0 and n_emb <= 0:
                emb_ratio = 0.0
            summary_lookup[sid] = dict(n_obs=n_total, n_obs_emb=n_emb, emb_ratio=emb_ratio)

    tracklets: Dict[int, Tracklet] = {}
    team_min_frames = int(getattr(config, "TEAM_MIN_FRAMES", 0))
    team_min_conf = float(getattr(config, "TEAM_MIN_CONF", 0.0))
    for sid, g in df_tracks.groupby("stable_id", sort=False):
        # Dedup por frame: queda la fila de mayor conf si existe, si no, la primera
        if "conf" in g.columns:
            g = g.sort_values(["frame", "conf"], ascending=[True, False], kind="mergesort")
        else:
            g = g.sort_values("frame", kind="mergesort")
        g = g.drop_duplicates(subset=["frame"], keep="first")
        frames = g["frame"].to_numpy(dtype=np.int64)
        pos = g[["pos_x", "pos_y"]].to_numpy(dtype=np.float32)
        conf_arr = g["conf"].to_numpy(dtype=np.float32) if "conf" in g.columns else np.ones(len(frames), dtype=np.float32)
        if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(g.columns):
            w = (g["bbox_x2"] - g["bbox_x1"]).clip(lower=0)
            h = (g["bbox_y2"] - g["bbox_y1"]).clip(lower=0)
            area_arr = (w * h).to_numpy(dtype=np.float32)
        else:
            area_arr = np.ones(len(frames), dtype=np.float32)
        if len(frames) < min_len:
            continue
        conf = float(g["conf"].mean()) if "conf" in g.columns else 1.0
        area = 0.0
        if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(g.columns):
            w = (g["bbox_x2"] - g["bbox_x1"]).clip(lower=0)
            h = (g["bbox_y2"] - g["bbox_y1"]).clip(lower=0)
            area = float((w * h).mean())

        emb_mean = emb_dict.get(int(sid)) if emb_dict else None
        emb_ratio = 0.0
        if sid in summary_lookup:
            emb_ratio = float(summary_lookup[sid].get("emb_ratio", 0.0))
            if emb_ratio <= 0.0:
                n_total = summary_lookup[sid].get("n_obs", 0)
                n_emb = summary_lookup[sid].get("n_obs_emb", 0)
                emb_ratio = float(n_emb) / float(n_total) if n_total > 0 else 0.0
        elif emb_mean is not None:
            emb_ratio = 1.0

        reid_disabled = emb_mean is None or emb_ratio < min_emb_ratio

        # Team por tracklet (mayoría)
        team_val = None
        team_conf = 0.0
        team_col = None
        for cand in ("team", "team_id"):
            if cand in g.columns:
                team_col = cand
                break
        if team_col is not None:
            series_team = g[team_col].dropna()
            if len(series_team) > 0:
                counts = series_team.value_counts()
                top_team = counts.idxmax()
                top_cnt = int(counts.max())
                total_team = int(counts.sum())
                team_conf = float(top_cnt) / float(total_team) if total_team > 0 else 0.0
                if total_team >= max(1, team_min_frames) and team_conf >= team_min_conf:
                    team_val = str(top_team)
                else:
                    team_val = None

        tracklets[int(sid)] = Tracklet(
            sid=int(sid),
            frames=frames,
            pos=pos,
            conf=conf_arr,
            area=area_arr,
            t0=int(frames[0]),
            t1=int(frames[-1]),
            mean_conf=float(conf),
            mean_area=float(area),
            emb_mean=(None if emb_mean is None else _l2norm(np.asarray(emb_mean, dtype=np.float32))),
            emb_ratio=float(emb_ratio),
            reid_disabled=bool(reid_disabled),
            team=team_val,
            team_conf=float(team_conf),
        )
    print(f"[BUILD] tracklets={len(tracklets)} (after len>={min_len})")
    return tracklets


def compute_overlap_frames(a: Tracklet, b: Tracklet) -> int:
    return max(0, min(a.t1, b.t1) - max(a.t0, b.t0) + 1)


def candidate_pairs(
    t1: Dict[int, Tracklet],
    t2: Dict[int, Tracklet],
    *,
    min_overlap: int,
    window: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, int]]:
    pairs = []
    for sid1, a in t1.items():
        for sid2, b in t2.items():
            if window is not None:
                t0, t1_w = window
                # solape con ventana
                if max(a.t0, b.t0, t0) > min(a.t1, b.t1, t1_w - 1):
                    continue
            if compute_overlap_frames(a, b) >= min_overlap:
                pairs.append((sid1, sid2))
    return pairs


def _get_positions(
    a: Tracklet,
    b: Tracklet,
    *,
    sample_step: int,
    min_overlap: int,
    align_mat: Optional[np.ndarray] = None,
    window: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    fa = a.frames
    fb = b.frames
    common = np.intersect1d(fa, fb, assume_unique=False)
    if window is not None:
        t0, t1 = window
        common = common[(common >= t0) & (common < t1)]
    if len(common) == 0:
        return None, None, 0, 0
    common = common[:: max(1, sample_step)]
    if len(common) < max(1, min_overlap // max(1, sample_step)):
        return None, None, len(common), 0

    idx_a = np.searchsorted(fa, common)
    idx_b = np.searchsorted(fb, common)
    pos_a = a.pos[idx_a]
    pos_b = b.pos[idx_b]
    conf_a = a.conf[idx_a] if a.conf is not None else np.ones(len(common), dtype=np.float32)
    conf_b = b.conf[idx_b] if b.conf is not None else np.ones(len(common), dtype=np.float32)
    area_a = a.area[idx_a] if a.area is not None else np.ones(len(common), dtype=np.float32)
    area_b = b.area[idx_b] if b.area is not None else np.ones(len(common), dtype=np.float32)

    qc_conf_min = float(getattr(config, "QC_CONF_MIN", 0.0))
    qc_area_min = float(getattr(config, "QC_AREA_MIN", 0.0))
    qc_use_area = bool(getattr(config, "QC_USE_AREA", False))
    qc_min_good_frames = int(getattr(config, "QC_MIN_GOOD_FRAMES", 0))
    qc_min_good_pct = float(getattr(config, "QC_MIN_GOOD_PCT", 0.0))

    good_mask = (conf_a >= qc_conf_min) & (conf_b >= qc_conf_min)
    if qc_use_area:
        good_mask &= (area_a >= qc_area_min) & (area_b >= qc_area_min)

    good_frames = common[good_mask]
    n_common = len(common)
    n_good = len(good_frames)
    if n_good == 0:
        return None, None, n_common, n_good
    if qc_min_good_frames > 0 and n_good < qc_min_good_frames:
        return None, None, n_common, n_good
    if qc_min_good_pct > 0:
        pct_good = n_good / float(max(1, n_common))
        if pct_good < qc_min_good_pct:
            return None, None, n_common, n_good

    idx_a = np.searchsorted(fa, good_frames)
    idx_b = np.searchsorted(fb, good_frames)
    pos_a = a.pos[idx_a]
    pos_b = b.pos[idx_b]
    if align_mat is not None:
        pos_b = _apply_transform(align_mat, pos_b)
    return pos_a, pos_b, n_common, n_good


def _endpoint_position(
    t: Tracklet,
    *,
    start: bool,
    n_frames: int,
    align_mat: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    frames = t.frames
    if len(frames) == 0:
        return None
    if start:
        idx = np.arange(min(len(frames), n_frames))
    else:
        idx = np.arange(max(0, len(frames) - n_frames), len(frames))
    pos = t.pos[idx]
    conf = t.conf[idx] if t.conf is not None else np.ones(len(idx), dtype=np.float32)
    area = t.area[idx] if t.area is not None else np.ones(len(idx), dtype=np.float32)

    qc_conf_min = float(getattr(config, "QC_CONF_MIN", 0.0))
    qc_area_min = float(getattr(config, "QC_AREA_MIN", 0.0))
    qc_use_area = bool(getattr(config, "QC_USE_AREA", False))

    mask = conf >= qc_conf_min
    if qc_use_area:
        mask &= area >= qc_area_min
    if not mask.any():
        return None
    pos = pos[mask]
    if align_mat is not None:
        pos = _apply_transform(align_mat, pos)
    return pos.mean(axis=0)


def pair_cost(
    a: Tracklet,
    b: Tracklet,
    *,
    sample_step: int,
    pos_thr: float,
    reid_thr: float,
    min_overlap: int,
    min_emb_ratio: float,
    w_team: float = 0.0,
    align_mat: Optional[np.ndarray] = None,
    window: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float, Optional[float], int, bool, float, int, float, float]:
    d_emb: Optional[float] = None
    pos_a, pos_b, n_common, n_good = _get_positions(
        a,
        b,
        sample_step=sample_step,
        min_overlap=min_overlap,
        align_mat=align_mat,
        window=window,
    )
    if pos_a is None or pos_b is None:
        return math.inf, math.inf, math.inf, None, n_common, False, 0.0, 0, math.inf, math.inf

    d = np.linalg.norm(pos_a - pos_b, axis=1)
    if len(d) == 0:
        return math.inf, math.inf, math.inf, None, n_common, False, 0.0, 0, math.inf, math.inf

    inlier_thr = float(getattr(config, "CROSSCAM_INLIER_THR", pos_thr))
    min_inlier_pct = float(getattr(config, "CROSSCAM_MIN_INLIER_PCT", 0.0))
    min_inlier_frames = int(getattr(config, "CROSSCAM_MIN_INLIER_FRAMES", 0))
    inliers = d <= inlier_thr
    inlier_count = int(inliers.sum())
    inlier_pct = float(inliers.mean()) if len(inliers) else 0.0

    d_pos_med = float(np.median(d))
    d_pos_p80 = float(np.percentile(d, 80))
    if d_pos_med > pos_thr or d_pos_p80 > (1.8 * pos_thr):
        return math.inf, d_pos_med, d_pos_p80, None, n_common, False, inlier_pct, inlier_count, float(d.mean()), float(d.max())

    # Team gating/penalty
    team_mode = str(getattr(config, "TEAM_MODE", "penalty")).strip().lower()
    team_unknown_policy = str(getattr(config, "TEAM_UNKNOWN_POLICY", "allow")).strip().lower()
    team_penalty_val = float(getattr(config, "TEAM_PENALTY", 1.0))
    penalty_team = 0.0
    if team_mode in ("gate", "penalty"):
        ta = a.team
        tb = b.team
        if ta is not None and tb is not None:
            if ta != tb:
                if team_mode == "gate":
                    return math.inf, d_pos_med, d_pos_p80, None, n_common, False, inlier_pct, inlier_count, float(d.mean()), float(d.max())
                else:
                    penalty_team = team_penalty_val
        else:
            # alguno unknown
            if team_unknown_policy == "block":
                return math.inf, d_pos_med, d_pos_p80, None, n_common, False, inlier_pct, inlier_count, float(d.mean()), float(d.max())

    # Gating por inliers (robusto)
    if inlier_thr > 0:
        if inlier_pct < min_inlier_pct:
            return math.inf, d_pos_med, d_pos_p80, None, n_common, False, inlier_pct, inlier_count, float(d.mean()), float(d.max())
        if min_inlier_frames > 0 and inlier_count < min_inlier_frames:
            return math.inf, d_pos_med, d_pos_p80, None, n_common, False, inlier_pct, inlier_count, float(d.mean()), float(d.max())

    d_emb = None
    if not a.reid_disabled and not b.reid_disabled and a.emb_mean is not None and b.emb_mean is not None:
        d_emb = float(1.0 - float(np.dot(a.emb_mean, b.emb_mean)))
        if d_emb > reid_thr:
            return math.inf, d_pos_med, d_pos_p80, d_emb, n_common, False, inlier_pct, inlier_count, float(d.mean()), float(d.max())

    norm_pos = d_pos_med / pos_thr
    norm_emb = (d_emb / reid_thr) if d_emb is not None else 0.0
    cost = float(config.CROSSCAM_W_POS) * norm_pos + float(config.CROSSCAM_W_REID) * norm_emb + w_team * penalty_team
    return cost, d_pos_med, d_pos_p80, d_emb, n_common, True, inlier_pct, inlier_count, float(d.mean()), float(d.max())


def solve_assignment(
    ids1: List[int],
    ids2: List[int],
    cost_lookup: Dict[Tuple[int, int], float],
    inf_cost: float = 1e9,
) -> List[Tuple[int, int, float]]:
    n1 = len(ids1)
    n2 = len(ids2)
    if n1 == 0 or n2 == 0:
        return []
    if linear_sum_assignment is None:
        # Greedy fallback
        pairs = [
            (i, j, c) for (i, j), c in cost_lookup.items() if math.isfinite(c) and c < inf_cost / 2.0
        ]
        pairs.sort(key=lambda x: x[2])
        used1 = set()
        used2 = set()
        out = []
        for sid1, sid2, c in pairs:
            if sid1 in used1 or sid2 in used2:
                continue
            used1.add(sid1)
            used2.add(sid2)
            out.append((sid1, sid2, c))
        return out

    cost_matrix = np.full((n1, n2), inf_cost, dtype=np.float32)
    for i, sid1 in enumerate(ids1):
        for j, sid2 in enumerate(ids2):
            c = cost_lookup.get((sid1, sid2), inf_cost)
            cost_matrix[i, j] = c
    finite_mask = np.isfinite(cost_matrix)
    row_keep = finite_mask.any(axis=1)
    col_keep = finite_mask.any(axis=0)
    if not row_keep.any() or not col_keep.any():
        return []
    cost_matrix_sub = cost_matrix[np.ix_(row_keep, col_keep)]
    ids1_sub = [sid for sid, keep in zip(ids1, row_keep) if keep]
    ids2_sub = [sid for sid, keep in zip(ids2, col_keep) if keep]
    # reemplaza inf por gran número para evitar ValueError
    cost_matrix_sub = np.where(np.isfinite(cost_matrix_sub), cost_matrix_sub, inf_cost)
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix_sub)
    except ValueError:
        # fallback a greedy si scipy considera la matriz infactible
        pairs = [
            (i, j, cost_matrix_sub[i, j])
            for i in range(cost_matrix_sub.shape[0])
            for j in range(cost_matrix_sub.shape[1])
            if math.isfinite(cost_matrix_sub[i, j]) and cost_matrix_sub[i, j] < inf_cost / 2.0
        ]
        pairs.sort(key=lambda x: x[2])
        used1 = set()
        used2 = set()
        out = []
        for i, j, c in pairs:
            if i in used1 or j in used2:
                continue
            used1.add(i)
            used2.add(j)
            out.append((ids1_sub[i], ids2_sub[j], float(c)))
        return out
    matches = []
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        cost = float(cost_matrix_sub[r, c])
        if not math.isfinite(cost) or cost >= inf_cost / 2.0:
            continue
        matches.append((ids1_sub[r], ids2_sub[c], cost))
    return matches


def accept_matches(
    matches: List[Tuple[int, int, float]],
    *,
    accept_margin: float,
    cost_lookup: Dict[Tuple[int, int], float],
    pair_lookup: Dict[Tuple[int, int], dict],
    pos_accept_thr: Optional[float] = None,
    max_accept_cost: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    # Precompute best/second per sid1 y sid2 sobre TODOS los candidatos finitos
    best1: Dict[int, Tuple[float, float]] = {}  # sid1 -> (best, second)
    best2: Dict[int, Tuple[float, float]] = {}  # sid2 -> (best, second)

    def _update_best(d: Dict[int, Tuple[float, float]], key: int, val: float):
        if not math.isfinite(val):
            return
        if key not in d:
            d[key] = (val, math.inf)
            return
        b, s = d[key]
        if val < b:
            d[key] = (val, b)
        elif val < s:
            d[key] = (b, val)

    for (sid1, sid2), c in cost_lookup.items():
        _update_best(best1, sid1, c)
        _update_best(best2, sid2, c)

    accepted = []
    for sid1, sid2, cost in matches:
        if not math.isfinite(cost):
            continue
        if max_accept_cost is not None and math.isfinite(max_accept_cost) and cost > max_accept_cost:
            continue
        row = pair_lookup.get((sid1, sid2))
        d_pos_med = row["d_pos_med"] if row is not None else math.inf
        if pos_accept_thr is not None and math.isfinite(pos_accept_thr) and d_pos_med > pos_accept_thr:
            continue
        b1, s1 = best1.get(sid1, (math.inf, math.inf))
        b2, s2 = best2.get(sid2, (math.inf, math.inf))
        margin1 = s1 - cost if math.isfinite(s1) else math.inf
        margin2 = s2 - cost if math.isfinite(s2) else math.inf
        if margin1 >= accept_margin and margin2 >= accept_margin:
            accepted.append((sid1, sid2, cost))
    return accepted


def build_gid_map(
    t1: Dict[int, Tracklet],
    t2: Dict[int, Tracklet],
    accepted: List[Tuple[int, int, float]],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    gid1: Dict[int, int] = {}
    gid2: Dict[int, int] = {}
    next_gid = 1
    for sid1, sid2, _ in accepted:
        g1 = gid1.get(sid1)
        g2 = gid2.get(sid2)
        if g1 is None and g2 is None:
            gid = next_gid
            next_gid += 1
            gid1[sid1] = gid
            gid2[sid2] = gid
        elif g1 is not None and g2 is None:
            gid2[sid2] = g1
        elif g1 is None and g2 is not None:
            gid1[sid1] = g2
        elif g1 != g2:
            # conflicto, asigna nuevo a sid2 para no mezclar
            gid2[sid2] = next_gid
            next_gid += 1
    # Unmatched -> gid nuevo
    for sid1 in t1.keys():
        if sid1 not in gid1:
            gid1[sid1] = next_gid
            next_gid += 1
    for sid2 in t2.keys():
        if sid2 not in gid2:
            gid2[sid2] = next_gid
            next_gid += 1
    return gid1, gid2


def write_crosscam_map(run_dir: str, gid_map_cam1: Dict[int, int], gid_map_cam2: Dict[int, int]) -> str:
    out_path_cfg = getattr(config, "OFFLINE_MAP_PATH", "crosscam_map.json")
    run_dir_norm = os.path.normpath(run_dir)
    if os.path.isabs(out_path_cfg):
        out_path = out_path_cfg
    else:
        candidate = os.path.normpath(out_path_cfg)
        if candidate.startswith(run_dir_norm):
            out_path = candidate
        else:
            out_path = os.path.join(run_dir_norm, out_path_cfg)
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {
        "cam1": {str(k): int(v) for k, v in gid_map_cam1.items()},
        "cam2": {str(k): int(v) for k, v in gid_map_cam2.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] {out_path}")
    return out_path


def write_report_crosscam(
    run_dir: str,
    *,
    summary: dict,
    pair_rows: List[dict],
) -> Tuple[str, str]:
    path = os.path.join(run_dir, "report_crosscam.csv")
    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(path, index=False)
    print(f"[WRITE] {path} ({len(df_pairs)} pairs)")
    # resumen aparte
    summary_path = os.path.join(run_dir, "report_crosscam_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[WRITE] {summary_path}")
    return path, summary_path


def write_global_tracks(
    run_dir: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    gid1: Dict[int, int],
    gid2: Dict[int, int],
    align_mat: Optional[np.ndarray],
) -> Optional[Tuple[str, Optional[str]]]:
    if not config.CROSSCAM_WRITE_GLOBAL_TRACKS:
        return None

    def _prep(df: pd.DataFrame, cam_label: str, gid_map: Dict[int, int]) -> pd.DataFrame:
        df = df.copy()
        df["gid"] = df["stable_id"].map(gid_map)
        df["source_cam"] = cam_label
        return df

    df1_out = _prep(df1, "cam1", gid1)
    df2_out = _prep(df2, "cam2", gid2)

    # percentiles de área por cam
    def _area_percentiles(df: pd.DataFrame) -> Tuple[float, float]:
        if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(df.columns):
            area = (df["bbox_x2"] - df["bbox_x1"]).clip(lower=0) * (df["bbox_y2"] - df["bbox_y1"]).clip(lower=0)
            if len(area) > 0:
                a10 = float(np.percentile(area, 10))
                a90 = float(np.percentile(area, 90))
                return a10, a90
        return 0.0, 1.0

    a10_c1, a90_c1 = _area_percentiles(df1_out)
    a10_c2, a90_c2 = _area_percentiles(df2_out)

    # opcional: aplicar alineación a posiciones cam2 para scoring/fusión
    use_align = bool(getattr(config, "FUSE_USE_ALIGN_FOR_CAM2", True))
    if align_mat is None or not use_align:
        def _align_pos(pos): return pos
    else:
        def _align_pos(pos):
            p = np.asarray(pos, dtype=np.float32)
            if p.ndim == 1:
                p = p.reshape(1, -1)
            out = _apply_transform(align_mat, p)
            if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[0] == 1:
                return out[0]
            return out

    # index por frame
    def _index_by_frame(df: pd.DataFrame) -> Dict[int, List[dict]]:
        out: Dict[int, List[dict]] = {}
        for f, g in df.groupby("frame"):
            out[int(f)] = g.to_dict("records")
        return out

    by_frame_c1 = _index_by_frame(df1_out)
    by_frame_c2 = _index_by_frame(df2_out)
    all_frames = sorted(set(by_frame_c1.keys()) | set(by_frame_c2.keys()))

    w_conf = float(getattr(config, "FUSE_W_CONF", 0.4))
    w_area = float(getattr(config, "FUSE_W_AREA", 0.4))
    w_border = float(getattr(config, "FUSE_W_BORDER", 0.2))
    conf_min = float(getattr(config, "FUSE_CONF_MIN", 0.25))
    border_min = float(getattr(config, "FUSE_BORDER_MIN_PX", 0.0))
    switch_margin = float(getattr(config, "FUSE_SWITCH_MARGIN", 0.15))
    min_hold = int(getattr(config, "FUSE_SWITCH_MIN_HOLD_FRAMES", 8))
    max_pred_gap = int(getattr(config, "FUSE_MAX_PRED_GAP_FRAMES", 24))
    max_speed = float(getattr(config, "FUSE_MAX_SPEED", 500.0))

    def _score(obs: dict, cam_label: str) -> float:
        conf = float(obs.get("conf", 1.0))
        conf_n = np.clip((conf - conf_min) / (1.0 - conf_min + 1e-6), 0.0, 1.0)
        if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(obs):
            area = max(0.0, (obs["bbox_x2"] - obs["bbox_x1"]) * (obs["bbox_y2"] - obs["bbox_y1"]))
        else:
            area = 1.0
        if cam_label == "cam1":
            a10, a90 = a10_c1, a90_c1
        else:
            a10, a90 = a10_c2, a90_c2
        if a90 <= a10:
            area_n = 1.0
        else:
            area_n = np.clip((area - a10) / (a90 - a10 + 1e-6), 0.0, 1.0)
        border_n = 1.0
        if border_min > 0 and {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(obs):
            border = min(obs["bbox_x1"], obs["bbox_y1"], max(0.0, obs["bbox_x2"]), max(0.0, obs["bbox_y2"]))
            border_n = np.clip(border / border_min, 0.0, 1.0)
        return float(w_conf * conf_n + w_area * area_n + w_border * border_n)

    # estado por gid
    state = {}
    rows_out = []
    # recolectar todos los gids
    gids = sorted(set(df1_out["gid"].dropna().astype(int).tolist()) | set(df2_out["gid"].dropna().astype(int).tolist()))

    for f in all_frames:
        c1 = by_frame_c1.get(f, [])
        c2 = by_frame_c2.get(f, [])
        # índice por gid en frame
        idx1 = {}
        for r in c1:
            gid = r.get("gid")
            if pd.isna(gid):
                continue
            gid = int(gid)
            prev = idx1.get(gid)
            if prev is None or float(r.get("conf", -1)) > float(prev.get("conf", -1)):
                idx1[gid] = r
        idx2 = {}
        for r in c2:
            gid = r.get("gid")
            if pd.isna(gid):
                continue
            gid = int(gid)
            prev = idx2.get(gid)
            if prev is None or float(r.get("conf", -1)) > float(prev.get("conf", -1)):
                idx2[gid] = r
        frame_gids = sorted(set(idx1.keys()) | set(idx2.keys()) | set(state.keys()))

        for gid in frame_gids:
            s = state.get(gid, {"last_source": None, "hold": 0, "last_pos": None, "last_frame": None, "prev_pos": None, "prev_frame": None})
            obs1 = idx1.get(gid)
            obs2 = idx2.get(gid)
            pos1 = None
            pos2 = None
            if obs1 is not None:
                pos1 = np.array([float(obs1["pos_x"]), float(obs1["pos_y"])], dtype=np.float32)
            if obs2 is not None:
                pos2_raw = np.array([float(obs2["pos_x"]), float(obs2["pos_y"])], dtype=np.float32)
                pos2 = _align_pos(pos2_raw)

            choice = None
            q1 = q2 = None
            if obs1 is not None:
                q1 = _score(obs1, "cam1")
            if obs2 is not None:
                q2 = _score(obs2, "cam2")

            # anti-teleport helper
            def _speed_ok(pos):
                if s["last_pos"] is None or s["last_frame"] is None:
                    return True
                dt = max(1, f - s["last_frame"])
                dist = np.linalg.norm(pos - s["last_pos"])
                return (dist / dt) <= max_speed

            # selección con histeresis
            if obs1 is not None and obs2 is not None:
                if s["last_source"] == "cam1" and s["hold"] < min_hold and _speed_ok(pos1):
                    choice = ("cam1", obs1, pos1, q1, q2)
                elif s["last_source"] == "cam2" and s["hold"] < min_hold and _speed_ok(pos2):
                    choice = ("cam2", obs2, pos2, q1, q2)
                else:
                    # comparar scores
                    if q1 is None:
                        pick = "cam2"
                    elif q2 is None:
                        pick = "cam1"
                    else:
                        if s["last_source"] == "cam1" and q1 + switch_margin >= q2 and _speed_ok(pos1):
                            pick = "cam1"
                        elif s["last_source"] == "cam2" and q2 + switch_margin >= q1 and _speed_ok(pos2):
                            pick = "cam2"
                        else:
                            pick = "cam1" if q1 >= q2 else "cam2"
                    if pick == "cam1" and _speed_ok(pos1):
                        choice = ("cam1", obs1, pos1, q1, q2)
                    elif pick == "cam2" and _speed_ok(pos2):
                        choice = ("cam2", obs2, pos2, q1, q2)
                    elif _speed_ok(pos1):
                        choice = ("cam1", obs1, pos1, q1, q2)
                    elif _speed_ok(pos2):
                        choice = ("cam2", obs2, pos2, q1, q2)
            elif obs1 is not None and _speed_ok(pos1):
                choice = ("cam1", obs1, pos1, q1, q2)
            elif obs2 is not None and _speed_ok(pos2):
                choice = ("cam2", obs2, pos2, q1, q2)

            is_pred = 0
            q_used = None
            source_cam = None
            pos_fused = None

            if choice is not None:
                source_cam, obs_chosen, pos_fused, q_used, q_other = choice
                q_used = q_used if q_used is not None else 0.0
                # actualizar estado
                if s["last_source"] == source_cam:
                    s["hold"] += 1
                else:
                    s["hold"] = 1
                s["prev_pos"] = s["last_pos"]
                s["prev_frame"] = s["last_frame"]
                s["last_pos"] = pos_fused
                s["last_frame"] = f
                s["last_source"] = source_cam
            else:
                # predicción si gap pequeño
                if s["last_frame"] is not None:
                    gap = f - s["last_frame"]
                    if gap > 0 and gap <= max_pred_gap and s["last_pos"] is not None:
                        if s["prev_pos"] is not None and s["prev_frame"] is not None:
                            dt = max(1, s["last_frame"] - s["prev_frame"])
                            v = (s["last_pos"] - s["prev_pos"]) / float(dt)
                        else:
                            v = np.zeros(2, dtype=np.float32)
                        pos_fused = s["last_pos"] + v * float(gap)
                        source_cam = "pred"
                        q_used = 0.0
                        is_pred = 1
                        s["hold"] += 1
                        s["prev_pos"] = s["last_pos"]
                        s["prev_frame"] = s["last_frame"]
                        s["last_pos"] = pos_fused
                        s["last_frame"] = f
                        s["last_source"] = "pred"
                # si no se pudo predecir, no escribimos fila (sin obs ni pred)
            if pos_fused is not None:
                rows_out.append(
                    dict(
                        frame=int(f),
                        gid=int(gid),
                        pos_x=float(pos_fused[0]),
                        pos_y=float(pos_fused[1]),
                        source_cam=source_cam,
                        q=float(q_used if q_used is not None else 0.0),
                        is_pred=int(is_pred),
                        q1=(float(q1) if q1 is not None else None),
                        q2=(float(q2) if q2 is not None else None),
                    )
                )
            state[gid] = s

    df_global = pd.DataFrame(rows_out)
    out_path = os.path.join(run_dir, "global_tracks.csv")
    df_global.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path} rows={len(df_global)}")

    report_path = None
    if getattr(config, "FUSE_WRITE_FUSE_REPORT", True):
        # resumen sencillo
        gid_stats = []
        switches_total = 0
        for gid, g in df_global.groupby("gid"):
            g_sorted = g.sort_values("frame")
            sources = g_sorted["source_cam"].tolist()
            switches = sum(1 for i in range(1, len(sources)) if sources[i] != sources[i - 1])
            switches_total += switches
            gid_stats.append(
                dict(
                    gid=int(gid),
                    frames=len(g_sorted),
                    pred_frames=int((g_sorted["is_pred"] == 1).sum()),
                    switches=int(switches),
                    avg_q=float(g_sorted["q"].mean()) if not g_sorted.empty else math.nan,
                )
            )
        report = dict(
            n_gids=len(gid_stats),
            n_rows=len(df_global),
            pred_frames=int((df_global["is_pred"] == 1).sum()) if not df_global.empty else 0,
            switches_total=int(switches_total),
        )
        report_path = os.path.join(run_dir, "report_fuse_summary.csv")
        pd.DataFrame([report]).to_csv(report_path, index=False)
        gid_report_path = os.path.join(run_dir, "report_fuse_per_gid.csv")
        pd.DataFrame(gid_stats).to_csv(gid_report_path, index=False)
        print(f"[WRITE] {report_path}")
        print(f"[WRITE] {gid_report_path}")

    return out_path, report_path


def main() -> None:
    run_dir = getattr(config, "RUN_DIR", "runs/default")
    print(f"[RUN] offline_crosscam | run_dir={run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    meta = load_meta(run_dir)
    fps = float(meta.get("fps", 30.0)) if isinstance(meta, dict) else 30.0
    print(f"[META] fps={fps}")

    df1 = load_tracks_stitched(run_dir, "c1")
    df2 = load_tracks_stitched(run_dir, "c2")
    if getattr(config, "CLIP_ENABLE", False):
        start = int(getattr(config, "CLIP_START_FRAME", 0))
        end = int(getattr(config, "CLIP_END_FRAME", 0))
        df1 = df1[df1["frame"].between(start, end if end > 0 else df1["frame"].max())].copy()
        df2 = df2[df2["frame"].between(start, end if end > 0 else df2["frame"].max())].copy()
    sum1 = load_tracklet_summary_stitched(run_dir, "c1")
    sum2 = load_tracklet_summary_stitched(run_dir, "c2")

    emb1 = load_stitched_embeddings(run_dir, "c1")
    emb2 = load_stitched_embeddings(run_dir, "c2")
    if emb1 is None:
        emb1 = reconstruct_stitched_embeddings(run_dir, "c1")
        print(f"[EMB] c1 reconstructed={emb1 is not None}")
    if emb2 is None:
        emb2 = reconstruct_stitched_embeddings(run_dir, "c2")
        print(f"[EMB] c2 reconstructed={emb2 is not None}")

    t1 = build_tracklets(
        df1,
        sum1,
        emb1,
        min_len=int(getattr(config, "CROSSCAM_MIN_TRACKLET_LEN_FRAMES", 48)),
        min_emb_ratio=float(getattr(config, "CROSSCAM_MIN_EMB_RATIO", 0.25)),
    )
    t2 = build_tracklets(
        df2,
        sum2,
        emb2,
        min_len=int(getattr(config, "CROSSCAM_MIN_TRACKLET_LEN_FRAMES", 48)),
        min_emb_ratio=float(getattr(config, "CROSSCAM_MIN_EMB_RATIO", 0.25)),
    )

    min_overlap = int(getattr(config, "CROSSCAM_MIN_OVERLAP_FRAMES", 24))
    sample_step = int(getattr(config, "CROSSCAM_SAMPLE_STEP", 2))
    pos_thr = float(getattr(config, "CROSSCAM_POS_THR", 150.0))
    reid_thr = float(getattr(config, "CROSSCAM_REID_THR", 0.35))
    min_emb_ratio = float(getattr(config, "CROSSCAM_MIN_EMB_RATIO", 0.25))
    accept_margin = float(getattr(config, "CROSSCAM_ACCEPT_MARGIN", 0.15))
    w_team = float(getattr(config, "CROSSCAM_W_TEAM", 0.0))

    def run_matching(
        align_mat: Optional[np.ndarray],
        window: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]], List[dict], Dict[Tuple[int, int], float]]:
        pairs = candidate_pairs(t1, t2, min_overlap=min_overlap, window=window)
        print(f"[PAIRS] candidates={len(pairs)} | align={'on' if align_mat is not None else 'off'}")
        cost_lookup: Dict[Tuple[int, int], float] = {}
        pair_rows: List[dict] = []
        for sid1, sid2 in pairs:
            c, d_med, d_p80, d_emb, overlap_used, ok, inlier_pct, inlier_count, dist_mean, dist_max = pair_cost(
                t1[sid1],
                t2[sid2],
                sample_step=sample_step,
                pos_thr=pos_thr,
                reid_thr=reid_thr,
                min_overlap=min_overlap,
                min_emb_ratio=min_emb_ratio,
                w_team=w_team,
                align_mat=align_mat,
                window=window,
            )
            if ok:
                cost_lookup[(sid1, sid2)] = c
            else:
                cost_lookup[(sid1, sid2)] = math.inf
            pair_rows.append(
                dict(
                    cam1_sid=int(sid1),
                    cam2_sid=int(sid2),
                    cost=float(cost_lookup[(sid1, sid2)]) if math.isfinite(cost_lookup[(sid1, sid2)]) else math.inf,
                    d_pos_med=float(d_med),
                    d_pos_p80=float(d_p80),
                    d_emb=(float(d_emb) if d_emb is not None else None),
                    overlap_frames=int(overlap_used),
                    ok=bool(ok),
                    inlier_pct=float(inlier_pct),
                    inlier_count=int(inlier_count),
                    dist_mean=float(dist_mean),
                    dist_max=float(dist_max),
                    team_a=t1[sid1].team,
                    team_b=t2[sid2].team,
                    team_match=(
                        None
                        if (t1[sid1].team is None or t2[sid2].team is None)
                        else bool(t1[sid1].team == t2[sid2].team)
                    ),
                )
            )
        matches = solve_assignment(list(t1.keys()), list(t2.keys()), cost_lookup)
        print(f"[MATCH] raw={len(matches)} | align={'on' if align_mat is not None else 'off'}")
        pair_lookup = {(p["cam1_sid"], p["cam2_sid"]): p for p in pair_rows}
        pos_accept_thr = getattr(config, "CROSSCAM_POS_ACCEPT_THR", None)
        max_accept_cost = getattr(config, "CROSSCAM_MAX_ACCEPT_COST", None)
        accepted = accept_matches(
            matches,
            accept_margin=accept_margin,
            cost_lookup=cost_lookup,
            pair_lookup=pair_lookup,
            pos_accept_thr=pos_accept_thr,
            max_accept_cost=max_accept_cost,
        )
        print(f"[MATCH] accepted={len(accepted)} | align={'on' if align_mat is not None else 'off'}")
        return accepted, matches, pair_rows, cost_lookup

    # Pasada 1 (sin alineación)
    accepted1, matches1, pair_rows1, cost_lookup1 = run_matching(None)

    # Estimar alineación con pares aceptados (pasada 1)
    align_enabled = bool(getattr(config, "CROSSCAM_ENABLE_ALIGN", False))
    align_model = str(getattr(config, "ALIGN_MODEL", "similarity")).strip().lower()
    align_ransac_iters = int(getattr(config, "ALIGN_RANSAC_ITERS", 1000))
    align_thr = float(getattr(config, "ALIGN_INLIER_THR", getattr(config, "CROSSCAM_INLIER_THR", pos_thr)))
    align_min_inliers = int(getattr(config, "ALIGN_MIN_INLIERS", 50))
    align_min_pairs = int(getattr(config, "ALIGN_MIN_PAIRS", 2 if align_model == "similarity" else 3))
    align_sample_step = int(getattr(config, "ALIGN_SAMPLE_STEP", sample_step))
    align_mat = None
    align_info = dict(
        enabled=align_enabled,
        model=align_model,
        inliers=0,
        inlier_pct=math.nan,
        err_median=math.nan,
        total_points=0,
        used=False,
    )

    if align_enabled and accepted1:
        src_pts = []
        dst_pts = []
        max_dist_loose = align_thr * 2.0
        for sid1, sid2, _ in accepted1:
            pos_a, pos_b, n_common, n_good = _get_positions(
                t1[sid1],
                t2[sid2],
                sample_step=align_sample_step,
                min_overlap=min_overlap,
                align_mat=None,
            )
            if pos_a is None or pos_b is None or len(pos_a) == 0:
                continue
            d = np.linalg.norm(pos_a - pos_b, axis=1)
            mask = d <= max_dist_loose
            if not mask.any():
                continue
            src_pts.append(pos_b[mask])
            dst_pts.append(pos_a[mask])
        if src_pts:
            src_all = np.concatenate(src_pts, axis=0)
            dst_all = np.concatenate(dst_pts, axis=0)
            align_info["total_points"] = len(src_all)
            align_mat_est, n_in, err_med = _ransac_align(
                src_all,
                dst_all,
                model=align_model,
                iters=align_ransac_iters,
                inlier_thr=align_thr,
                min_pairs=align_min_pairs,
                min_inliers=align_min_inliers,
            )
            if align_mat_est is not None:
                align_mat = align_mat_est
                align_info.update(
                    inliers=n_in,
                    inlier_pct=float(n_in) / float(len(src_all)) if len(src_all) > 0 else math.nan,
                    err_median=err_med,
                    used=True,
                )
    # Pasada 2 (con alineación si existe) o voting por ventanas
    vote_enable = bool(getattr(config, "VOTE_ENABLE", False))
    if vote_enable:
        W = int(getattr(config, "VOTE_WINDOW_FRAMES", 240))
        S = int(getattr(config, "VOTE_STEP_FRAMES", max(1, W // 2)))
        min_wins = int(getattr(config, "VOTE_MIN_WINS", 2))
        margin_wins = int(getattr(config, "VOTE_MARGIN_WINS", 1))
        require_mutual = bool(getattr(config, "VOTE_REQUIRE_MUTUAL", True))
        max_mean_cost = float(getattr(config, "VOTE_MAX_MEAN_COST", 1e9))
        vote_min_inlier_pct = float(getattr(config, "VOTE_MIN_INLIER_PCT", 0.0))

        wins = {}
        cost_sum = {}
        cost_cnt = {}
        inlier_sum = {}
        inlier_cnt = {}
        frames_min = min(int(df1["frame"].min()), int(df2["frame"].min()))
        frames_max = max(int(df1["frame"].max()), int(df2["frame"].max()))
        t = frames_min
        while t <= frames_max:
            window = (t, t + W)
            acc_w, matches_w, pair_rows_w, cost_lookup_w = run_matching(align_mat, window=window)
            for sid1, sid2, c in acc_w:
                wins[(sid1, sid2)] = wins.get((sid1, sid2), 0) + 1
                cost_sum[(sid1, sid2)] = cost_sum.get((sid1, sid2), 0.0) + float(c)
                cost_cnt[(sid1, sid2)] = cost_cnt.get((sid1, sid2), 0) + 1
                row = next((p for p in pair_rows_w if p["cam1_sid"] == sid1 and p["cam2_sid"] == sid2), None)
                if row and row.get("inlier_pct") is not None:
                    inlier_sum[(sid1, sid2)] = inlier_sum.get((sid1, sid2), 0.0) + float(row["inlier_pct"])
                    inlier_cnt[(sid1, sid2)] = inlier_cnt.get((sid1, sid2), 0) + 1
            t += S

        candidates = []
        best_for_sid2 = {}
        by_sid1 = {}
        for (sid1, sid2), w in wins.items():
            mc = cost_sum.get((sid1, sid2), 0.0) / max(1, cost_cnt.get((sid1, sid2), 1))
            mi = inlier_sum.get((sid1, sid2), 0.0) / max(1, inlier_cnt.get((sid1, sid2), 1))
            by_sid1.setdefault(sid1, []).append((sid2, w, mc, mi))
        for sid1, lst in by_sid1.items():
            lst.sort(key=lambda x: (-x[1], x[2]))
            best = lst[0]
            second_w = lst[1][1] if len(lst) > 1 else 0
            if best[1] < min_wins:
                continue
            if (best[1] - second_w) < margin_wins:
                continue
            if best[2] > max_mean_cost:
                continue
            if best[3] is not None and best[3] < vote_min_inlier_pct:
                continue
            candidates.append((sid1, best[0], best[1], best[2], best[3]))
            if best[0] not in best_for_sid2 or best_for_sid2[best[0]][2] < best[1]:
                best_for_sid2[best[0]] = (sid1, best[0], best[1], best[2], best[3])

        candidates.sort(key=lambda x: (-x[2], x[3]))
        used_sid2 = set()
        final_pairs = []
        for sid1, sid2, w, mc, mi in candidates:
            if require_mutual:
                best_sid2 = best_for_sid2.get(sid2)
                if not best_sid2 or best_sid2[0] != sid1:
                    continue
            if sid2 in used_sid2:
                continue
            used_sid2.add(sid2)
            final_pairs.append((sid1, sid2, mc))
        accepted = final_pairs
        matches = matches1
        pair_rows = pair_rows1
        cost_lookup = cost_lookup1
    else:
        accepted, matches, pair_rows, cost_lookup = run_matching(align_mat)

    # Handoff / complementario: emparejar tracklets sin solape suficiente pero cercanos en tiempo/espacio
    ho_enable = bool(getattr(config, "CROSSCAM_HO_ENABLE", True))
    n_handoff_candidates = 0
    n_handoff_accepted = 0
    if ho_enable:
        max_gap = int(getattr(config, "CROSSCAM_HO_MAX_GAP_FRAMES", 60))
        ho_pos_thr = float(getattr(config, "CROSSCAM_HO_POS_THR", pos_thr))
        ho_reid_thr = float(getattr(config, "CROSSCAM_HO_REID_THR", reid_thr))
        ho_w_pos = float(getattr(config, "CROSSCAM_HO_W_POS", getattr(config, "CROSSCAM_W_POS", 1.0)))
        ho_w_reid = float(getattr(config, "CROSSCAM_HO_W_REID", getattr(config, "CROSSCAM_W_REID", 0.0)))
        end_frames = int(getattr(config, "CROSSCAM_HO_ENDPOINT_FRAMES", 5))

        matched1 = {sid1 for sid1, _, _ in accepted}
        matched2 = {sid2 for _, sid2, _ in accepted}
        cand_rows = []
        for sid1, tr1 in t1.items():
            if sid1 in matched1:
                continue
            for sid2, tr2 in t2.items():
                if sid2 in matched2:
                    continue
                ov = compute_overlap_frames(tr1, tr2)
                if ov >= min_overlap:
                    continue  # ya cubierto por matching principal
                # gap temporal y direcciСn
                if tr1.t1 < tr2.t0:
                    gap = tr2.t0 - tr1.t1
                    pos_a = _endpoint_position(tr1, start=False, n_frames=end_frames, align_mat=None)
                    pos_b = _endpoint_position(tr2, start=True, n_frames=end_frames, align_mat=align_mat)
                elif tr2.t1 < tr1.t0:
                    gap = tr1.t0 - tr2.t1
                    pos_a = _endpoint_position(tr1, start=True, n_frames=end_frames, align_mat=None)
                    pos_b = _endpoint_position(tr2, start=False, n_frames=end_frames, align_mat=align_mat)
                else:
                    continue
                if gap > max_gap:
                    continue
                if pos_a is None or pos_b is None:
                    continue
                dist = float(np.linalg.norm(pos_a - pos_b))
                if dist > ho_pos_thr:
                    continue
                d_emb = None
                if (
                    not tr1.reid_disabled
                    and not tr2.reid_disabled
                    and tr1.emb_mean is not None
                    and tr2.emb_mean is not None
                ):
                    d_emb = float(1.0 - float(np.dot(tr1.emb_mean, tr2.emb_mean)))
                    if d_emb > ho_reid_thr:
                        continue
                norm_pos = dist / ho_pos_thr
                norm_emb = (d_emb / ho_reid_thr) if d_emb is not None else 0.0
                cost_ho = ho_w_pos * norm_pos + ho_w_reid * norm_emb
                cand_rows.append(
                    dict(
                        cam1_sid=int(sid1),
                        cam2_sid=int(sid2),
                        cost=float(cost_ho),
                        d_pos_med=dist,
                        d_pos_p80=dist,
                        d_emb=d_emb,
                        overlap_frames=0,
                        ok=True,
                        inlier_pct=math.nan,
                        inlier_count=0,
                        dist_mean=dist,
                        dist_max=dist,
                        team_a=tr1.team,
                        team_b=tr2.team,
                        team_match=(
                            None
                            if (tr1.team is None or tr2.team is None)
                            else bool(tr1.team == tr2.team)
                        ),
                        handoff=True,
                        gap_frames=int(gap),
                    )
                )
        n_handoff_candidates = len(cand_rows)
        cand_rows.sort(key=lambda r: r["cost"])
        used1 = set()
        used2 = set()
        handoff_pairs = []
        for row in cand_rows:
            sid1 = row["cam1_sid"]
            sid2 = row["cam2_sid"]
            if sid1 in used1 or sid2 in used2:
                continue
            used1.add(sid1)
            used2.add(sid2)
            handoff_pairs.append((sid1, sid2, row["cost"]))
            pair_rows.append(row)
        n_handoff_accepted = len(handoff_pairs)
        if handoff_pairs:
            accepted.extend(handoff_pairs)

    gid1, gid2 = build_gid_map(t1, t2, accepted)
    # Cobertura total: asigna GID nuevo a cualquier stable_id presente en los CSV stitched que no quedó en el mapping
    all_sid1 = set(df1["stable_id"].unique().tolist())
    all_sid2 = set(df2["stable_id"].unique().tolist())
    next_gid = 1 + max(list(gid1.values()) + list(gid2.values()) + [0])
    for sid in sorted(all_sid1):
        if sid not in gid1:
            gid1[sid] = next_gid
            next_gid += 1
    for sid in sorted(all_sid2):
        if sid not in gid2:
            gid2[sid] = next_gid
            next_gid += 1
    # Aplicar overrides manuales (manual_overrides.json con formato players) si existe
    def _parse_manual_overrides(obj) -> dict[int, dict[str, list[int]]]:
        """
        Espera formato:
          { "players": { "1": { "cam1": [..], "cam2": [..] }, ... } }
        Devuelve GID -> ids por camara.
        """
        if not isinstance(obj, dict):
            return {}
        players = obj.get("players")
        if not isinstance(players, dict):
            return {}
        out: dict[int, dict[str, list[int]]] = {}
        for gid_key, data in players.items():
            try:
                gid = int(gid_key)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            cam1_ids = []
            cam2_ids = []
            for sid in data.get("cam1") or data.get("cam1_ids") or []:
                try:
                    cam1_ids.append(int(sid))
                except Exception:
                    continue
            for sid in data.get("cam2") or data.get("cam2_ids") or []:
                try:
                    cam2_ids.append(int(sid))
                except Exception:
                    continue
            out[gid] = {
                "cam1": sorted(set(cam1_ids)),
                "cam2": sorted(set(cam2_ids)),
            }
        return out

    manual_overrides_path = os.path.join(run_dir, "manual_overrides.json")
    if os.path.exists(manual_overrides_path):
        try:
            with open(manual_overrides_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manual_players = _parse_manual_overrides(data)
        except Exception as exc:
            print(f"[WARN] No se pudo leer manual_overrides.json: {exc!r}")
            manual_players = {}
        if manual_players:
            applied_cam1 = 0
            applied_cam2 = 0
            for gid, entry in manual_players.items():
                for sid1 in entry.get("cam1", []):
                    if sid1 in all_sid1:
                        gid1[int(sid1)] = int(gid)
                        applied_cam1 += 1
                for sid2 in entry.get("cam2", []):
                    if sid2 in all_sid2:
                        gid2[int(sid2)] = int(gid)
                        applied_cam2 += 1
            print(
                f"[MANUAL] overrides applied | players={len(manual_players)} cam1_ids={applied_cam1} cam2_ids={applied_cam2}"
            )

    map_path = write_crosscam_map(run_dir, gid1, gid2)

    # Guardar info de alineación
    align_save_path_cfg = getattr(config, "ALIGN_SAVE_PATH", "crosscam_align.json")
    if not os.path.isabs(align_save_path_cfg):
        align_save_path = os.path.join(run_dir, align_save_path_cfg)
    else:
        align_save_path = align_save_path_cfg
    if align_enabled:
        try:
            with open(align_save_path, "w", encoding="utf-8") as f:
                json.dump(align_info, f, ensure_ascii=False, indent=2)
            print(f"[WRITE] {align_save_path}")
        except Exception as exc:
            print(f"[WARN] No se pudo guardar align info: {exc!r}")
    else:
        align_save_path = None

    # lookup para métricas de aceptados
    pair_lookup = {(p["cam1_sid"], p["cam2_sid"]): p for p in pair_rows}
    d_pos_vals = []
    d_emb_vals = []
    inlier_ok_pct = [p.get("inlier_pct") for p in pair_rows if p.get("ok") and p.get("inlier_pct") is not None]
    inlier_acc_pct = []
    for sid1, sid2, _c in accepted:
        row = pair_lookup.get((sid1, sid2))
        if row:
            if math.isfinite(row.get("d_pos_med", math.inf)):
                d_pos_vals.append(row["d_pos_med"])
            if row.get("d_emb") is not None and math.isfinite(row.get("d_emb")):
                d_emb_vals.append(row["d_emb"])
            if row.get("inlier_pct") is not None:
                inlier_acc_pct.append(row["inlier_pct"])
    summary = dict(
        n_tracklets_cam1=len(t1),
        n_tracklets_cam2=len(t2),
        n_candidates=len(pair_rows),
        n_matches=len(matches),
        n_accepted=len(accepted),
        n_tracklets_team_known_cam1=int(sum(1 for t in t1.values() if t.team is not None)),
        n_tracklets_team_known_cam2=int(sum(1 for t in t2.values() if t.team is not None)),
        pos_thr=pos_thr,
        reid_thr=reid_thr,
        min_overlap=min_overlap,
        accept_margin=accept_margin,
        map_path=map_path,
        align_enabled=align_enabled,
        align_model=align_model,
        align_inliers=int(align_info.get("inliers", 0)),
        align_inlier_pct=float(align_info.get("inlier_pct", math.nan)),
        align_err_median=float(align_info.get("err_median", math.nan)),
        align_total_points=int(align_info.get("total_points", 0)),
        align_used=bool(align_info.get("used", False)),
        align_save_path=align_save_path,
        d_pos_median_accepted=float(np.median(d_pos_vals)) if d_pos_vals else math.nan,
        d_emb_median_accepted=float(np.median(d_emb_vals)) if d_emb_vals else math.nan,
        inlier_median_ok=float(np.median(inlier_ok_pct)) if inlier_ok_pct else math.nan,
        inlier_median_accepted=float(np.median(inlier_acc_pct)) if inlier_acc_pct else math.nan,
        n_pairs_blocked_team=int(
            sum(
                1
                for p in pair_rows
                if p.get("team_match") is False and str(getattr(config, "TEAM_MODE", "penalty")).lower() == "gate"
            )
        ),
        n_handoff_candidates=int(n_handoff_candidates),
        n_handoff_accepted=int(n_handoff_accepted),
    )
    report_path, summary_path = write_report_crosscam(run_dir, summary=summary, pair_rows=pair_rows)

    global_tracks_path = None
    fuse_report_path = None
    if getattr(config, "CROSSCAM_WRITE_GLOBAL_TRACKS", True):
        res = write_global_tracks(run_dir, df1, df2, gid1, gid2, align_mat)
        if res:
            if isinstance(res, tuple):
                global_tracks_path, fuse_report_path = res
            else:
                global_tracks_path = res

    print("[DONE] offline_crosscam outputs:")
    print(f"  map: {map_path}")
    print(f"  report: {report_path}")
    print(f"  report_summary: {summary_path}")
    if global_tracks_path:
        print(f"  global_tracks: {global_tracks_path}")
    if fuse_report_path:
        print(f"  fuse_report: {fuse_report_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import config


# =========================
# Stitch config (edit here)
# =========================
DEFAULT_FPS = 30.0
MAX_GAP_FRAMES = 60
DT_SPLIT_MAX = 2
V_MAX = 600.0
POS_GATE_A = 120.0
POS_GATE_B = 2.0
REID_THR = 0.35
MIN_EMB_RATIO = 0.25
VEL_K = 5
W_POS = 1.0
W_EMB = 1.0


@dataclass(frozen=True)
class StitchConfig:
    fps: float
    max_gap_frames: int
    dt_split_max: int
    v_max: float
    pos_gate_a: float
    pos_gate_b: float
    reid_thr: float
    min_emb_ratio: float
    vel_k: int
    w_pos: float
    w_emb: float


@dataclass
class Tracklet:
    tid: int
    orig_id: int
    frames: np.ndarray
    pos: np.ndarray
    n_obs: int
    emb_mean: Optional[np.ndarray]
    emb_trust: bool
    emb_obs_count: int
    emb_obs_ratio: float

    @property
    def first_frame(self) -> int:
        return int(self.frames[0]) if len(self.frames) else -1

    @property
    def last_frame(self) -> int:
        return int(self.frames[-1]) if len(self.frames) else -1

    @property
    def pos_start(self) -> Tuple[float, float]:
        if len(self.pos) == 0:
            return (float("nan"), float("nan"))
        return (float(self.pos[0, 0]), float(self.pos[0, 1]))

    @property
    def pos_end(self) -> Tuple[float, float]:
        if len(self.pos) == 0:
            return (float("nan"), float("nan"))
        return (float(self.pos[-1, 0]), float(self.pos[-1, 1]))


def _load_meta(run_dir: str) -> dict:
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_config(run_dir: str) -> StitchConfig:
    meta = _load_meta(run_dir)
    fps = float(meta.get("fps", DEFAULT_FPS))
    return StitchConfig(
        fps=fps,
        max_gap_frames=int(MAX_GAP_FRAMES),
        dt_split_max=int(DT_SPLIT_MAX),
        v_max=float(V_MAX),
        pos_gate_a=float(POS_GATE_A),
        pos_gate_b=float(POS_GATE_B),
        reid_thr=float(REID_THR),
        min_emb_ratio=float(MIN_EMB_RATIO),
        vel_k=int(VEL_K),
        w_pos=float(W_POS),
        w_emb=float(W_EMB),
    )


def _l2norm(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = float(np.sqrt((vec * vec).sum())) + eps
    return vec.astype(np.float32) / denom


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def _pos_thr(cfg: StitchConfig, gap: int) -> float:
    return float(cfg.pos_gate_a + cfg.pos_gate_b * float(gap))


def _distribute_counts(total: int, lengths: List[int]) -> List[int]:
    if total <= 0 or not lengths:
        return [0 for _ in lengths]
    total_len = float(sum(lengths))
    raw = [total * (l / total_len) for l in lengths]
    base = [int(np.floor(v)) for v in raw]
    rem = total - sum(base)
    if rem <= 0:
        return base
    frac = [(v - b, idx) for idx, (v, b) in enumerate(zip(raw, base))]
    frac.sort(reverse=True)
    for i in range(rem):
        base[frac[i % len(base)][1]] += 1
    return base


def _split_by_speed(
    frames: np.ndarray, pos: np.ndarray, cfg: StitchConfig
) -> Tuple[List[Tuple[int, int]], bool]:
    n = len(frames)
    if n <= 1:
        return [(0, n - 1)], False
    segments = []
    start = 0
    high_count = 0
    split_triggered = False
    for i in range(1, n):
        dt = int(frames[i] - frames[i - 1])
        if dt <= 0:
            continue
        if dt > cfg.dt_split_max:
            high_count = 0
            continue
        if not np.isfinite(pos[i]).all() or not np.isfinite(pos[i - 1]).all():
            high_count = 0
            continue
        dist = float(np.linalg.norm(pos[i] - pos[i - 1]))
        v = dist / float(dt)
        strong = v > (1.5 * cfg.v_max)
        if strong:
            split_triggered = True
            if start <= i - 1:
                segments.append((start, i - 1))
            start = i
            high_count = 0
            continue
        if v > cfg.v_max:
            high_count += 1
            if high_count >= 2:
                split_triggered = True
                if start <= i - 1:
                    segments.append((start, i - 1))
                start = i
                high_count = 0
        else:
            high_count = 0
    if start <= n - 1:
        segments.append((start, n - 1))
    return segments, split_triggered


def _estimate_tail_velocity(frames: np.ndarray, pos: np.ndarray, cfg: StitchConfig) -> np.ndarray:
    if len(frames) <= 1:
        return np.zeros(2, dtype=np.float32)
    vels = []
    for i in range(len(frames) - 1, 0, -1):
        if len(vels) >= cfg.vel_k:
            break
        dt = int(frames[i] - frames[i - 1])
        if dt <= 0 or dt > cfg.dt_split_max:
            continue
        if not np.isfinite(pos[i]).all() or not np.isfinite(pos[i - 1]).all():
            continue
        vels.append((pos[i] - pos[i - 1]) / float(dt))
    if not vels:
        return np.zeros(2, dtype=np.float32)
    return np.median(np.stack(vels, axis=0), axis=0).astype(np.float32)


def _load_embeddings(path: str) -> Dict[int, np.ndarray]:
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    ids = data.get("stable_ids")
    embs = data.get("embeddings")
    data.close()
    if ids is None or embs is None:
        return {}
    out = {}
    for sid, emb in zip(ids.tolist(), embs):
        if emb is None:
            continue
        out[int(sid)] = _l2norm(np.asarray(emb, dtype=np.float32))
    return out


def _load_summary(path: str) -> Dict[int, dict]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    if df.empty or "stable_id" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        sid = int(row["stable_id"])
        out[sid] = {
            "n_obs": int(row.get("n_obs", 0)) if "n_obs" in row else 0,
            "n_obs_emb": int(row.get("n_obs_emb", 0)) if "n_obs_emb" in row else 0,
        }
    return out


def _build_tracklets(
    df_tracks: pd.DataFrame,
    summary: Dict[int, dict],
    embeddings: Dict[int, np.ndarray],
    cfg: StitchConfig,
) -> Tuple[List[Tracklet], Dict[int, List[Tuple[int, int, int]]], int]:
    tracklets: List[Tracklet] = []
    segments_info: Dict[int, List[Tuple[int, int, int]]] = {}
    next_tid = 1
    split_events = 0

    for sid, g in df_tracks.groupby("stable_id_original", sort=False):
        g = g.sort_values("frame", kind="mergesort")
        frames = g["frame"].to_numpy(dtype=np.int64)
        pos = g[["pos_x", "pos_y"]].to_numpy(dtype=np.float32)
        if len(frames) == 0:
            continue
        stats = summary.get(int(sid), {})
        n_obs = int(stats.get("n_obs", len(frames)))
        if n_obs <= 0:
            n_obs = int(len(frames))
        n_obs_emb = int(stats.get("n_obs_emb", 0))
        emb_obs_ratio = (float(n_obs_emb) / float(n_obs)) if n_obs > 0 else 0.0
        base_emb = embeddings.get(int(sid))

        segments, split_triggered = _split_by_speed(frames, pos, cfg)
        if split_triggered:
            split_events += max(0, len(segments) - 1)
        emb_counts = _distribute_counts(n_obs_emb, [e - s + 1 for s, e in segments])
        segments_info[int(sid)] = []

        for (s, e), seg_emb_cnt in zip(segments, emb_counts):
            seg_frames = frames[s : e + 1]
            seg_pos = pos[s : e + 1]
            emb_trust = True  # no apagues embeddings por split (evita quedarse sin pegamento)
            emb_mean = None
            if emb_trust and base_emb is not None and emb_obs_ratio >= cfg.min_emb_ratio:
                emb_mean = base_emb
            if len(seg_frames) < 12:
                continue  # descarta micro-tracklets
            tracklets.append(
                Tracklet(
                    tid=next_tid,
                    orig_id=int(sid),
                    frames=seg_frames,
                    pos=seg_pos,
                    n_obs=int(len(seg_frames)),
                    emb_mean=emb_mean,
                    emb_trust=emb_trust,
                    emb_obs_count=int(seg_emb_cnt),
                    emb_obs_ratio=float(emb_obs_ratio),
                )
            )
            segments_info[int(sid)].append(
                (int(seg_frames[0]), int(seg_frames[-1]), int(next_tid))
            )
            next_tid += 1

    return tracklets, segments_info, split_events


def _stitch_tracklets(tracklets: List[Tracklet], cfg: StitchConfig) -> Tuple[Dict[int, int], int]:
    n = len(tracklets)
    if n == 0:
        return {}, 0
    big = 1e9
    cost = np.full((n, n), big, dtype=np.float32)
    tail_vel = [ _estimate_tail_velocity(t.frames, t.pos, cfg) for t in tracklets ]

    for i, a in enumerate(tracklets):
        for j, b in enumerate(tracklets):
            gap = b.first_frame - a.last_frame
            if gap <= 0 or gap > cfg.max_gap_frames:
                continue
            pos_thr = _pos_thr(cfg, gap)
            if pos_thr <= 0:
                continue
            v_a = tail_vel[i]
            pos_pred = np.array(a.pos_end, dtype=np.float32) + v_a * float(gap)
            pos_b = np.array(b.pos_start, dtype=np.float32)
            if not np.isfinite(pos_pred).all() or not np.isfinite(pos_b).all():
                continue
            d_pos = float(np.linalg.norm(pos_pred - pos_b))
            if d_pos > pos_thr:
                continue
            if a.emb_mean is not None and b.emb_mean is not None and a.emb_trust and b.emb_trust:
                d_emb = _cosine_dist(a.emb_mean, b.emb_mean)
                if d_emb > cfg.reid_thr:
                    continue
                cost[i, j] = cfg.w_pos * (d_pos / pos_thr) + cfg.w_emb * (d_emb / cfg.reid_thr)
            else:
                cost[i, j] = cfg.w_pos * (d_pos / pos_thr)

    row_ind, col_ind = linear_sum_assignment(cost)
    next_map: Dict[int, int] = {}
    prev_map: Dict[int, int] = {}
    merges = 0
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[r, c] >= big / 2.0:
            continue
        a_id = tracklets[r].tid
        b_id = tracklets[c].tid
        next_map[a_id] = b_id
        prev_map[b_id] = a_id
        merges += 1

    stitched_map: Dict[int, int] = {}
    stitched_id = 1
    for t in sorted(tracklets, key=lambda x: x.first_frame):
        if t.tid in prev_map:
            continue
        cur = t.tid
        while cur is not None and cur not in stitched_map:
            stitched_map[cur] = stitched_id
            cur = next_map.get(cur)
        stitched_id += 1
    for t in tracklets:
        if t.tid not in stitched_map:
            stitched_map[t.tid] = stitched_id
            stitched_id += 1
    return stitched_map, merges


def _apply_stitch_mapping(
    df_tracks: pd.DataFrame,
    segments_info: Dict[int, List[Tuple[int, int, int]]],
    stitched_map: Dict[int, int],
) -> pd.DataFrame:
    def _lookup_sub_id(row):
        segs = segments_info.get(int(row["stable_id_original"]), [])
        f = int(row["frame"])
        for start, end, sub_id in segs:
            if start <= f <= end:
                return sub_id
        return None

    df_out = df_tracks.copy()
    df_out["subtracklet_id"] = df_out.apply(_lookup_sub_id, axis=1)

    # Importante:
    # - subtracklet_id vive en el espacio de IDs internos (next_tid en _build_tracklets)
    # - stable_id_original vive en el espacio de IDs del tracker/StableIDAssigner
    # Mezclarlos (fallback directo) puede colisionar y producir el bug: mismo ID para 2 jugadores.
    missing_sub = df_out["subtracklet_id"].isna()
    if missing_sub.any():
        # Crea pseudo-IDs negativos (no colisionan con subtracklet_id reales que son positivos)
        df_out.loc[missing_sub, "subtracklet_id"] = -df_out.loc[missing_sub, "stable_id_original"].astype(int)

        # Asegura que TODO subtracklet_id tenga una salida en stitched_map
        next_sid = (max(stitched_map.values()) + 1) if stitched_map else 1
        for tid in pd.unique(df_out.loc[missing_sub, "subtracklet_id"].astype(int)):
            if int(tid) not in stitched_map:
                stitched_map[int(tid)] = int(next_sid)
                next_sid += 1

    df_out["stitched_id"] = df_out["subtracklet_id"].astype(int).map(stitched_map)
    missing_stitched = df_out["stitched_id"].isna()
    if missing_stitched.any():
        # Si por alguna razón aún falta mapping (no debería), crea IDs nuevos sin colisionar.
        next_sid = (max(stitched_map.values()) + 1) if stitched_map else 1
        for tid in pd.unique(df_out.loc[missing_stitched, "subtracklet_id"].astype(int)):
            if int(tid) not in stitched_map:
                stitched_map[int(tid)] = int(next_sid)
                next_sid += 1
        df_out["stitched_id"] = df_out["subtracklet_id"].astype(int).map(stitched_map)

    df_out["stable_id"] = df_out["stitched_id"].astype(int)
    df_out = df_out.drop(columns=["subtracklet_id", "stitched_id"])
    return df_out


def _build_stitched_embeddings(
    tracklets: List[Tracklet], stitched_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    bucket: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    for t in tracklets:
        if t.emb_mean is None:
            continue
        sid = int(stitched_map[t.tid])
        w = int(t.emb_obs_count) if t.emb_obs_count > 0 else int(t.n_obs)
        bucket.setdefault(sid, []).append((t.emb_mean, max(1, w)))
    stitched_ids = sorted(bucket.keys())
    if not stitched_ids:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)
    emb_list = []
    for sid in stitched_ids:
        pairs = bucket[sid]
        total = float(sum(w for _, w in pairs))
        emb = sum((vec * float(w) for vec, w in pairs)) / total
        emb_list.append(_l2norm(np.asarray(emb, dtype=np.float32)))
    return np.asarray(stitched_ids, dtype=np.int32), np.stack(emb_list, axis=0)


def _build_stitched_summary(
    tracklets: List[Tracklet], stitched_map: Dict[int, int]
) -> pd.DataFrame:
    groups: Dict[int, List[Tracklet]] = {}
    for t in tracklets:
        sid = int(stitched_map[t.tid])
        groups.setdefault(sid, []).append(t)
    rows = []
    for sid, tlist in groups.items():
        first_t = min(tlist, key=lambda x: x.first_frame)
        last_t = max(tlist, key=lambda x: x.last_frame)
        n_obs_total = int(sum(t.n_obs for t in tlist))
        n_obs_emb_total = int(sum(t.emb_obs_count for t in tlist))
        emb_ratio = (float(n_obs_emb_total) / float(n_obs_total)) if n_obs_total > 0 else 0.0
        rows.append(
            dict(
                stitched_id=int(sid),
                first_frame=int(first_t.first_frame),
                last_frame=int(last_t.last_frame),
                n_obs_total=n_obs_total,
                n_obs_emb_total=n_obs_emb_total,
                emb_obs_ratio_total=float(emb_ratio),
                pos_start_x=float(first_t.pos_start[0]),
                pos_start_y=float(first_t.pos_start[1]),
                pos_end_x=float(last_t.pos_end[0]),
                pos_end_y=float(last_t.pos_end[1]),
            )
        )
    return pd.DataFrame(rows)


def _process_camera(run_dir: str, cam: str, cfg: StitchConfig) -> Optional[dict]:
    tracks_path = os.path.join(run_dir, f"{cam}_tracks.csv")
    if not os.path.exists(tracks_path):
        print(f"[WARN] Missing tracks file: {tracks_path}")
        return None

    df_tracks = pd.read_csv(tracks_path)
    if df_tracks.empty:
        print(f"[WARN] Empty tracks file: {tracks_path}")
        return None
    # Clip window opcional
    if getattr(config, "CLIP_ENABLE", False):
        start = int(getattr(config, "CLIP_START_FRAME", 0))
        end = int(getattr(config, "CLIP_END_FRAME", 0))
        if "frame" in df_tracks.columns:
            df_tracks = df_tracks[df_tracks["frame"].between(start, end if end > 0 else df_tracks["frame"].max())]
            df_tracks = df_tracks.copy()

    required_cols = ["frame", "stable_id", "pos_x", "pos_y"]
    for col in required_cols:
        if col not in df_tracks.columns:
            raise ValueError(f"Missing column {col} in {tracks_path}")

    orig_col = "stable_id_original" if "stable_id_original" in df_tracks.columns else "stable_id"
    df_tracks["stable_id_original"] = df_tracks[orig_col].astype(int)
    df_tracks["frame"] = df_tracks["frame"].astype(int)

    summary_path = os.path.join(run_dir, f"{cam}_tracklet_summary.csv")
    emb_path = os.path.join(run_dir, f"{cam}_embeddings.npz")
    summary = _load_summary(summary_path)
    embeddings = _load_embeddings(emb_path)

    tracklets, segments_info, split_events = _build_tracklets(df_tracks, summary, embeddings, cfg)
    stitched_map, merges = _stitch_tracklets(tracklets, cfg)

    df_out = _apply_stitch_mapping(df_tracks, segments_info, stitched_map)

    # Vanilla: sin limpieza ni glitch removal
    keep_ids = set(df_out["stable_id"].unique().astype(int).tolist())

    # Sin glitch-fix: dejamos pos_x/pos_y tal cual homografía

    out_tracks_path = os.path.join(run_dir, f"{cam}_tracks_stitched.csv")
    df_out.to_csv(out_tracks_path, index=False)

    stitched_ids, stitched_embs = _build_stitched_embeddings(tracklets, stitched_map)
    if not getattr(config, "STITCH_DISABLE_CLEAN", False):
        mask = np.array([int(sid) in keep_ids for sid in stitched_ids.tolist()], dtype=bool)
        stitched_ids = stitched_ids[mask]
        stitched_embs = stitched_embs[mask]
    out_emb_path = os.path.join(run_dir, f"{cam}_stitched_embeddings.npz")
    np.savez_compressed(out_emb_path, stitched_ids=stitched_ids, embeddings=stitched_embs)

    summary_df = _build_stitched_summary(tracklets, stitched_map)
    if not getattr(config, "STITCH_DISABLE_CLEAN", False):
        summary_df = summary_df[summary_df["stitched_id"].astype(int).isin(keep_ids)].copy()
    out_summary_path = os.path.join(run_dir, f"{cam}_tracklet_summary_stitched.csv")
    summary_df.to_csv(out_summary_path, index=False)

    orig_to_stitched: Dict[int, List[int]] = {}
    for t in tracklets:
        orig_to_stitched.setdefault(t.orig_id, set()).add(int(stitched_map[t.tid]))
    orig_to_stitched = {str(k): sorted(v) for k, v in orig_to_stitched.items()}
    sub_to_stitched = {str(k): int(v) for k, v in stitched_map.items()}
    map_path = os.path.join(run_dir, f"{cam}_map_stitch.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "orig_to_stitched": orig_to_stitched,
                "subtracklet_to_stitched": sub_to_stitched,
            },
            f,
            indent=2,
        )

    n_tracklets_in = int(df_tracks["stable_id_original"].nunique())
    n_tracklets_after_split = int(len(tracklets))
    n_tracklets_out = int(len(set(stitched_map.values())))
    n_no_emb = sum(1 for t in tracklets if t.emb_mean is None)
    n_low_ratio = sum(1 for t in tracklets if t.emb_obs_ratio < cfg.min_emb_ratio)
    return dict(
        cam=cam,
        fps=float(cfg.fps),
        max_gap_frames=int(cfg.max_gap_frames),
        dt_split_max=int(cfg.dt_split_max),
        v_max=float(cfg.v_max),
        reid_thr=float(cfg.reid_thr),
        min_emb_ratio=float(cfg.min_emb_ratio),
        n_tracklets_in=n_tracklets_in,
        n_tracklets_after_split=n_tracklets_after_split,
        n_splits=int(split_events),
        n_merges=int(merges),
        n_tracklets_out=n_tracklets_out,
        n_no_emb=n_no_emb,
        n_low_emb_ratio=n_low_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline stitch for per-camera tracklets.")
    parser.add_argument("--run-dir", default=None, help="Run directory with c1/c2 exports.")
    parser.add_argument("--cam", action="append", choices=["c1", "c2"], help="Which camera to process.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        try:
            import config

            run_dir = getattr(config, "RUN_DIR", "runs/default")
        except Exception:
            run_dir = "runs/default"

    cfg = _build_config(run_dir)
    cams = args.cam if args.cam else ["c1", "c2"]
    report_rows = []
    for cam in cams:
        row = _process_camera(run_dir, cam, cfg)
        if row:
            report_rows.append(row)
    if report_rows:
        report_path = os.path.join(run_dir, "report_stitch.csv")
        pd.DataFrame(report_rows).to_csv(report_path, index=False)


if __name__ == "__main__":
    main()

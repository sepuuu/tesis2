from dataclasses import asdict, is_dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from config import TRACKER_CONFIG
except Exception:  # pragma: no cover - en caso de ejecuciones sin config
    TRACKER_CONFIG = None


def _cfg_to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj.copy()
    if is_dataclass(obj):
        return asdict(obj)
    attrs = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if not callable(value):
            attrs[key] = value
    return attrs


class Track:
    def __init__(self, track_id, position, embedding):
        self.track_id = track_id
        self.position = position.astype(np.float32)      # pos en cancha (x,y)
        self.embedding = embedding.astype(np.float32)    # vector reid
        self.velocity = np.zeros(2, dtype=np.float32)    # vel en cancha (x,y)
        self.state = "ACTIVE"        # ACTIVE o LOST
        self.lost_frames = 0         # frames consecutivos sin match
        self.age = 0            # frames desde el ǧltimo match
        self.hits = 1           # veces que fue matcheado
        self.time_since_update = 0
        self.pred_position = self.position.copy()


class SimpleTracker:
    """
    Asociador simple con memoria:
      - Costo = w_reid * dist_reid + w_pos * (dist_pos / pos_thr)
      - Gating por dist_pos y dist_reid
      - max_age: si no hay match por N frames, se borra
      - EMA en posici�n y embedding para estabilidad
      - Modelo de movimiento lineal (pos + vel) para gating
    """
    def __init__(self, cfg=None, **overrides):
        params = _cfg_to_dict(TRACKER_CONFIG)
        params.update(_cfg_to_dict(cfg) if cfg is not None else {})
        params.update(overrides)
        self.reid_weight = params.get("reid_weight", 0.7)
        self.pos_weight = params.get("pos_weight", 0.3)
        self.reid_threshold = params.get("reid_threshold", 0.8)
        self.pos_threshold = params.get("pos_threshold", 220.0)
        self.pos_thr_growth = params.get("pos_thr_growth", 6.0)
        self.pos_thr_max = params.get("pos_thr_max", 600.0)
        self.max_age = params.get("max_age", 15)
        self.emb_momentum = params.get("emb_momentum", 0.2)
        self.pos_alpha = params.get("pos_alpha", 0.4)
        self.motion_momentum = params.get("motion_momentum", 0.6)
        self.pos_gate_mult = params.get("pos_gate_mult", 1.2)
        self.max_lost = params.get("max_lost", 8)
        self.lost_reid_weight = params.get("lost_reid_weight", 0.9)
        self.lost_pos_weight = params.get("lost_pos_weight", 0.2)
        self.lost_pos_gate_mult = params.get("lost_pos_gate_mult", 1.3)
        self.lost_reid_threshold = params.get("lost_reid_threshold", 0.6)
        # Reutilizaci�n de IDs borrados para que el contador no se dispare
        self.reuse_pool_max_age = params.get("reuse_pool_max_age", 60)
        self.reuse_pos_gate_mult = params.get("reuse_pos_gate_mult", 2.0)
        self.reuse_reid_threshold = params.get("reuse_reid_threshold", 0.8)
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.reuse_pool = []  # [{track_id, position, embedding, last_seen}]
        self.frame_idx = 0

    def _ensure_legacy_defaults(self):
        """Compatibilidad: inicializa atributos y m�todos ausentes en instancias antiguas."""
        if not hasattr(self, "tracks"):
            self.tracks = {}
        if not hasattr(self, "next_id"):
            self.next_id = 1
        if not hasattr(self, "max_age"):
            self.max_age = 15
        if not hasattr(self, "motion_momentum"):
            self.motion_momentum = 0.6
        if not hasattr(self, "pos_gate_mult"):
            self.pos_gate_mult = 1.2
        if not hasattr(self, "pos_thr_growth"):
            self.pos_thr_growth = 6.0
        if not hasattr(self, "pos_thr_max"):
            self.pos_thr_max = 600.0
        if not hasattr(self, "max_lost"):
            self.max_lost = 8
        if not hasattr(self, "lost_reid_weight"):
            self.lost_reid_weight = 0.9
        if not hasattr(self, "lost_pos_weight"):
            self.lost_pos_weight = 0.2
        if not hasattr(self, "lost_pos_gate_mult"):
            self.lost_pos_gate_mult = 1.3
        if not hasattr(self, "lost_reid_threshold"):
            self.lost_reid_threshold = 0.6
        if not hasattr(self, "reuse_pool"):
            self.reuse_pool = []
        if not hasattr(self, "reuse_pool_max_age"):
            self.reuse_pool_max_age = 60
        if not hasattr(self, "reuse_pos_gate_mult"):
            self.reuse_pos_gate_mult = 2.0
        if not hasattr(self, "reuse_reid_threshold"):
            self.reuse_reid_threshold = 0.8
        if not hasattr(self, "frame_idx"):
            self.frame_idx = 0

        # M�todo faltante por versiones viejas serializadas (pickle) del tracker
        if not callable(getattr(self, "_norm_emb", None)):
            self._norm_emb = SimpleTracker._norm_emb.__get__(self, SimpleTracker)
        if not callable(getattr(self, "_reid_dist", None)):
            self._reid_dist = SimpleTracker._reid_dist.__get__(self, SimpleTracker)
        if not callable(getattr(self, "_pos_dist", None)):
            self._pos_dist = SimpleTracker._pos_dist.__get__(self, SimpleTracker)
        if not callable(getattr(self, "_cost", None)):
            self._cost = SimpleTracker._cost.__get__(self, SimpleTracker)
        if not callable(getattr(self, "_predict", None)):
            self._predict = SimpleTracker._predict.__get__(self, SimpleTracker)
        if not callable(getattr(self, "_dyn_pos_thr", None)):
            self._dyn_pos_thr = SimpleTracker._dyn_pos_thr.__get__(self, SimpleTracker)
        if not callable(getattr(self, "_cost_lost", None)):
            self._cost_lost = SimpleTracker._cost_lost.__get__(self, SimpleTracker)

    def _norm_emb(self, e):
        n = np.linalg.norm(e) + 1e-6
        return e / n

    def _reid_dist(self, e1, e2):
        e1 = self._norm_emb(e1)
        e2 = self._norm_emb(e2)
        return np.linalg.norm(e1 - e2)

    def _pos_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _predict(self, track: Track):
        # dt = frames desde el último update (ya lo incrementas al inicio del frame)
        dt = int(getattr(track, "time_since_update", 1))
        if dt < 1:
            dt = 1
        return track.position + track.velocity * dt

    def _dyn_pos_thr(self, track: Track) -> float:
        dt = int(getattr(track, "time_since_update", 1))
        if dt < 1:
            dt = 1
        thr = self.pos_threshold + self.pos_thr_growth * (dt - 1)
        return min(thr, self.pos_thr_max)

    def _cost(self, emb, pos, track: Track):
        pred_pos = getattr(track, "pred_position", track.position)
        d_reid = self._reid_dist(emb, track.embedding)
        d_pos = self._pos_dist(pos, pred_pos)

        base_thr = self._dyn_pos_thr(track)
        gate = base_thr * self.pos_gate_mult

        if d_pos > gate or d_reid > self.reid_threshold:
            return np.inf
        # normaliza por base_thr (dinámico), no por el fijo
        return self.reid_weight * d_reid + self.pos_weight * (d_pos / (base_thr + 1e-6))

    def _cost_lost(self, emb, pos, track: Track):
        # Segunda pasada: m�s peso a ReID, gating de posici�n m�s laxo
        pred_pos = getattr(track, "pred_position", track.position)
        d_reid = self._reid_dist(emb, track.embedding)
        d_pos = self._pos_dist(pos, pred_pos)

        base_thr = self._dyn_pos_thr(track)
        gate = base_thr * self.lost_pos_gate_mult

        if d_pos > gate or d_reid > self.lost_reid_threshold:
            return np.inf
        return self.lost_reid_weight * d_reid + self.lost_pos_weight * (d_pos / (base_thr + 1e-6))

    def _cost_reuse(self, emb, pos, entry):
        d_reid = self._reid_dist(emb, entry["embedding"])
        d_pos = self._pos_dist(pos, entry["position"])
        if d_pos > (self.pos_threshold * self.reuse_pos_gate_mult) or d_reid > self.reuse_reid_threshold:
            return np.inf
        return self.reid_weight * d_reid + self.pos_weight * (d_pos / self.pos_threshold)

    def update(self, detections):
        """
        detections: list[(pos (2,), emb (D,))]
        return: list[(track_id, pos, emb)] en el MISMO orden que 'detections'
        """
        self._ensure_legacy_defaults()
        self.frame_idx += 1
        # expira ids antiguos del pool de reutilizaci�n
        if self.reuse_pool:
            self.reuse_pool = [
                e for e in self.reuse_pool
                if (self.frame_idx - e.get("last_seen", 0)) <= self.reuse_pool_max_age
            ]

        # envejecer y predecir
        for tr in self.tracks.values():
            tr.age += 1
            tr.time_since_update += 1
            if not hasattr(tr, "velocity"):
                tr.velocity = np.zeros(2, dtype=np.float32)
            if not hasattr(tr, "state"):
                tr.state = "ACTIVE"
            if not hasattr(tr, "lost_frames"):
                tr.lost_frames = 0
            # Si sigue LOST, debe envejecer cada frame para poder expirar
            if tr.state == "LOST":
                tr.lost_frames += 1
            tr.pred_position = self._predict(tr)

        num_dets = len(detections)
        assignments = [None] * num_dets

        if num_dets == 0:
            # No hay detecciones: todos los ACTIVE pasan a LOST y envejecen
            for tid, tr in self.tracks.items():
                if tr.state == "ACTIVE":
                    tr.state = "LOST"
                    tr.lost_frames = max(tr.lost_frames, 1)

            # eliminar tracks viejos (esto ya lo haces al final, pero aquí ayuda a cortar “fantasmas”)
            to_del = [tid for tid, tr in self.tracks.items() if tr.lost_frames > self.max_lost]
            for tid in to_del:
                tr = self.tracks[tid]
                self.reuse_pool.append({
                    "track_id": tid,
                    "position": tr.position.copy(),
                    "embedding": tr.embedding.copy(),
                    "last_seen": self.frame_idx,
                })
                del self.tracks[tid]

            if len(self.reuse_pool) > 200:
                self.reuse_pool = self.reuse_pool[-200:]

            return assignments

        if num_dets:
            active_track_ids = [tid for tid, tr in self.tracks.items() if tr.state == "ACTIVE"]
            lost_track_ids = [tid for tid, tr in self.tracks.items() if tr.state == "LOST"]
            num_tracks = len(active_track_ids)
            invalid_cost = 1e6

            unmatched_dets = set(range(num_dets))
            matched_active = set()

            if num_tracks:
                cost_matrix = np.full((num_dets, num_tracks), invalid_cost, dtype=np.float32)
                for i, (pos, emb) in enumerate(detections):
                    for j, tid in enumerate(active_track_ids):
                        c = self._cost(emb, pos, self.tracks[tid])
                        if np.isfinite(c):
                            cost_matrix[i, j] = c

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r, c in zip(row_ind, col_ind):
                    cost = cost_matrix[r, c]
                    if not np.isfinite(cost) or cost >= invalid_cost:
                        continue

                    tid = active_track_ids[c]
                    tr = self.tracks[tid]
                    pos, emb = detections[r]

                    prev_pos = tr.position.copy()
                    # EMA en embedding y posici�n
                    tr.embedding = (1 - self.emb_momentum) * tr.embedding + self.emb_momentum * emb
                    tr.position = self.pos_alpha * tr.position + (1 - self.pos_alpha) * pos
                    dt = int(getattr(tr, "time_since_update", 1))
                    if dt < 1:
                        dt = 1
                    delta = (pos - prev_pos) / dt
                    tr.velocity = self.motion_momentum * tr.velocity + (1 - self.motion_momentum) * delta
                    tr.age = 0
                    tr.hits += 1
                    tr.time_since_update = 0
                    tr.state = "ACTIVE"
                    tr.lost_frames = 0

                    assignments[r] = (tid, tr.position.copy(), tr.embedding.copy())
                    unmatched_dets.discard(r)
                    matched_active.add(tid)

            # Marcar ACTIVE no matcheados como LOST
            for tid in active_track_ids:
                if tid not in matched_active:
                    tr = self.tracks[tid]
                    tr.state = "LOST"
                    tr.lost_frames += 1

            # Segunda pasada: reacquire con tracks LOST
            if unmatched_dets and lost_track_ids:
                cost_matrix2 = np.full((num_dets, len(lost_track_ids)), invalid_cost, dtype=np.float32)
                for i, (pos, emb) in enumerate(detections):
                    if i not in unmatched_dets:
                        continue
                    for j, tid in enumerate(lost_track_ids):
                        c = self._cost_lost(emb, pos, self.tracks[tid])
                        if np.isfinite(c):
                            cost_matrix2[i, j] = c

                row2, col2 = linear_sum_assignment(cost_matrix2)
                for r, c in zip(row2, col2):
                    if r not in unmatched_dets:
                        continue
                    cost = cost_matrix2[r, c]
                    if not np.isfinite(cost) or cost >= invalid_cost:
                        continue
                    tid = lost_track_ids[c]
                    tr = self.tracks[tid]
                    pos, emb = detections[r]

                    prev_pos = tr.position.copy()
                    tr.embedding = (1 - self.emb_momentum) * tr.embedding + self.emb_momentum * emb
                    tr.position = self.pos_alpha * tr.position + (1 - self.pos_alpha) * pos
                    dt = int(getattr(tr, "time_since_update", 1))
                    if dt < 1:
                        dt = 1
                    delta = (pos - prev_pos) / dt
                    tr.velocity = self.motion_momentum * tr.velocity + (1 - self.motion_momentum) * delta
                    tr.age = 0
                    tr.hits += 1
                    tr.time_since_update = 0
                    tr.state = "ACTIVE"
                    tr.lost_frames = 0

                    assignments[r] = (tid, tr.position.copy(), tr.embedding.copy())
                    unmatched_dets.discard(r)

            for idx in sorted(unmatched_dets):
                pos, emb = detections[idx]
                # Intentar reutilizar un track recientemente borrado
                best_reuse_idx, best_reuse_cost = None, np.inf
                for pool_i, entry in enumerate(self.reuse_pool):
                    c = self._cost_reuse(emb, pos, entry)
                    if c < best_reuse_cost:
                        best_reuse_cost = c
                        best_reuse_idx = pool_i
                if best_reuse_idx is not None and np.isfinite(best_reuse_cost):
                    entry = self.reuse_pool.pop(best_reuse_idx)
                    tid = entry["track_id"]
                else:
                    tid = self.next_id
                    self.next_id += 1
                self.tracks[tid] = Track(tid, pos.copy(), emb.copy())
                assignments[idx] = (tid, pos.copy(), emb.copy())

        # Manejo defensivo de IDs duplicados sin crashear
        frame_ids = [tid for tid, _, _ in assignments]
        seen = set()
        duplicates = []
        for idx, tid in enumerate(frame_ids):
            if tid in seen:
                duplicates.append(tid)
                _, pos_dup, emb_dup = assignments[idx]
                new_tid = self.next_id
                self.next_id += 1
                self.tracks[new_tid] = Track(new_tid, pos_dup.copy(), emb_dup.copy())
                assignments[idx] = (new_tid, pos_dup.copy(), emb_dup.copy())
                frame_ids[idx] = new_tid
            else:
                seen.add(tid)
        if duplicates:
            print(f"[SimpleTracker] Warning: duplicated track IDs {duplicates} en frame; reasignados a nuevos IDs.")

        # eliminar tracks viejos
        to_del = [tid for tid, tr in self.tracks.items() if tr.lost_frames > self.max_lost]
        for tid in to_del:
            tr = self.tracks[tid]
            self.reuse_pool.append({
                "track_id": tid,
                "position": tr.position.copy(),
                "embedding": tr.embedding.copy(),
                "last_seen": self.frame_idx,
            })
            del self.tracks[tid]
        # evita crecimiento indefinido del pool
        if len(self.reuse_pool) > 200:
            self.reuse_pool = self.reuse_pool[-200:]

        return assignments

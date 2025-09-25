# codes/utils/tracking.py
import numpy as np

class Track:
    def __init__(self, track_id, position, embedding):
        self.track_id = track_id
        self.position = position.astype(np.float32)      # pos en cancha (x,y)
        self.embedding = embedding.astype(np.float32)    # vector reid
        self.age = 0            # frames desde el último match
        self.hits = 1           # veces que fue matcheado
        self.time_since_update = 0

class SimpleTracker:
    """
    Asociador simple con memoria:
      - Costo = w_reid * dist_reid + w_pos * (dist_pos / pos_thr)
      - Gating por dist_pos y dist_reid
      - max_age: si no hay match por N frames, se borra
      - EMA en posición y embedding para estabilidad
    """
    def __init__(self,
                 reid_weight=0.7, pos_weight=0.3,
                 reid_threshold=0.8, pos_threshold=220.0,
                 max_age=15, emb_momentum=0.2, pos_alpha=0.4):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.reid_weight = reid_weight
        self.pos_weight = pos_weight
        self.reid_threshold = reid_threshold
        self.pos_threshold = pos_threshold
        self.max_age = max_age
        self.emb_momentum = emb_momentum
        self.pos_alpha = pos_alpha

    def _norm_emb(self, e):
        n = np.linalg.norm(e) + 1e-6
        return e / n

    def _reid_dist(self, e1, e2):
        e1 = self._norm_emb(e1); e2 = self._norm_emb(e2)
        return np.linalg.norm(e1 - e2)

    def _pos_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _cost(self, emb, pos, track: Track):
        d_reid = self._reid_dist(emb, track.embedding)
        d_pos = self._pos_dist(pos, track.position)
        if d_pos > self.pos_threshold or d_reid > self.reid_threshold:
            return np.inf
        return self.reid_weight * d_reid + self.pos_weight * (d_pos / self.pos_threshold)

    def update(self, detections):
        """
        detections: list[(pos (2,), emb (D,))]
        return: list[(track_id, pos, emb)] en el MISMO orden que 'detections'
        """
        # envejecer
        for tr in self.tracks.values():
            tr.age += 1
            tr.time_since_update += 1

        assignments = []

        for pos, emb in detections:
            best_id, best_cost = None, np.inf
            for tid, tr in self.tracks.items():
                c = self._cost(emb, pos, tr)
                if c < best_cost:
                    best_cost = c; best_id = tid

            if best_id is not None and np.isfinite(best_cost):
                tr = self.tracks[best_id]
                # EMA en embedding y posición
                tr.embedding = (1 - self.emb_momentum) * tr.embedding + self.emb_momentum * emb
                tr.position  = self.pos_alpha * tr.position + (1 - self.pos_alpha) * pos
                tr.age = 0
                tr.hits += 1
                tr.time_since_update = 0
                assignments.append((best_id, tr.position.copy(), tr.embedding.copy()))
            else:
                # nuevo track
                tid = self.next_id
                self.tracks[tid] = Track(tid, pos.copy(), emb.copy())
                self.next_id += 1
                assignments.append((tid, pos.copy(), emb.copy()))

        # eliminar tracks viejos
        to_del = [tid for tid, tr in self.tracks.items() if tr.age > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        return assignments

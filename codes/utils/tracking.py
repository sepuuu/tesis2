# utils/tracking.py
import numpy as np

# ---------- Kalman 2D (x,y,vx,vy), medida: (x,y) ----------
class KalmanFilter2D:
    def __init__(self, dt=1.0, std_acc=80.0, std_meas=40.0):
        # Estado: [x, y, vx, vy]^T
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        # Ruido de proceso (a partir de aceleración)
        q = float(std_acc)
        G = np.array([[0.5*dt*dt, 0],
                      [0, 0.5*dt*dt],
                      [dt, 0],
                      [0, dt]], dtype=np.float32)
        Qa = np.diag([q*q, q*q]).astype(np.float32)
        self.Q = G @ Qa @ G.T
        # Ruido de medida
        r = float(std_meas)
        self.R = np.diag([r*r, r*r]).astype(np.float32)
        # Inicialización
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1e4  # incertidumbre alta inicial

    def init_state(self, z_xy):
        x, y = float(z_xy[0]), float(z_xy[1])
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z_xy):
        z = np.array([[float(z_xy[0])], [float(z_xy[1])]], dtype=np.float32)
        y = z - (self.H @ self.x)             # innovación
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def maha(self, z_xy):
        y, S = self.innovation(z_xy)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.inf
        return float(y.T @ Sinv @ y)  # distancia de Mahalanobis^2

    def update(self, z_xy):
        y, S = self.innovation(z_xy)
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # fallback numérico
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    @property
    def pos(self):
        return self.x[0:2, 0]  # (x, y) como array 1D


class Track:
    def __init__(self, track_id, position, embedding,
                 dt=1.0, std_acc=80.0, std_meas=40.0,
                 emb_momentum=0.2):
        self.track_id = track_id
        self.kf = KalmanFilter2D(dt=dt, std_acc=std_acc, std_meas=std_meas)
        self.kf.init_state(position)
        self.embedding = embedding.astype(np.float32)
        self.emb_momentum = emb_momentum

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    @property
    def position(self):
        return self.kf.pos.copy()

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, z_pos, emb):
        # actualización Kalman + EMA de embedding
        self.kf.update(z_pos)
        self.embedding = (1 - self.emb_momentum) * self.embedding + self.emb_momentum * emb
        self.hits += 1
        self.time_since_update = 0
        self.age = 0


class SimpleTracker:
    """
    Asociador global con Kalman + ReID:
      - Predice cada track (x,y,vx,vy) en cancha
      - Gating por Mahalanobis (medida en elipsoide de incertidumbre)
      - Costo = w_reid * dist_reid + w_pos * (maha / maha_thr)
      - max_age: aguanta caídas cortas sin matar el track
      - EMA en embedding
    API:
      update(detections)  # detections: list[(pos(2,), emb(D,))]
      -> list[(track_id, pos(2,), emb(D,))]   (en el mismo orden de 'detections')
    """
    def __init__(self,
                 reid_weight=0.7, pos_weight=0.3,
                 reid_threshold=0.8,
                 mahalanobis_threshold=9.21,  # ~ Chi2(df=2, 99%)
                 max_age=15,
                 emb_momentum=0.2,
                 dt=1.0, std_acc=80.0, std_meas=40.0):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.reid_weight = reid_weight
        self.pos_weight = pos_weight
        self.reid_threshold = reid_threshold
        self.maha_threshold = mahalanobis_threshold
        self.max_age = max_age
        self.emb_momentum = emb_momentum
        self.dt = dt
        self.std_acc = std_acc
        self.std_meas = std_meas

    def _norm_emb(self, e):
        n = np.linalg.norm(e) + 1e-6
        return e / n

    def _reid_dist(self, e1, e2):
        e1 = self._norm_emb(e1); e2 = self._norm_emb(e2)
        return float(np.linalg.norm(e1 - e2))

    def _cost(self, emb, zpos, tr: Track):
        # Gating por apariencia
        d_reid = self._reid_dist(emb, tr.embedding)
        if d_reid > self.reid_threshold:
            return np.inf
        # Gating posicional por Mahalanobis
        d_maha = tr.kf.maha(zpos)
        if not np.isfinite(d_maha) or d_maha > self.maha_threshold:
            return np.inf
        # Costo combinado
        return self.reid_weight * d_reid + self.pos_weight * (d_maha / self.maha_threshold)

    def update(self, detections):
        # 1) predecir todos los tracks
        for tr in self.tracks.values():
            tr.predict()

        assignments = []
        used_tracks = set()

        # 2) asociar greedy detección -> track con menor costo
        for zpos, emb in detections:
            best_id, best_cost = None, np.inf
            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                c = self._cost(emb, zpos, tr)
                if c < best_cost:
                    best_cost, best_id = c, tid

            if best_id is not None and np.isfinite(best_cost):
                tr = self.tracks[best_id]
                tr.update(zpos, emb)
                used_tracks.add(best_id)
                assignments.append((best_id, tr.position, tr.embedding.copy()))
            else:
                # 3) sin match → nuevo track
                tid = self.next_id
                self.tracks[tid] = Track(
                    tid, np.asarray(zpos, dtype=np.float32), np.asarray(emb, dtype=np.float32),
                    dt=self.dt, std_acc=self.std_acc, std_meas=self.std_meas,
                    emb_momentum=self.emb_momentum
                )
                self.next_id += 1
                tr = self.tracks[tid]
                assignments.append((tid, tr.position, tr.embedding.copy()))

        # 4) borrar tracks muy viejos (sin update)
        to_del = [tid for tid, tr in self.tracks.items() if tr.time_since_update > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        return assignments

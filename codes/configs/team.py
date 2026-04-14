# configs/team.py
import os
from collections import defaultdict, Counter, deque
from dataclasses import asdict, is_dataclass

import cv2
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from transformers import AutoProcessor, SiglipVisionModel

try:
    from config import TEAM_CLASSIFIER_CONFIG
except Exception:  # pragma: no cover - en entornos sin config
    TEAM_CLASSIFIER_CONFIG = None


def _cfg_to_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg.copy()
    if is_dataclass(cfg):
        return asdict(cfg)
    attrs = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        value = getattr(cfg, key)
        if not callable(value):
            attrs[key] = value
    return attrs


def _torso_crop(bgr, bbox):
    """
    Recorta el torso: 15%..60% de la altura del bbox. Devuelve (224x224) BGR.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = bgr.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), np.uint8)

    hh = y2 - y1
    yy1 = y1 + int(0.15 * hh)
    yy2 = y1 + int(0.60 * hh)
    yy1 = max(0, min(yy1, h - 1)); yy2 = max(0, min(yy2, h))
    crop = bgr[yy1:yy2, x1:x2]
    if crop.size == 0:
        crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((0, 0, 3), np.uint8)
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)


class TeamClassifier:
    """
    SigLIP + (Lab stats) -> StandardScaler -> UMAP(8) -> KMeans(k=2).

    - fit_from_readers(vr1, vr2, detector): muestrea crops de AMBAS cámaras y entrena un único modelo.
    - predict(frame1, bbox1, frame2=None, bbox2=None, tracker_id=None): predice equipo con 1 o 2 vistas.
    - Anti-parpadeo:
        * voto por ID (ventana deslizante)
        * lock por confianza o mayoría
    - Fallback por color (Lab) si la confianza del cluster es baja.
    """

    def __init__(self, device: str = "cpu", cfg=None):
        params = _cfg_to_dict(cfg if cfg is not None else TEAM_CLASSIFIER_CONFIG)
        self.device = device
        self.use_umap = params.get("use_umap", True)
        self.umap_components = params.get("umap_components", 8)
        self.umap_n_neighbors = params.get("umap_n_neighbors", 15)
        self.umap_min_dist = params.get("umap_min_dist", 0.05)
        self.kmeans_n_init = params.get("kmeans_n_init", 10)

        self.vote_window = params.get("vote_window", 25)
        self.lock_votes = params.get("lock_votes", 8)
        self.tight_lock_p = params.get("tight_lock_p", 0.85)
        self.low_conf_thr = params.get("low_conf_thr", 0.55)

        self.debug_dir = params.get("debug_dir", "debug")
        self.fit_sample_frames = params.get("sample_frames", 180)
        self.fit_per_frame_limit = params.get("per_frame_limit", 8)
        self.fit_imgsz = params.get("imgsz", 1792)
        self.fit_iou = params.get("iou", 0.7)
        os.makedirs(self.debug_dir, exist_ok=True)

        # SigLIP
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(self.device).eval()

        # Preproc
        self.scaler = StandardScaler()
        self.reducer = None
        self.kmeans = None

        # Estado
        self._fitted = False
        self._proj_centers = None              # (2, Dproj)
        self._cluster2team = {}                # {0/1: "equipo_blanco"/"equipo_negro"}
        self._lab_prototypes = {}              # {"equipo_blanco": [L,a,b,...], "equipo_negro": ...}
        self._dist_scale = 1.0                 # escala para convertir distancia -> confianza

        # Memoria anti-parpadeo
        self._votes = defaultdict(lambda: deque(maxlen=self.vote_window))  # rid -> deque[str]
        self._locked = {}  # rid -> team

    # ---------------------------- API pública ---------------------------- #

    def fit_from_readers(
        self,
        vr1,
        vr2,
        detector,
        sample_frames=None,
        per_frame_limit=None,
        imgsz=None,
        iou=None,
        start_frame: int = 0,
    ):
        """
        Muestrea crops de las dos cámaras (primeros `sample_frames` aprox), entrena UMAP+KMeans.
        """
        if sample_frames is None:
            sample_frames = self.fit_sample_frames
        if per_frame_limit is None:
            per_frame_limit = self.fit_per_frame_limit
        if imgsz is None:
            imgsz = self.fit_imgsz
        if iou is None:
            iou = self.fit_iou

        crops_rgb = []
        lab_stats = []

        total = min(len(vr1), len(vr2))
        start = int(max(0, min(total - 1, start_frame)))
        sample_max = min(total, start + sample_frames)
        if sample_max <= start:
            start, sample_max = 0, min(total, sample_frames)
        step = max(1, max(1, (sample_max - start)) // 30)  # ~30 frames muestreados

        for i in range(start, sample_max, step):
            fr1_rgb = vr1[i].asnumpy()
            fr2_rgb = vr2[i].asnumpy()
            fr1 = cv2.cvtColor(fr1_rgb, cv2.COLOR_RGB2BGR)
            fr2 = cv2.cvtColor(fr2_rgb, cv2.COLOR_RGB2BGR)

            # detecciones jugadores
            res1 = detector.predict(fr1, imgsz=imgsz, iou=iou)[0]
            det1 = self._ultra_to_xyxy(res1)
            res2 = detector.predict(fr2, imgsz=imgsz, iou=iou)[0]
            det2 = self._ultra_to_xyxy(res2)

            for bbox in det1[:per_frame_limit]:
                crop = _torso_crop(fr1, bbox)
                if crop.size:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crops_rgb.append(rgb)
                    lab_stats.append(self._lab_feat(crop))

            for bbox in det2[:per_frame_limit]:
                crop = _torso_crop(fr2, bbox)
                if crop.size:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crops_rgb.append(rgb)
                    lab_stats.append(self._lab_feat(crop))

        if len(crops_rgb) < 20:
            print("[Team] WARN: muy pocos crops para entrenar (min ~20). Se usará fallback color.")
            self._fitted = False
            return

        # Embeddings SigLIP
        emb = self._siglip_embed(crops_rgb)  # (N, 768) normalizados
        lab_arr = np.asarray(lab_stats, dtype=np.float32)    # (N, 6)

        feats = np.hstack([emb, lab_arr])                    # (N, 774)
        feats = self.scaler.fit_transform(feats.astype(np.float32))

        if self.use_umap:
            self.reducer = umap.UMAP(
                n_components=self.umap_components,
                metric="cosine",
                random_state=0,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
            )
            proj = self.reducer.fit_transform(feats)
        else:
            self.reducer = None
            proj = feats

        # KMeans (k=2)
        self.kmeans = KMeans(n_clusters=2, n_init=self.kmeans_n_init, random_state=0)
        labels = self.kmeans.fit_predict(proj)

        # Centroides y escala de distancia (mediana de dist a centroides)
        centers = self.kmeans.cluster_centers_  # (2, D)
        self._proj_centers = centers.copy()
        dists = np.linalg.norm(proj - centers[labels], axis=1)
        self._dist_scale = np.median(dists) if np.median(dists) > 1e-6 else (np.mean(dists) + 1e-6)

        # Asignar cluster->equipo por brillo L medio de sus miembros (Lab)
        L_vals = lab_arr[:, 0]
        L0 = float(L_vals[labels == 0].mean()) if np.any(labels == 0) else 0.0
        L1 = float(L_vals[labels == 1].mean()) if np.any(labels == 1) else 0.0
        if L0 >= L1:
            self._cluster2team = {0: "equipo_blanco", 1: "equipo_negro"}
        else:
            self._cluster2team = {1: "equipo_blanco", 0: "equipo_negro"}

        # Prototipos Lab por equipo (media)
        self._lab_prototypes.clear()
        for c in [0, 1]:
            idx = np.where(labels == c)[0]
            if idx.size > 0:
                team = self._cluster2team[c]
                self._lab_prototypes[team] = lab_arr[idx].mean(axis=0)

        self._fitted = True
        self._save_debug_umap(proj, labels)




    @torch.no_grad()
    def predict(
        self,
        frame1_bgr,
        bbox1,
        frame2_bgr=None,
        bbox2=None,
        tracker_id=None,
        return_details=False,
    ):
        """
        Predice 'equipo_blanco' / 'equipo_negro' / 'UNKNOWN'.
        Puede usar 1 o 2 vistas (si pasas frame2+bbox2).
        Con estabilizacion temporal por tracker_id.
        """
        if tracker_id is not None and tracker_id in self._locked:
            locked = self._locked[tracker_id]
            if return_details:
                details = {
                    "raw_team": locked,
                    "locked_team": locked,
                    "confidence": 1.0,
                    "proj_points": [],
                    "selected_view_index": None,
                    "cluster_centers": self._prepare_plot_points(self._proj_centers).tolist()
                    if self._proj_centers is not None
                    else [],
                    "cluster_team_map": {int(k): v for k, v in self._cluster2team.items()},
                    "selected_cluster": None,
                    "distances": [],
                    "labs": [],
                }
                return locked, details
            return locked

        crops = []
        labs = []
        c1 = _torso_crop(frame1_bgr, bbox1)
        if c1.size:
            crops.append(cv2.cvtColor(c1, cv2.COLOR_BGR2RGB))
            labs.append(self._lab_feat(c1))
        if frame2_bgr is not None and bbox2 is not None:
            c2 = _torso_crop(frame2_bgr, bbox2)
            if c2.size:
                crops.append(cv2.cvtColor(c2, cv2.COLOR_BGR2RGB))
                labs.append(self._lab_feat(c2))

        if not crops:
            fallback = {
                "raw_team": "UNKNOWN",
                "locked_team": "UNKNOWN",
                "confidence": 0.0,
                "proj_points": [],
                "selected_view_index": None,
                "cluster_centers": self._prepare_plot_points(self._proj_centers).tolist()
                if self._proj_centers is not None
                else [],
                "cluster_team_map": {int(k): v for k, v in self._cluster2team.items()},
                "selected_cluster": None,
                "distances": [],
                "labs": [],
            }
            return ("UNKNOWN", fallback) if return_details else "UNKNOWN"

        if not self._fitted:
            team = self._fallback_by_lab(np.mean(np.vstack(labs), axis=0))
            locked = self._vote_and_lock(tracker_id, team, conf=0.0)
            details = {
                "raw_team": team,
                "locked_team": locked,
                "confidence": 0.0,
                "proj_points": [],
                "selected_view_index": None,
                "cluster_centers": self._prepare_plot_points(self._proj_centers).tolist()
                if self._proj_centers is not None
                else [],
                "cluster_team_map": {int(k): v for k, v in self._cluster2team.items()},
                "selected_cluster": None,
                "distances": [],
                "labs": [lab.tolist() for lab in labs],
            }
            return (locked, details) if return_details else locked

        emb = self._siglip_embed(crops)
        feats = np.hstack([emb, np.asarray(labs, dtype=np.float32)])
        feats = self.scaler.transform(feats.astype(np.float32))
        if self.use_umap and self.reducer is not None:
            try:
                proj = self.reducer.transform(feats)
            except Exception:
                proj = feats
        else:
            proj = feats

        proj_points = self._prepare_plot_points(proj)
        centers_plot = self._prepare_plot_points(self._proj_centers)

        preds = []
        view_infos = []
        for idx, p in enumerate(proj):
            team_v, conf_v, cluster_v, dist_pair = self._predict_single_proj(p)
            preds.append((team_v, conf_v))
            view_infos.append(
                {
                    "view_index": idx,
                    "team": team_v,
                    "confidence": float(conf_v),
                    "cluster": int(cluster_v),
                    "distances": [float(dist_pair[0]), float(dist_pair[1])],
                    "proj_point": proj_points[idx].tolist() if idx < len(proj_points) else [0.0, 0.0],
                }
            )

        if len(preds) == 1:
            team, conf = preds[0]
            selected_idx = 0
        else:
            (t1, c1_conf), (t2, c2_conf) = preds[0], preds[1]
            if t1 == t2:
                team, conf = (t1, max(c1_conf, c2_conf))
                selected_idx = 0 if c1_conf >= c2_conf else 1
            else:
                if c1_conf > c2_conf + 0.05:
                    team, conf = (t1, c1_conf)
                    selected_idx = 0
                elif c2_conf > c1_conf + 0.05:
                    team, conf = (t2, c2_conf)
                    selected_idx = 1
                else:
                    lab_mean = np.mean(np.asarray(labs, dtype=np.float32), axis=0)
                    team = self._fallback_by_lab(lab_mean)
                    conf = 0.5
                    selected_idx = None

        if conf < self.low_conf_thr:
            lab_mean = np.mean(np.asarray(labs, dtype=np.float32), axis=0)
            team = self._fallback_by_lab(lab_mean)

        locked_team = self._vote_and_lock(tracker_id, team, conf)
        if not return_details:
            return locked_team

        cluster_team_map = {int(k): v for k, v in self._cluster2team.items()}
        selected_cluster = None
        if selected_idx is not None and 0 <= selected_idx < len(view_infos):
            selected_cluster = view_infos[selected_idx]["cluster"]

        details = {
            "raw_team": team,
            "locked_team": locked_team,
            "confidence": float(conf),
            "proj_points": proj_points.tolist(),
            "selected_view_index": selected_idx,
            "cluster_centers": centers_plot.tolist(),
            "cluster_team_map": cluster_team_map,
            "selected_cluster": selected_cluster,
            "distances": view_infos,
            "labs": [lab.tolist() for lab in labs],
        }
        return locked_team, details


    def _prepare_plot_points(self, array):
        if array is None:
            return np.zeros((0, 2), dtype=np.float32)
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] >= 2:
            return arr[:, :2]
        if arr.shape[1] == 0:
            return np.zeros((arr.shape[0], 2), dtype=np.float32)
        padding = np.zeros((arr.shape[0], 2 - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr, padding])

    def _ultra_to_xyxy(self, result):
        """
        result: objeto de Ultralytics; devolvemos una lista de bboxes xyxy de class_id==1 (players).
        """
        try:
            # supervision opcional; evitamos hard-dep para que team.py sea independiente
            # Si tienes supervision, podrías hacer sv.Detections.from_ultralytics y filtrar class==1
            boxes = []
            if hasattr(result, "boxes") and result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy().astype(int)
                for b, c in zip(xyxy, cls):
                    if c == 1:
                        boxes.append(b.astype(np.int32))
            return boxes
        except Exception:
            return []

    @torch.no_grad()
    def _siglip_embed(self, rgb_list):
        """
        rgb_list: lista de imágenes RGB (224x224 ideal pero acepta tamaños arbitrarios).
        Devuelve (N,768) normalizados L2 por fila.
        """
        feats = []
        bs = 16
        for i in range(0, len(rgb_list), bs):
            batch = rgb_list[i:i + bs]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model(**inputs).last_hidden_state.mean(1)  # (B,768)
            v = out.detach().cpu().numpy().astype(np.float32)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            feats.append(v)
        return np.vstack(feats)

    def _lab_feat(self, bgr):
        """
        [L_mean, a_mean, b_mean, L_std, a_std, b_std]
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(lab)
        return np.array(
            [float(L.mean()), float(a.mean()), float(b.mean()),
             float(L.std()),  float(a.std()),  float(b.std())],
            dtype=np.float32
        )




    def _predict_single_proj(self, p):
        """
        p: vector proyectado (Dproj). Devuelve (team, conf 0..1, cluster, distancias)
        """
        if self.kmeans is None or self._proj_centers is None:
            return "UNKNOWN", 0.0, -1, (float("inf"), float("inf"))
        c0 = float(np.linalg.norm(p - self._proj_centers[0]))
        c1 = float(np.linalg.norm(p - self._proj_centers[1]))
        if c0 <= c1:
            cluster = 0
            d = c0
        else:
            cluster = 1
            d = c1
        team = self._cluster2team.get(cluster, "UNKNOWN")
        scale = max(self._dist_scale, 1e-6)
        x = d / scale
        conf = 1.0 / (1.0 + np.exp(2.5 * (x - 1.0)))
        return team, float(conf), cluster, (float(c0), float(c1))

    def project_crop(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0 or not self._fitted:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        emb = self._siglip_embed([rgb])
        lab = self._lab_feat(crop_bgr)[None, :]
        feat = np.hstack([emb, lab]).astype(np.float32)
        feat = self.scaler.transform(feat)
        if self.use_umap and self.reducer is not None:
            try:
                proj = self.reducer.transform(feat)
            except Exception:
                proj = feat
        else:
            proj = feat
        pts = self._prepare_plot_points(proj)
        return pts[0] if len(pts) else None

    def project_bbox(self, frame_bgr, bbox):
        """
        Helper usado por la revisión manual: recorta el torso y lo proyecta al espacio reducido.
        """
        crop = self.torso_crop(frame_bgr, bbox)
        if crop is None or crop.size == 0:
            return None
        return self.project_crop(crop)

    def torso_crop(self, frame_bgr, bbox):
        """
        Devuelve el mismo recorte (224x224 BGR) que usa el clasificador.
        """
        if frame_bgr is None or bbox is None:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        crop = _torso_crop(frame_bgr, bbox)
        return crop if crop is not None else np.zeros((0, 0, 3), dtype=np.uint8)

    def get_cluster_centers_plot(self):
        """
        Devuelve los centroides en 2D para graficar (ya truncados/padded a 2 dims).
        """
        if self._proj_centers is None:
            return None
        return self._prepare_plot_points(self._proj_centers)
    def _fallback_by_lab(self, lab_mean):
        """
        Compara contra prototipos Lab de cada equipo. Si no hay, umbral por L.
        """
        if isinstance(lab_mean, (list, tuple)):
            lab_mean = np.array(lab_mean, dtype=np.float32)
        if self._lab_prototypes:
            best_team, best_d = "UNKNOWN", 1e9
            for t, proto in self._lab_prototypes.items():
                d = float(np.linalg.norm(lab_mean[:3] - proto[:3]))
                if d < best_d:
                    best_d, best_team = d, t
            return best_team
        # umbral por L si no hay prototipos
        L = float(lab_mean[0]) if lab_mean is not None else 128.0
        return "equipo_blanco" if L >= 128.0 else "equipo_negro"

    def _vote_and_lock(self, tracker_id, team, conf):
        """
        - agrega voto por ID
        - si conf>=tight_lock_p -> lock inmediato
        - si mayoría con >= lock_votes -> lock
        """
        if tracker_id is None:
            return team

        # si ya estaba bloqueado
        if tracker_id in self._locked:
            return self._locked[tracker_id]

        # lock por alta confianza
        if conf >= self.tight_lock_p and team != "UNKNOWN":
            self._locked[tracker_id] = team
            return team

        votes = self._votes[tracker_id]
        votes.append(team)
        counts = Counter(votes)
        top_team, top_count = counts.most_common(1)[0]

        if top_count >= self.lock_votes and top_team != "UNKNOWN":
            self._locked[tracker_id] = top_team
            return top_team

        # sin lock aún: devuelve la mayoría actual
        return top_team



    def manual_lock(self, tracker_id, team):
        if tracker_id is None:
            return
        if team not in {"equipo_blanco", "equipo_negro", "UNKNOWN"}:
            raise ValueError(f"Equipo no reconocido para lock manual: {team}")
        self._locked[tracker_id] = team
        if tracker_id in self._votes:
            self._votes.pop(tracker_id, None)

    def _save_debug_umap(self, proj, labels):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            proj2 = proj if proj.shape[1] >= 2 else np.pad(proj, ((0,0),(0,2 - proj.shape[1])), 'edge')[:, :2]
            colors = [cm.Set1(int(l) % 9) for l in labels]
            plt.figure(figsize=(5, 5))
            plt.scatter(proj2[:, 0], proj2[:, 1], c=colors, s=6)
            for c, team in self._cluster2team.items():
                cen = self.kmeans.cluster_centers_[c]
                if proj.shape[1] >= 2:
                    plt.scatter(cen[0], cen[1], c='k', s=40, marker='x')
                plt.text(cen[0], cen[1], team, fontsize=8, color='k')
            plt.title("UMAP + KMeans (teams)")
            os.makedirs(self.debug_dir, exist_ok=True)
            plt.savefig(os.path.join(self.debug_dir, "team_umap_kmeans.png"), dpi=140, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[Team] DEBUG: no se pudo guardar scatter UMAP: {e}")

import os, cv2, torch, numpy as np
from collections import defaultdict, deque, Counter
from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


class TeamClassifier:
    def __init__(self, device: str = "cuda", n_clusters: int = 2, use_umap: bool = True):
        self.device, self.n_clusters, self.use_umap = device, n_clusters, use_umap

        self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        if use_umap:
            self.reducer = umap.UMAP(n_components=3, metric="cosine", random_state=0, n_neighbors=10, min_dist=0.0)

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=512)
        self._cluster2team = defaultdict(lambda: "UNKNOWN")
        self._history: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=25))
        self._stable_team: dict[int, str] = {}
        self._frames_since_change: dict[int, int] = {}
        self._debug_crop_counter = 0  # para guardar los primeros crops

    def fit(self, crops: list[np.ndarray]):
        if len(crops) < self.n_clusters:
            raise ValueError("No hay suficientes crops para entrenar TeamClassifier.")

        emb = self._extract_embeddings(crops)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        proj = self.reducer.fit_transform(emb) if self.use_umap else emb
        self.kmeans.fit(proj)

        counts = np.bincount(self.kmeans.labels_)
        sil = silhouette_score(proj, self.kmeans.labels_, metric="cosine")
        print(f"[DEBUG] silhouette = {sil:.3f}; tamaños = {counts.tolist()}")

        hsv_means = []
        for c in range(self.n_clusters):
            idxs = np.where(self.kmeans.labels_ == c)[0]
            if len(idxs) == 0:
                hsv_means.append(0)
                continue
            v_vals = [cv2.cvtColor(crops[i], cv2.COLOR_RGB2HSV)[..., 2].mean() for i in idxs[:50]]
            hsv_means.append(np.mean(v_vals))

        major_idx = np.argsort(counts)[-2:]
        bright_cluster = int(max(major_idx, key=lambda c: hsv_means[c]))
        for c in major_idx:
            self._cluster2team[c] = "equipo_blanco" if c == bright_cluster else "equipo_negro"

        print(f"[DEBUG] cluster→team: {dict(self._cluster2team)}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs("debug", exist_ok=True)
            proj2d = proj[:, :2]
            plt.figure(figsize=(5, 5))
            plt.scatter(proj2d[:, 0], proj2d[:, 1], c=self.kmeans.labels_, cmap="coolwarm", s=6)
            plt.title("Clusters después de entrenamiento")
            plt.savefig("debug/cluster_scatter.png", dpi=140, bbox_inches="tight")
            plt.close()
            print("[DEBUG] scatter guardado en debug/cluster_scatter.png")
        except Exception as e:
            print(f"[DEBUG] no se pudo guardar scatter: {e}")

    def predict_team(self, frame: np.ndarray, bbox: np.ndarray, tracker_id: int | None = None) -> str:
        if not self._cluster2team:
            return "UNKNOWN"

        crop = self._crop(frame, bbox)
        if crop.size == 0:
            return "UNKNOWN"

        # guardar los primeros crops para depurar
        if self._debug_crop_counter < 50:
            os.makedirs("debug/crops", exist_ok=True)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"debug/crops/crop_{self._debug_crop_counter:03}.jpg", crop_bgr)
            self._debug_crop_counter += 1

        emb = self._extract_embeddings([crop])[0]
        emb /= np.linalg.norm(emb) + 1e-8
        proj = self.reducer.transform([emb]) if self.use_umap else [emb]
        cluster = int(self.kmeans.predict(proj)[0])
        team_pred = self._cluster2team.get(cluster, "UNKNOWN")

        if tracker_id is None:
            return team_pred

        hist = self._history[tracker_id]
        hist.append(team_pred)
        common, freq = Counter(hist).most_common(1)[0]
        ratio = freq / len(hist)
        self._frames_since_change[tracker_id] = self._frames_since_change.get(tracker_id, 0) + 1

        if tracker_id not in self._stable_team:
            if ratio >= 0.8:
                self._stable_team[tracker_id] = common
                self._frames_since_change[tracker_id] = 0
        else:
            current = self._stable_team[tracker_id]
            if common != current and ratio >= 0.8 and self._frames_since_change[tracker_id] >= 30:
                self._stable_team[tracker_id] = common
                self._frames_since_change[tracker_id] = 0

        return self._stable_team.get(tracker_id, team_pred)

    def _crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        y1 = y1 + int(0.15 * h)
        y2 = y1 + int(0.40 * h)
        return frame[y1:y2, x1:x2]

    @torch.no_grad()
    def _extract_embeddings(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 771), dtype=np.float64)
        feats, hsv_feats = [], []
        for i in range(0, len(crops), 32):
            batch = crops[i:i + 32]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model(**inputs).last_hidden_state.mean(1).cpu().numpy()
            feats.append(out)
            hsv_batch = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).mean(0)/[180, 255, 255] for img in batch]
            hsv_feats.append(np.array(hsv_batch))
        emb = np.vstack(feats)
        hsv = np.vstack(hsv_feats)
        return np.hstack([emb, hsv, hsv, hsv]).astype(np.float64)

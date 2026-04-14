from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x) + eps)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2norm(a)
    b = _l2norm(b)
    return float(1.0 - np.dot(a, b))


def _edges_from_bbox_xyxy(
    bbox: Tuple[float, float, float, float],
    w: int,
    h: int,
    margin: int = 20,
) -> Set[str]:
    x1, y1, x2, y2 = bbox
    edges = set()
    if x1 <= margin:
        edges.add("L")
    if x2 >= w - margin:
        edges.add("R")
    if y1 <= margin:
        edges.add("T")
    if y2 >= h - margin:
        edges.add("B")
    return edges


@dataclass
class BankEntry:
    stable_id: int
    emb: np.ndarray
    last_frame: int
    exit_edges: Set[str]
    last_bbox: Tuple[float, float, float, float]


class StableIDAssigner:
    """
    Mantiene stable_id por cámara encima del track_id del tracker monocámara.
    - Si un track_id muere y luego aparece uno nuevo parecido (ReID + borde + ventana temporal),
      se reusa el stable_id.
    """

    def __init__(
        self,
        max_gap_frames: int = 300,  # cuanto gap toleras para reattach
        reattach_cos_thr: float = 0.25,  # cosine distance (menor es mejor)
        emb_momentum: float = 0.15,  # EMA del embedding del stable_id
        edge_bonus: float = 0.04,  # reduce el costo si coincide borde
        bank_size: int = 200,
        edge_margin: int = 20,
        ds_dead_after: int = 0,  # frames sin "alive" para considerar ds_id muerto (0 = inmediato)
    ):
        self.max_gap_frames = max_gap_frames
        self.reattach_cos_thr = reattach_cos_thr
        self.emb_momentum = emb_momentum
        self.edge_bonus = edge_bonus
        self.bank_size = bank_size
        self.edge_margin = edge_margin
        self.ds_dead_after = max(0, int(ds_dead_after))

        self.ds_to_stable: Dict[int, int] = {}
        self.ds_last_seen: Dict[int, int] = {}
        self.stable_emb: Dict[int, np.ndarray] = {}
        self.stable_first_seen: Dict[int, int] = {}
        self.stable_last_seen: Dict[int, int] = {}
        self.stable_obs_count: Dict[int, int] = {}
        self.stable_emb_obs_count: Dict[int, int] = {}
        self.stable_last_bbox: Dict[int, Tuple[float, float, float, float]] = {}

        self.bank: List[BankEntry] = []
        self.next_stable_id = 1

    def _new_stable(
        self,
        emb: Optional[np.ndarray],
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
    ) -> int:
        sid = self.next_stable_id
        self.next_stable_id += 1
        if emb is not None:
            self.stable_emb[sid] = _l2norm(emb)
        self.stable_first_seen[sid] = frame_idx
        self.stable_last_seen[sid] = frame_idx
        self.stable_obs_count[sid] = 0
        self.stable_emb_obs_count[sid] = 0
        self.stable_last_bbox[sid] = bbox
        return sid

    def _update_stable(
        self,
        sid: int,
        emb: Optional[np.ndarray],
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        if sid not in self.stable_first_seen:
            self.stable_first_seen[sid] = frame_idx
        if emb is not None:
            if sid not in self.stable_emb:
                self.stable_emb[sid] = _l2norm(emb)
            else:
                cur = self.stable_emb[sid]
                new = _l2norm(emb)
                self.stable_emb[sid] = _l2norm(
                    (1.0 - self.emb_momentum) * cur + self.emb_momentum * new
                )
        self.stable_last_seen[sid] = frame_idx
        self.stable_obs_count[sid] = int(self.stable_obs_count.get(sid, 0)) + 1
        if emb is not None:
            self.stable_emb_obs_count[sid] = int(self.stable_emb_obs_count.get(sid, 0)) + 1
        self.stable_last_bbox[sid] = bbox

    def update(
        self,
        frame_idx: int,
        alive_ds_ids: Set[int],
        observed: List[dict],
        frame_wh: Tuple[int, int],
    ) -> List[dict]:
        """
        observed: lista de dicts, cada uno con:
          - ds_id (int)
          - bbox (xyxy tuple)
          - emb (np.ndarray | None)
          - conf (float)  (opcional)
        Retorna la misma lista con 'stable_id' agregado.
        """
        w, h = frame_wh

        # registrar "alive" (para death delay)
        for ds in alive_ds_ids:
            self.ds_last_seen[int(ds)] = frame_idx

        # 1) tracks que ya no están vivos en el tracker => mover a bank (si hay emb)
        dead_ds = []
        for ds in list(self.ds_to_stable.keys()):
            if ds in alive_ds_ids:
                continue
            last = self.ds_last_seen.get(ds)
            if last is None:
                # si no sabemos cuándo se vio, no lo matamos en el mismo frame
                self.ds_last_seen[ds] = frame_idx
                continue
            if (self.ds_dead_after <= 0 and ds not in alive_ds_ids) or (
                self.ds_dead_after > 0 and (frame_idx - int(last)) >= self.ds_dead_after
            ):
                dead_ds.append(ds)
        for ds in dead_ds:
            sid = self.ds_to_stable.pop(ds)
            self.ds_last_seen.pop(ds, None)
            last_frame = self.stable_last_seen.get(sid, frame_idx)
            last_bbox = self.stable_last_bbox.get(sid, (0.0, 0.0, 0.0, 0.0))
            exit_edges = _edges_from_bbox_xyxy(last_bbox, w, h, self.edge_margin)
            emb = self.stable_emb.get(sid)
            if emb is not None:
                # Evita colisiones: un stable_id no puede quedar duplicado en el bank
                self.bank = [be for be in self.bank if be.stable_id != sid]
                self.bank.append(BankEntry(sid, emb.copy(), last_frame, exit_edges, last_bbox))

        if len(self.bank) > self.bank_size:
            self.bank = self.bank[-self.bank_size :]

        # Conjunto de stable_ids activos este frame
        active_sids = set(self.ds_to_stable.values())

        # 2) asignar stable_id a observados
        for o in observed:
            ds_id = int(o["ds_id"])
            bbox = tuple(map(float, o["bbox"]))
            emb = o.get("emb")
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)

            # este ds_id fue observado este frame
            self.ds_last_seen[ds_id] = frame_idx

            if emb is None:
                if ds_id in self.ds_to_stable:
                    sid = self.ds_to_stable[ds_id]
                else:
                    sid = self._new_stable(None, frame_idx, bbox)
                    self.ds_to_stable[ds_id] = sid
                active_sids.add(sid)

                self._update_stable(sid, None, frame_idx, bbox)
                o["stable_id"] = sid
                continue

            if ds_id in self.ds_to_stable:
                sid = self.ds_to_stable[ds_id]
                self._update_stable(sid, emb, frame_idx, bbox)
                o["stable_id"] = sid
                continue

            sid = None
            if emb is not None:
                # intento de reattach desde bank
                entry_edges = _edges_from_bbox_xyxy(bbox, w, h, self.edge_margin)
                best_idx = None
                best_cost = 1e9

                for j, be in enumerate(self.bank):
                    # Nunca reutilizar un stable_id que ya está activo
                    if be.stable_id in active_sids:
                        continue
                    gap = frame_idx - be.last_frame
                    if gap <= 0 or gap > self.max_gap_frames:
                        continue

                    d = _cosine_dist(emb, be.emb)

                    # bonus si coincide borde salida/entrada (si hay info)
                    if (
                        be.exit_edges
                        and entry_edges
                        and (len(be.exit_edges.intersection(entry_edges)) > 0)
                    ):
                        d = d - self.edge_bonus

                    if d < best_cost:
                        best_cost = d
                        best_idx = j

                if best_idx is not None and best_cost <= self.reattach_cos_thr:
                    sid = self.bank.pop(best_idx).stable_id
                    # limpiar duplicados remanentes
                    self.bank = [be for be in self.bank if be.stable_id != sid]
                    if sid in active_sids:
                        sid = None

            if sid is None:
                sid = self._new_stable(emb, frame_idx, bbox)

            self.ds_to_stable[ds_id] = sid
            active_sids.add(sid)
            self._update_stable(sid, emb, frame_idx, bbox)
            o["stable_id"] = sid

        return observed

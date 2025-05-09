import numpy as np

class Track:
    def __init__(self, track_id, position, embedding):
        self.track_id = track_id
        self.position = position
        self.embedding = embedding
        self.age = 0

    def update(self, position, embedding):
        self.position = position
        self.embedding = embedding
        self.age = 0

class SimpleTracker:
    def __init__(self, reid_weight=0.7, pos_weight=0.3, reid_threshold=0.8, pos_threshold=150):
        self.tracks = {}  # track_id: Track
        self.next_id = 1
        self.reid_weight = reid_weight
        self.pos_weight = pos_weight
        self.reid_threshold = reid_threshold
        self.pos_threshold = pos_threshold

    def _normalized_distance(self, emb1, emb2):
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
        return np.linalg.norm(emb1 - emb2)

    def match(self, position, embedding):
        best_id = None
        best_score = float('inf')

        for track_id, track in self.tracks.items():
            dist_visual = self._normalized_distance(embedding, track.embedding)
            dist_pos = np.linalg.norm(position - track.position)
            if dist_pos > self.pos_threshold:
                continue

            score = self.reid_weight * dist_visual + self.pos_weight * (dist_pos / self.pos_threshold)
            if score < best_score and dist_visual < self.reid_threshold:
                best_id = track_id
                best_score = score

        return best_id

    def update(self, detections):
        """
        detections: list of tuples (position, embedding)
        returns: list of tuples (track_id, position, embedding)
        """
        assignments = []
        for position, embedding in detections:
            match_id = self.match(position, embedding)
            if match_id is not None:
                self.tracks[match_id].update(position, embedding)
            else:
                match_id = self.next_id
                self.tracks[match_id] = Track(match_id, position, embedding)
                self.next_id += 1
            assignments.append((match_id, position, embedding))
        return assignments

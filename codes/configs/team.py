import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from typing import List

class TeamClassifier:
    def __init__(self, team_names=("equipo_negro", "equipo_blanco")):
        self.team_names = team_names
        self.kmeans = MiniBatchKMeans(n_clusters=2, init='k-means++', n_init=5)
        self.scaler = StandardScaler()
        self.training_data = []
        
    def _preprocess_jersey(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        
        # Zona ajustada para evitar el pelo (30%-50% de la altura del jugador)
        jersey_y_start = y1 + int(height * 0.30)
        jersey_y_end = y1 + int(height * 0.50)
        
        # Margen horizontal para eliminar bordes
        horizontal_margin = int((x2 - x1) * 0.15)
        jersey_region = frame[
            max(0, jersey_y_start):jersey_y_end,
            max(x1 + horizontal_margin, 0):min(x2 - horizontal_margin, frame.shape[1])
        ]
        
        if jersey_region.size == 0: 
            return None
            
        # Usar canales Hue y Saturation para mejor discriminación
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [90, 128], [0, 180, 0, 256])
        return hist.flatten().astype(np.float64)
    
    def add_training_data(self, frame: np.ndarray, detections: list):
        for bbox in detections:
            features = self._preprocess_jersey(frame, bbox)
            if features is not None and features.mean() > 10:  # Filtrar áreas oscuras
                self.training_data.append(features)
                
    def train(self):
        if len(self.training_data) >= 8:  # Mínimo 4 ejemplos por equipo
            data = np.array(self.training_data)
            self.scaler.fit(data)
            self.kmeans.fit(self.scaler.transform(data))
            
    def predict_team(self, frame: np.ndarray, bbox: list) -> str:
        features = self._preprocess_jersey(frame, bbox)
        if features is None or features.mean() < 15: 
            return "UNKNOWN"
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        return self.team_names[self.kmeans.predict(scaled_features)[0]]
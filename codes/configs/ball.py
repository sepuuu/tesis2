from collections import deque
import cv2
import numpy as np
import supervision as sv

class BallBuffer:
    """Clase base para manejo eficiente de buffers de posiciones."""
    
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)
    
    def update_buffer(self, positions: np.ndarray):
        """Actualiza el buffer con nuevas posiciones."""
        self.buffer.append(positions)
    
    def get_centroid(self) -> np.ndarray:
        """Calcula el centroide de todas las posiciones en el buffer."""
        if not self.buffer:
            return np.array([])
        return np.mean(np.concatenate(self.buffer), axis=0)

class BallTracker(BallBuffer):
    """Tracker optimizado que hereda de BallBuffer."""
    
    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """Filtra detecciones basado en el centroide histórico."""
        if detections.empty:
            return detections
        
        positions = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.update_buffer(positions)
        
        centroid = self.get_centroid()
        if centroid.size == 0:
            return detections
        
        distances = np.linalg.norm(positions - centroid, axis=1)
        return detections[[np.argmin(distances)]]

class BallAnnotator(BallBuffer):
    """Anotador vectorizado con gestión de colores optimizada."""
    
    def __init__(self, max_radius: int, buffer_size: int, thickness: int = 2):
        super().__init__(buffer_size)
        self.max_radius = max_radius
        self.thickness = thickness
        self.color_palette = sv.ColorPalette.from_hex(["#FF6B6B", "#4ECDC4", "#45B7D1"])
    
    def draw_trail(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja la trayectoria del balón usando operaciones vectorizadas."""
        if not self.buffer:
            return frame
            
        positions = np.concatenate(self.buffer)
        radii = np.linspace(5, self.max_radius, len(positions)).astype(int)
        
        for idx, (x, y) in enumerate(positions):
            cv2.circle(
                img=frame,
                center=(int(x), int(y)),
                radius=radii[idx],
                color=self.color_palette.by_idx(idx).as_bgr(),
                thickness=self.thickness
            )
        return frame
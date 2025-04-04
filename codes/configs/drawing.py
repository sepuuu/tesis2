import numpy as np
import cv2
from scipy.spatial import Voronoi
from typing import List, Tuple, Dict, Optional
from configs.soccer import SoccerPitchConfiguration
import supervision as sv

class PitchRenderer:

    
    def __init__(
        self,
        config: SoccerPitchConfiguration,
        scale: float = 0.1,
        padding: int = 50,
        background_color: sv.Color = sv.Color(34, 139, 34),
        line_color: sv.Color = sv.Color.WHITE
    ):
        self.config = config
        self.scale = scale
        self.padding = padding
        self.background_color = background_color.as_bgr()
        self.line_color = line_color.as_bgr()
        
        # Calcular dimensiones escaladas
        self.scaled_width = int(config.width * scale)
        self.scaled_length = int(config.length * scale)
        self.base_pitch = self._create_base_pitch()

    def _create_base_pitch(self) -> np.ndarray:

        pitch = np.full(
            (self.scaled_width + 2*self.padding, 
             self.scaled_length + 2*self.padding, 3),
            self.background_color,
            dtype=np.uint8
        )
        
        # Dibujar líneas principales
        for start, end in self.config.edges:
            pt1 = self._scale_point(self.config.vertices[start - 1])
            pt2 = self._scale_point(self.config.vertices[end - 1])
            cv2.line(pitch, pt1, pt2, self.line_color, 2)
        
        # Dibujar círculo central
        center = self._scale_point([self.config.length/2, self.config.width/2])
        cv2.circle(pitch, center, int(self.config.centre_circle_radius * self.scale), 
                  self.line_color, 2)
        
        return pitch

    def _scale_point(self, point: List[float]) -> Tuple[int, int]:
        return (
            int(point[0] * self.scale) + self.padding,
            int(point[1] * self.scale) + self.padding
        )

    def draw(
        self,
        elements: Optional[Dict] = None
    ) -> np.ndarray:
 
        output = self.base_pitch.copy()
        
        if not elements:
            return output
            
        # Dibujar puntos
        for x, y, color in elements.get('points', []):
            cv2.circle(output, self._scale_point([x, y]), 8, color, -1)
        
        # Dibujar trayectorias
        for path, color, thickness in elements.get('paths', []):
            scaled_path = [self._scale_point(p) for p in path]
            cv2.polylines(output, [np.array(scaled_path)], False, color, thickness)
        
        # Dibujar Voronoi (optimizado con SciPy)
        if 'voronoi' in elements:
            output = self._draw_voronoi(output, *elements['voronoi'])
        
        return output

    def _draw_voronoi(
        self,
        pitch: np.ndarray,
        team1_points: np.ndarray,
        team2_points: np.ndarray,
        colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    ) -> np.ndarray:
        all_points = np.vstack([team1_points, team2_points])
        vor = Voronoi(all_points)
        
        # Crear máscara de colores
        voronoi_layer = np.zeros_like(pitch)
        for i, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if -1 not in vertices:
                polygon = [vor.vertices[v] for v in vertices]
                color = colors[0] if i < len(team1_points) else colors[1]
                cv2.fillPoly(voronoi_layer, [np.array(polygon, dtype=int)], color)
        
        # Combinar con transparencia
        return cv2.addWeighted(voronoi_layer, 0.3, pitch, 0.7, 0)
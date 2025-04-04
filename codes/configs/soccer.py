from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SoccerPitchConfiguration:
    width: int = 2000  # [cm] Ancho de la cancha de futbolito
    length: int = 4000  # [cm] Largo de la cancha de futbolito
    penalty_box_width: int = 1000  # [cm] Ancho del área de penalti
    penalty_box_length: int = 500  # [cm] Largo del área de penalti
    goal_box_width: int = 0  # [cm] No tiene área de meta
    goal_box_length: int = 0  # [cm] No tiene área de meta
    centre_circle_radius: int = 0  # [cm] No hay círculo central
    penalty_spot_distance: int = 0  # [cm] No hay punto de penalti específico

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # Ajustar las posiciones de los puntos basados en la imagen proporcionada
        return [
            (self.length / 2, self.width / 2),  # 1 - Centro de la cancha
            (0, 0),  # 2 - Esquina inferior izquierda
            (0, 500),  # 3 - División del área grande
            (0, 800),  # 4 - Palo del arco izquierdo (ajustado)
            (0, 1200),  # 5 - Palo del arco derecho (ajustado)
            (0, 1500),  # 6 - División del área grande
            (0, self.width),  # 7 - Esquina superior izquierda
            (500, 500),  # 8 - Esquina del área
            (500, 1500),  # 9 - Esquina del área
            (self.length, 0),  # 10 - Esquina inferior derecha
            (self.length, 500),  # 11 - Conexión con el área derecha (ajustado)
            (self.length, 800),  # 12 - Lado derecho del arco (ajustado)
            (self.length, 1200),  # 13 - Lado izquierdo del arco (ajustado)
            (self.length, 1500),  # 14 - Conexión con el área derecha (ajustado)
            (self.length, self.width),  # 15 - Esquina superior derecha
            (self.length - 500, 500),  # 16 - Punto de referencia inferior (ajustado)
            (self.length - 500, 1500)   # 17 - Punto de referencia superior (ajustado)
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1,1), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
     (6, 9), (3, 8), (8, 9), (10, 11), (11, 12),
        (12, 13), (13, 14), (14, 15), (17, 14), (16, 11), (16, 17), (7, 15), (2,10), 
    ])

    labels: List[str] = field(default_factory=lambda: [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
        "12", "13", "14", "15", "16", "17"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", 
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", 
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", 
        "#FF1493", "#FF1493"
    ])
import os
import numpy as np
import torchreid
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from configs.soccer import SoccerPitchConfiguration
from configs.view_transformer import ViewTransformer
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import supervision as sv
import numpy as np
import cv2
import torch
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import PitchRenderer  # Asegúrate de que el archivo se llame drawing.py####################################
from configs.view_transformer import ViewTransformer
import torchreid
from torchvision import transforms
from configs.ball import BallTracker, BallAnnotator 
from configs.team import TeamClassifier

os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
os.environ["PATH"] += os.pathsep + os.environ["CUDA_PATH"] + "\\bin"

device ='cuda'

reid_model = torchreid.models.build_model(name='resnet50', num_classes=12, pretrained=False)
checkpoint = torch.load('codes\models\model.pth.tar-300', map_location=device)
reid_model.load_state_dict(checkpoint['state_dict'])
reid_model = reid_model.to(device)
reid_model.eval()

team_classifier = TeamClassifier()
team_decision_frames = 30  # Aumentar frames de entrenamiento

# Ruta del modelo y del video
PLAYER_DETECTION_MODEL = YOLO("codes/models/players.onnx")
BALL_DETECTION_MODEL = YOLO("codes/models/ball.onnx")

VIDEO_PATH = "codes/videos/prueba2.mp4"
TARGET_VIDEO_PATH = "codes/videos/Pruebas-output2.mov"

# Configuración de la cancha y homografía
CONFIG = SoccerPitchConfiguration()

# Configuración del renderizador del campo
pitch_renderer = PitchRenderer(
    config=CONFIG,
    scale=0.1,                # Mismo valor que usabas antes
    padding=50,
    background_color=sv.Color(34, 139, 34),  # Verde del campo
    line_color=sv.Color.WHITE
)

# Video (Decord con GPU)

vr = VideoReader(VIDEO_PATH, ctx=cpu(0))  # Usar Decord con GPU
total_frames = len(vr)

# Preprocesamiento para re-identificación
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


tracked_embeddings = {}
embedding_buffers = {}
player_tracks = {}  # Para predicción de movimiento (ID -> posición previa)
buffer_size = 10
next_id = 1
player_teams = {}
initial_positions = {}
team_decision_frames = 20# Frames para asignar equipos

# Configuración de transformación de vista
points_image = np.array([[891, 273], [905, 207], [1001, 217], [1057, 223], [1178, 240], [1232, 248], [957, 221], [1216, 256], [586, 518]], dtype=np.float32)
points_pitch = np.array([[2000, 1000], [0, 0], [0, 500], [0, 800], [0, 1200], [0, 1500], [500, 500], [500, 1500], [3500, 1500]], dtype=np.float32)
view_transformer = ViewTransformer(source=points_image, target=points_pitch)

# Función para extraer embeddings (GPU)
def extract_embedding(frame, bbox, model, preprocess, device):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(crop).to(device).cpu().numpy()[0]  # Procesar en GPU, devolver en CPU
    return embedding

# Función para encontrar coincidencias en re-identificación (GPU)
def find_best_match(embedding, tracked_embeddings, threshold=0.8):
    embedding = embedding / np.linalg.norm(embedding)
    best_match = None
    best_score = threshold
    for track_id, track_emb in tracked_embeddings.items():
        track_emb = track_emb / np.linalg.norm(track_emb)
        similarity = np.linalg.norm(embedding - track_emb)
        if similarity < best_score:
            best_match = track_id
            best_score = similarity
    return best_match

def get_video_frames_generator(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))  # Inicializar Decord VideoReader con CPU
    for frame in vr:
        yield frame.asnumpy()  # Convertir el frame a array de NumPy y devolverlo



# Función para dibujar caja y texto
def draw_player_box(frame, bbox, player_id, team_color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
    cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)


# Configuración del slicer para mejorar la detección de balón
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = BALL_DETECTION_MODEL(image_slice, imgsz=1024, conf=0.71)[0]
    return sv.Detections.from_ultralytics(result)

ball_tracker = BallTracker(buffer_size=20)
ball_annotator = BallAnnotator(max_radius=15, buffer_size=20)

# Función para determinar equipo basado en posición
def determine_team(x_position):
    return "equipo_negro" if x_position < 1950 else "equipo_blanco"

# Función para dibujar caja y texto
def draw_player_box(frame, bbox, player_id, team_color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
    cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

def draw_box(frame, bbox, label, color=(255, 0, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Configuración de Decord para leer el video
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
frame_width, frame_height = vr[0].shape[1], vr[0].shape[0]
fps = vr.get_avg_fps()  # Reemplaza tu línea "fps = 59"
total_frames = len(vr)


# Configuración de VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec AVI
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))


# Configuración del radar
radar_width = int(frame_width * 0.4)  # Ancho del radar: 40% del ancho del frame
radar_height = int(frame_height * 0.3)  # Alto del radar: 30% del alto del frame
radar_position = (int((frame_width - radar_width) / 2), int(frame_height - radar_height - 20))  # Posición centrada abajo

# Inicializar contador de frames
i = 0

# Inicializar DataFrame para almacenar las posiciones
posiciones_df = pd.DataFrame(columns=['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y', 'Team'])

# Inicializar contador de frames
last_ball_position = [np.nan, np.nan]  # Usar NaN en lugar de None
for i, frame in tqdm(enumerate(vr), total=total_frames):
    frame = frame.asnumpy()
    frame_ball = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 1. Detección del balón ---
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,  # ❌ Deshabilitar completamente
        overlap_wh=(0, 0),      # ✅ Sin overlap (0 píxeles)
        overlap_filter=sv.OverlapFilter.NONE  # Sin fusión de detecciones
    )
    detections_result_ball = slicer(frame_ball).with_nms(threshold=0.05)
    detections_result_ball = ball_tracker.filter_detections(detections_result_ball)
    ball_detections = detections_result_ball[detections_result_ball.class_id == 0]

    pitch_ball_xy1 = []  # Inicializar posición proyectada del balón
    if len(ball_detections.xyxy) > 0:
        # Tomar la detección más confiable del balón
        confidences = ball_detections.confidence
        best_idx = np.argmax(confidences)
        ball_detections.xyxy = ball_detections.xyxy[best_idx:best_idx+1]
        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy1 = view_transformer.transform_points(frame_ball_xy)
        if len(pitch_ball_xy1) > 0:
            last_ball_position = [pitch_ball_xy1[0, 0], pitch_ball_xy1[0, 1]]
        else:
            last_ball_position = [np.nan, np.nan]  # Mantener NaN si no hay detección
    else:
        pitch_ball_xy1 = np.empty((0, 2))
        last_ball_position = [np.nan, np.nan]        

    # Si no hay detección válida, mantener la última posición conocida
    ball_x, ball_y = last_ball_position  # Ahora son NaN si no hay detección

    # --- 2. Detección de jugadores ---
    result = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792, iou=0.7)[0]
    detections_players = sv.Detections.from_ultralytics(result)
    players_detections = detections_players[detections_players.class_id == 1]  # Solo jugadores
    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = view_transformer.transform_points(np.array(frame_players_xy)) if frame_players_xy is not None else []


    # Diccionario de posiciones de jugadores
    players_positions = {}
    detected_player_ids = []

    # --- 3. Reidentificación de jugadores ---
    for bbox, player_position, pitch_player_position in zip(players_detections.xyxy, frame_players_xy, pitch_players_xy):
        # Aplicar reidentificación únicamente a los jugadores
        embedding = extract_embedding(frame, bbox, reid_model, preprocess, device)
        if embedding is None:
            continue
        match_id = find_best_match(embedding, tracked_embeddings)
        if match_id is None:
            match_id = next_id
            next_id += 1
            tracked_embeddings[match_id] = embedding
        players_positions[match_id] = pitch_player_position
        detected_player_ids.append(match_id)

        if i < team_decision_frames:
            if match_id not in initial_positions:
                initial_positions[match_id] = []
            initial_positions[match_id].append(pitch_player_position[0])

    # --- 4. Asignar equipos ---
    if i < team_decision_frames:
        team_classifier.add_training_data(frame, players_detections.xyxy)
        # Guardar cortes para depuración
        if i < team_decision_frames:
            for idx, bbox in enumerate(players_detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.imwrite(f"debug/{i}_{idx}.jpg", frame[y1:y2, x1:x2])

    if i == team_decision_frames - 1:
        team_classifier.train()    
    # --- 5. Dibujar cajas y texto según el equipo ---
    for bbox, match_id in zip(players_detections.xyxy, detected_player_ids):
        team = team_classifier.predict_team(frame, bbox) if i >= team_decision_frames else "UNKNOWN"
        player_teams[match_id] = team
        team_color = (0, 0, 0) if team == "equipo_negro" else (255, 255, 255)
        draw_player_box(frame, bbox, match_id, team_color)

    # --- 6. Dibujar la caja delimitadora del balón con confianza ---
    if len(ball_detections.xyxy) > 0:
        for bbox, confidence in zip(ball_detections.xyxy, ball_detections.confidence):
            label = f"Ball {confidence:.2f}"  # Etiqueta con confianza
            draw_box(frame, bbox, label, color=(0, 255, 255))  # Amarillo para el balón

    # --- 7. Capturar posiciones en el DataFrame ---
    rows = []
    for match_id, player_position in players_positions.items():
        team = player_teams.get(match_id, "UNKNOWN")
        rows.append({
            'Frame': i,
            'Id': match_id,
            'Pos X': player_position[0],
            'Pos Y': player_position[1],
            'Ball X': ball_x,
            'Ball Y': ball_y,
            'Team': team
        })
    posiciones_df = pd.concat([posiciones_df, pd.DataFrame(rows)], ignore_index=True)   



    # --- 8. Dibujar radar táctico ---
    elements = {
        'points': [],
        'paths': []
    }

    # Añadir jugadores al radar
    for pid, pos in players_positions.items():
        color = (0, 0, 0) if player_teams.get(pid) == "equipo_negro" else (255, 255, 255)
        elements['points'].append((pos[0], pos[1], color))  # (x, y, color BGR)

    # Añadir balón al radar (si hay posición)
    if not np.isnan(ball_x) and not np.isnan(ball_y):  # Ahora funciona con NaN
        elements['points'].append((ball_x, ball_y, (0, 255, 255)))

    # Opcional: Añadir diagrama de Voronoi (requiere team1_points y team2_points)
    # elements['voronoi'] = (
    #     team1_positions_array,  # np.array de forma (N, 2)
    #     team2_positions_array,  # np.array de forma (M, 2)
    #     ((0, 0, 128), (128, 128, 255))  # Colores BGR para cada equipo
    # )

    # Generar imagen del radar
    radar_image = pitch_renderer.draw(elements)
    

    # Redimensionar al tamaño del radar (ajusta según tu configuración)
    radar_resized = cv2.resize(radar_image, (radar_width, radar_height))

    # --- 9. Crear un frame combinado con radar ---
    frame_with_radar = frame.copy()
    x, y = radar_position
    alpha = 0.6
    for c in range(3):  # Mezcla de canales (RGB)
        frame_with_radar[y:y+radar_height, x:x+radar_width, c] = (
            alpha * radar_resized[:, :, c] + 
            (1 - alpha) * frame_with_radar[y:y+radar_height, x:x+radar_width, c]
        )

    # Convertir a BGR y escribir (¡solo una vez!)
    frame_with_radar = cv2.cvtColor(frame_with_radar, cv2.COLOR_RGB2BGR)
    out.write(frame_with_radar) 
 

    #Guardar el DataFrame en un archivo Excel

input_file = "Posiciones-jugadores-balon.xlsx"
posiciones_df.to_excel(input_file, index=False)


#Limpieza de dataframe

from postprocess import process_file

# Ejecución del flujo completo
process_file(
    file_path=input_file,
    cleaned_output_path='limpieza.xlsx',
    output_possession_path='posesion.xlsx',
    output_passes_path='pases.xlsx',
    output_team_passes_path='passes_by_{team}.xlsx'
)
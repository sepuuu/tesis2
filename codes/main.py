import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"          # fuerza a ignorar TensorRT
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"  # por si acaso
import torchreid
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from configs.soccer import SoccerPitchConfiguration
from decord import VideoReader, cpu
from torchvision import transforms
import supervision as sv
import numpy as np
import cv2
import torch
from configs.drawing import PitchRenderer
from configs.view_transformer import ViewTransformer
from configs.team import TeamClassifier
from utils.drawing_utils import draw_player_box, draw_box
from utils.ball_setup import callback, ball_tracker
from utils.train_team_classifier import train_team_classifier

device = "cuda"
team_classifier = TeamClassifier(device=device)

reid_model = torchreid.models.build_model(name='resnet50', num_classes=12, pretrained=False)
checkpoint = torch.load('codes\models\model.pth.tar-300', map_location=device)
reid_model.load_state_dict(checkpoint['state_dict'])
reid_model = reid_model.to(device)
reid_model.eval()

PLAYER_DETECTION_MODEL = YOLO("codes\models\players.onnx")
VIDEO_PATH = "codes\inputs\prueba2.mp4"
TARGET_VIDEO_PATH = "codes/outputs/Pruebas-output2.mov"

CONFIG = SoccerPitchConfiguration()
pitch_renderer = PitchRenderer(
    config=CONFIG,
    scale=0.1,
    padding=50,
    background_color=sv.Color(34, 139, 34),
    line_color=sv.Color.WHITE
)

vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tracked_embeddings = {}
embedding_buffers = {}
player_tracks = {}
buffer_size = 10
next_id = 1
player_teams = {}

points_image = np.array([[891, 273], [905, 207], [1001, 217], [1057, 223], [1178, 240], [1232, 248], [957, 221], [1216, 256], [586, 518]], dtype=np.float32)
points_pitch = np.array([[2000, 1000], [0, 0], [0, 500], [0, 800], [0, 1200], [0, 1500], [500, 500], [500, 1500], [3500, 1500]], dtype=np.float32)
view_transformer = ViewTransformer(source=points_image, target=points_pitch)

def extract_embedding(frame, bbox, model, preprocess, device):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(crop).to(device).cpu().numpy()[0]
    return embedding

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

fps = vr.get_avg_fps()
train_team_classifier(VIDEO_PATH, team_classifier, fps)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (vr[0].shape[1], vr[0].shape[0]))

radar_width = int(vr[0].shape[1] * 0.4)
radar_height = int(vr[0].shape[0] * 0.3)
radar_position = (int((vr[0].shape[1] - radar_width) / 2), int(vr[0].shape[0] - radar_height - 20))

posiciones_df = pd.DataFrame(columns=['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y', 'Team'])
last_ball_position = [np.nan, np.nan]

for i, frame in tqdm(enumerate(vr), total=total_frames):
    frame = frame.asnumpy()
    frame_ball = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección del balón
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,
        overlap_wh=(0, 0),
        overlap_filter=sv.OverlapFilter.NONE
    )
    detections_result_ball = slicer(frame_ball).with_nms(threshold=0.05)
    detections_result_ball = ball_tracker.filter_detections(detections_result_ball)
    ball_detections = detections_result_ball[detections_result_ball.class_id == 0]

    pitch_ball_xy1 = []
    if len(ball_detections.xyxy) > 0:
        confidences = ball_detections.confidence
        best_idx = np.argmax(confidences)
        ball_detections.xyxy = ball_detections.xyxy[best_idx:best_idx+1]
        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy1 = view_transformer.transform_points(frame_ball_xy)
        last_ball_position = [pitch_ball_xy1[0, 0], pitch_ball_xy1[0, 1]] if len(pitch_ball_xy1) > 0 else [np.nan, np.nan]
    else:
        pitch_ball_xy1 = np.empty((0, 2))
        last_ball_position = [np.nan, np.nan]

    ball_x, ball_y = last_ball_position

    # Detección de jugadores
    result = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792, iou=0.7)[0]
    detections_players = sv.Detections.from_ultralytics(result)
    players_detections = detections_players[detections_players.class_id == 1]

    # Paso 1: extraer embeddings
    detections_to_track = []
    for bbox in players_detections.xyxy:
        emb = extract_embedding(frame, bbox, reid_model, preprocess, device)
        if emb is not None:
            detections_to_track.append((bbox, emb))

    # Paso 2: asignar IDs
    assignments = []
    for (bbox, emb) in detections_to_track:
        match_id = find_best_match(emb, tracked_embeddings)
        if match_id is None:
            match_id = next_id
            next_id += 1
        tracked_embeddings[match_id] = emb
        assignments.append((match_id, bbox, emb))

    # Paso 3: obtener posiciones proyectadas
    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = view_transformer.transform_points(np.array(frame_players_xy))

    players_positions = {}
    detected_player_ids = []

    for (match_id, bbox, _), pitch_pos in zip(assignments, pitch_players_xy):
        players_positions[match_id] = pitch_pos
        detected_player_ids.append(match_id)

        team = team_classifier.predict_team(frame, bbox, tracker_id=match_id)
        player_teams[match_id] = team

        color_map = {
            "equipo_negro":  (0, 0, 0),
            "equipo_blanco": (255, 255, 255),
            "UNKNOWN":       (128, 128, 128)
        }
        team_color = color_map.get(team, (128, 128, 128))
        draw_player_box(frame, bbox, match_id, team_color)

    if len(ball_detections.xyxy) > 0:
        for bbox, confidence in zip(ball_detections.xyxy, ball_detections.confidence):
            label = f"Ball {confidence:.2f}"
            draw_box(frame, bbox, label, color=(0, 255, 255))

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

    elements = {
        'points': [],
        'paths': []
    }
    for pid, pos in players_positions.items():
        color = (0, 0, 0) if player_teams.get(pid) == "equipo_negro" else (255, 255, 255)
        elements['points'].append((pos[0], pos[1], color))

    if not np.isnan(ball_x) and not np.isnan(ball_y):
        elements['points'].append((ball_x, ball_y, (0, 255, 255)))

    radar_image = pitch_renderer.draw(elements)
    radar_resized = cv2.resize(radar_image, (radar_width, radar_height))
    frame_with_radar = frame.copy()
    x, y = radar_position
    alpha = 0.6
    for c in range(3):
        frame_with_radar[y:y+radar_height, x:x+radar_width, c] = (
            alpha * radar_resized[:, :, c] + 
            (1 - alpha) * frame_with_radar[y:y+radar_height, x:x+radar_width, c]
        )
    frame_with_radar = cv2.cvtColor(frame_with_radar, cv2.COLOR_RGB2BGR)
    out.write(frame_with_radar)

from postprocess.postprocess import process_file
input_file = "codes/data/Posiciones-jugadores-balon.xlsx"
posiciones_df.to_excel(input_file, index=False)

process_file(
    file_path='codes/data/Posiciones-jugadores-balon.xlsx',
    cleaned_output_path='codes/data/limpieza.xlsx',
    output_possession_path='codes/data/posesion.xlsx',
    output_passes_path='codes/data/pases.xlsx',
    output_team_passes_path='codes/data/passes_by_{team}.xlsx'
)

import cv2
import numpy as np
from decord import VideoReader, cpu

def draw_player_box(frame, bbox, player_id, team_color, track_id=None, global_id=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
    labels = []
    if global_id is not None:
        labels.append(f"G{global_id}")
    if track_id is not None:
        labels.append(f"T{track_id}")
    labels.append(f"R{player_id}")
    label = " ".join(labels)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, team_color, 2)

def draw_box(frame, bbox, label, color=(255, 0, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 10)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_video_frames_generator(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    for frame in vr:
        yield frame.asnumpy()

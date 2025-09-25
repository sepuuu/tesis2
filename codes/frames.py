from decord import VideoReader, cpu

video1 = "codes/inputs/video_blancos_corto.mp4"
video2 = "codes/inputs/video_negros_corto.mp4"

vr1 = VideoReader(video1, ctx=cpu(0))
vr2 = VideoReader(video2, ctx=cpu(0))

print(f"Video 1: {video1}")
print(f"  Frames: {len(vr1)}")
print(f"  FPS:    {vr1.get_avg_fps():.2f}")

print(f"Video 2: {video2}")
print(f"  Frames: {len(vr2)}")
print(f"  FPS:    {vr2.get_avg_fps():.2f}")

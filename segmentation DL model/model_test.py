import cv2
from ultralytics import YOLO
import time

model = YOLO("/home/sukja/drone_Referee/segmentation DL model/line_seg/rev02/weights/best.pt")

video_path = "/home/sukja/drone_Referee/dataset/3배_1.MP4"
cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter("result1_v11.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print(f"총 {total}프레임, {fps:.1f}fps, {w}x{h}")

frame_idx = 0
fps_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    results = model.predict(frame, conf=0.5, device=0, verbose=False, imgsz=1280, retina_masks=True)
    elapsed = time.time() - t0
    current_fps = 1.0 / elapsed
    fps_list.append(current_fps)

    # 마스크만 표시 (라벨/conf 숨김)
    result_frame = results[0].plot(labels=False, conf=False, boxes=False)
    out.write(result_frame)

    frame_idx += 1
    if frame_idx % 30 == 0:
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        print(f"[{frame_idx}/{total}] {frame_idx/total*100:.1f}% | 현재 {current_fps:.1f}fps | 평균 {avg_fps:.1f}fps")

cap.release()
out.release()

avg_fps = sum(fps_list) / len(fps_list)
print(f"\n완료 → result1_v11.mp4")
print(f"전체 평균 FPS: {avg_fps:.1f}")
print(f"실시간 가능 여부: {'✅ 가능' if avg_fps >= fps else f'❌ 어려움 (영상 {fps:.1f}fps, 추론 {avg_fps:.1f}fps)'}")

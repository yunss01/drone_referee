import cv2
import os

def extract_frames_with_clahe(video_path, output_dir, fps=10,
                               clip_limit=2.0, tile_size=(8, 8)):
    os.makedirs(output_dir, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = round(video_fps / fps)
    
    frame_idx = 0
    saved_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            # CLAHE는 단채널에 적용 → LAB 변환 후 L채널에만 적용
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            result = cv2.merge([l_clahe, a, b])
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

            filename = os.path.join(output_dir, f"frame2_{saved_idx:05d}.jpg")
            cv2.imwrite(filename, result)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"✅ CLAHE 적용 완료: {saved_idx}장 저장")

if __name__ == "__main__":
    extract_frames_with_clahe(
        video_path="5.mp4",
        output_dir="frames_clahe/",
        fps=10,
        clip_limit=2.0,    # 높을수록 대비 강해짐 (1.5~3.0 권장)
        tile_size=(4, 4)   # 작을수록 로컬 대비 강해짐
    )

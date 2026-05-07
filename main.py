"""
main.py  —  드론 심판 시스템 통합 실행

MODE 설정:
  'camera' : USB 카메라 실시간 (preprocess.py의 CAM_NUM 사용)
  'video'  : 영상 파일 처리 후 결과 영상 저장
  'image'  : 이미지 1장 또는 폴더 전체 처리 후 저장

종료: 'q' 키 (camera/video 모드)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

from preprocess import camera_stream, _apply_clahe, CLIP_LIMIT, TILE_SIZE
from detector import WheelDetector
from lane_checker import (
    run_segmentation, filter_small_contours,
    get_instance_half_width, get_centerline,
    compute_distances, check_violation,
)
from violation_tracker import ViolationTracker

# ──────────────────────────────────────────────────────────
#  모드 설정
#
#  MODE = 'camera' : USB 카메라 실시간
#  MODE = 'video'  : 영상 파일 처리
#  MODE = 'image'  : 이미지 1장 또는 폴더
# ──────────────────────────────────────────────────────────
MODE = 'video'

# [video 모드] 입력 영상 / 출력 영상
VIDEO_INPUT  = "dataset/3배_4.MP4"
VIDEO_OUTPUT = "result_video.mp4"

# [image 모드] 이미지 1장 또는 폴더 경로 / 출력 폴더
IMAGE_INPUT  = "test.jpg"          # 파일 or 폴더
IMAGE_OUTPUT = "result_images/"

# ── 공통 설정 ──────────────────────────────────────────────
SEG_MODEL_PATH  = "best_seg_rev01.pt"
POSE_MODEL_PATH = "best_referee.pt"
MAX_WHEELS      = 4
SHOW_WINDOW     = False   # True: 화면 출력, False: 파일 저장만
# ──────────────────────────────────────────────────────────


# ── 차선 데이터 구성 (Step 1~4) ──────────────────────────
def build_lane_data(seg_model, frame):
    instance_masks = run_segmentation(seg_model, frame)
    lane_data = []

    for mask_bool in instance_masks:
        mask_bin = (mask_bool * 255).astype(np.uint8)
        contours = filter_small_contours(mask_bin)
        if not contours:
            continue

        half_width = get_instance_half_width(contours)
        largest    = max(contours, key=cv2.contourArea)
        centerline = get_centerline(largest)

        lane_data.append({
            "mask_bin"  : mask_bin,
            "contours"  : contours,
            "half_width": half_width,
            "centerline": centerline,
        })

    return lane_data


# ── 시각화 ───────────────────────────────────────────────
def draw_frame(frame, lane_data, wheels, tracker_states):
    vis = frame.copy()

    # 차선 마스크 오버레이
    for data in lane_data:
        color = (0, 200, 255)
        overlay = vis.copy()
        overlay[data["mask_bin"] > 0] = color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        pt1, pt2 = data["centerline"]
        cv2.line(vis, pt1, pt2, color, 2)

    # 바퀴 bbox + 접지점
    any_confirmed = False
    for wheel, state in zip(wheels, tracker_states):
        x1, y1, x2, y2 = wheel["bbox"]
        kx, ky          = wheel["keypoint"]
        kp_valid        = wheel["kp_valid"]
        confirmed       = state["confirmed"]
        violated        = state["violated"]

        if confirmed:
            any_confirmed = True

        if confirmed:
            bbox_color = (0, 0, 255)
        elif violated:
            bbox_color = (0, 128, 255)
        elif kp_valid:
            bbox_color = (0, 255, 0)
        else:
            bbox_color = (0, 215, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), bbox_color, 2)
        cv2.circle(vis, (kx, ky), 5, (0, 0, 255), -1)

        label = f"{'KP' if kp_valid else 'FB'} {wheel['conf']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bbox_color, 1)

        if confirmed:
            cv2.putText(vis, f"VIOLATION {state['duration']*1000:.0f}ms",
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    status       = "VIOLATION" if any_confirmed else "NORMAL"
    status_color = (0, 0, 255) if any_confirmed else (0, 255, 0)
    cv2.putText(vis, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    cv2.putText(vis, f"wheels:{len(wheels)}  lanes:{len(lane_data)}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    return vis


# ── 프레임 1장 처리 ──────────────────────────────────────
def process_frame(seg_model, wheel_det, trackers, frame):
    lane_data = build_lane_data(seg_model, frame)
    wheels    = wheel_det.predict(frame)

    tracker_states = []
    for w_idx, wheel in enumerate(wheels[:MAX_WHEELS]):
        kx, ky = wheel["keypoint"]
        compute_distances(lane_data, kx, ky)
        viol_results   = check_violation(lane_data)
        frame_violated = any(vr["violated"] for vr in viol_results)
        state          = trackers[w_idx].update(frame_violated)
        tracker_states.append(state)

    for i in range(len(wheels), MAX_WHEELS):
        trackers[i].reset()

    vis = draw_frame(frame, lane_data, wheels[:MAX_WHEELS], tracker_states)
    return vis


# ── 모델 / tracker 초기화 공통 ───────────────────────────
def load_models():
    print("🔍 모델 로딩 중...")
    seg_model = YOLO(SEG_MODEL_PATH)
    wheel_det = WheelDetector()
    trackers  = [ViolationTracker(wheel_id=i) for i in range(MAX_WHEELS)]
    print("✅ 모델 로딩 완료\n")
    return seg_model, wheel_det, trackers


# ── camera 모드 ───────────────────────────────────────────
def run_camera():
    seg_model, wheel_det, trackers = load_models()
    print("🎥 카메라 모드 시작. 종료: 'q'")

    for frame in camera_stream():
        vis = process_frame(seg_model, wheel_det, trackers, frame)

        if SHOW_WINDOW:
            cv2.imshow("Drone Referee", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("✅ 종료")


# ── video 모드 ────────────────────────────────────────────
def run_video():
    seg_model, wheel_det, trackers = load_models()

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {VIDEO_INPUT}")
        return

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out   = cv2.VideoWriter(VIDEO_OUTPUT,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, (w, h))

    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    print(f"🎬 영상 모드: {VIDEO_INPUT} ({total}프레임, {fps:.1f}fps, {w}×{h})")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = _apply_clahe(frame, clahe)
        vis   = process_frame(seg_model, wheel_det, trackers, frame)
        out.write(vis)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  [{frame_idx}/{total}] {frame_idx/total*100:.1f}%")

        if SHOW_WINDOW:
            cv2.imshow("Drone Referee", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print(f"✅ 완료 → {VIDEO_OUTPUT}")


# ── image 모드 ────────────────────────────────────────────
def run_image():
    seg_model, wheel_det, trackers = load_models()

    # 단일 파일 or 폴더 판별
    if os.path.isfile(IMAGE_INPUT):
        img_files = [IMAGE_INPUT]
        out_dir   = IMAGE_OUTPUT if IMAGE_OUTPUT else "result_images/"
    elif os.path.isdir(IMAGE_INPUT):
        exts      = ('.jpg', '.jpeg', '.png')
        img_files = sorted([
            os.path.join(IMAGE_INPUT, f)
            for f in os.listdir(IMAGE_INPUT)
            if f.lower().endswith(exts)
        ])
        out_dir   = IMAGE_OUTPUT
    else:
        print(f"❌ 경로를 찾을 수 없습니다: {IMAGE_INPUT}")
        return

    os.makedirs(out_dir, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    print(f"🖼️  이미지 모드: {len(img_files)}장 처리 → {out_dir}")

    for idx, fpath in enumerate(img_files):
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"  스킵 (읽기 실패): {fpath}")
            continue

        # 이미지 모드에서는 tracker를 매 이미지마다 초기화
        for t in trackers:
            t.reset()

        frame = _apply_clahe(frame, clahe)
        vis   = process_frame(seg_model, wheel_det, trackers, frame)

        fname    = os.path.basename(fpath)
        out_path = os.path.join(out_dir, f"result_{fname}")
        cv2.imwrite(out_path, vis)
        print(f"  [{idx+1}/{len(img_files)}] {fname} → {out_path}")

        if SHOW_WINDOW:
            cv2.imshow("Drone Referee", vis)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print(f"✅ 완료 → {out_dir}")


# ── 진입점 ───────────────────────────────────────────────
if __name__ == "__main__":
    if MODE == 'camera':
        run_camera()
    elif MODE == 'video':
        run_video()
    elif MODE == 'image':
        run_image()
    else:
        print(f"❌ 알 수 없는 MODE: '{MODE}' → 'camera' / 'video' / 'image'")

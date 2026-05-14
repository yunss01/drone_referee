"""
main.py  —  드론 심판 시스템 통합 실행

MODE 설정:
  'camera' : USB 카메라 실시간
  'video'  : 영상 파일 처리 후 결과 영상 저장
  'image'  : 이미지 1장 또는 폴더 전체 처리 후 저장

재위치 토글:
  터미널에서 Enter 키 → 재위치 모드 ON/OFF
  재위치 모드 ON 중에는 침범 판정이 일시정지됨

종료: 'q' 키 (camera/video 모드)
"""

import os
import cv2
import threading
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
# ──────────────────────────────────────────────────────────
MODE = 'video'

VIDEO_INPUT  = "dataset/3배_2.MP4"
VIDEO_OUTPUT = "result_video_2.mp4"

IMAGE_INPUT  = "test.jpg"
IMAGE_OUTPUT = "result_images_2/"

SEG_MODEL_PATH  = "best_seg_rev02.pt"
POSE_MODEL_PATH = "best_referee.pt"
MAX_WHEELS      = 4
SHOW_WINDOW     = False
# ──────────────────────────────────────────────────────────


# ── 재위치 토글 상태 관리 ────────────────────────────────
class RefereeState:
    """
    재위치 모드 토글을 스레드 안전하게 관리.
    터미널에서 Enter 키를 누르면 paused가 반전됨.
    """
    def __init__(self):
        self._lock  = threading.Lock()
        self._paused = False

    @property
    def paused(self):
        with self._lock:
            return self._paused

    def toggle(self):
        with self._lock:
            self._paused = not self._paused
        status = "⏸  재위치 모드 ON  (침범 판정 일시정지)" if self._paused \
                 else "▶  재위치 모드 OFF (침범 판정 재개)"
        print(f"\n[토글] {status}\n")


def start_input_listener(state: RefereeState):
    """백그라운드 스레드: 터미널 Enter 입력 대기 → 토글"""
    print("💡 터미널에서 Enter 키를 누르면 재위치 모드 ON/OFF 전환")

    def _listen():
        while True:
            try:
                input()
                state.toggle()
            except EOFError:
                break   # 파이프 입력 등으로 stdin이 닫힌 경우

    t = threading.Thread(target=_listen, daemon=True)
    t.start()


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
        centerline = get_centerline(mask_bin)
        if centerline is None:   # 비율 불안정 → skip
            continue

        lane_data.append({
            "mask_bin"  : mask_bin,
            "contours"  : contours,
            "half_width": half_width,
            "centerline": centerline,
        })

    return lane_data


# ── 시각화 ───────────────────────────────────────────────
def draw_frame(frame, lane_data, wheels, tracker_states, paused):
    vis = frame.copy()

    # 차선 마스크 오버레이
    for data in lane_data:
        color = (0, 200, 255)
        overlay = vis.copy()
        overlay[data["mask_bin"] > 0] = color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        pt1, pt2 = data["centerline"]
        cv2.line(vis, pt1, pt2, color, 2)

    # 바퀴 bbox + 접지점 (② conf/KP 레이블 제거)
    any_confirmed = False
    for wheel, state in zip(wheels, tracker_states):
        x1, y1, x2, y2 = wheel["bbox"]
        kx, ky          = wheel["keypoint"]
        confirmed       = state["confirmed"]
        violated        = state["violated"]

        if confirmed:
            any_confirmed = True

        if confirmed:
            bbox_color = (0, 0, 255)
        elif violated:
            bbox_color = (0, 128, 255)
        elif wheel["kp_valid"]:
            bbox_color = (0, 255, 0)
        else:
            bbox_color = (0, 215, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), bbox_color, 2)
        cv2.circle(vis, (kx, ky), 5, (0, 0, 255), -1)

        # 침범 확정 시 지속 시간 표시
        if confirmed:
            cv2.putText(vis, f"VIOLATION {state['duration']*1000:.0f}ms",
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 상단 상태 표시
    if paused:
        # 재위치 모드 중: 노란색으로 표시
        cv2.putText(vis, "REPOSITIONING", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
    else:
        status       = "VIOLATION" if any_confirmed else "NORMAL"
        status_color = (0, 0, 255) if any_confirmed else (0, 255, 0)
        cv2.putText(vis, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    cv2.putText(vis, f"wheels:{len(wheels)}  lanes:{len(lane_data)}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    return vis


# ── 프레임 1장 처리 ──────────────────────────────────────
def process_frame(seg_model, wheel_det, trackers, frame, state: RefereeState):
    lane_data = build_lane_data(seg_model, frame)
    wheels    = wheel_det.predict(frame)
    paused    = state.paused

    tracker_states = []
    for w_idx, wheel in enumerate(wheels[:MAX_WHEELS]):
        if paused:
            # 재위치 모드: tracker 초기화하고 정상 상태 반환
            trackers[w_idx].reset()
            tracker_states.append({
                "wheel_id" : w_idx,
                "violated" : False,
                "confirmed": False,
                "duration" : 0.0,
            })
        else:
            kx, ky = wheel["keypoint"]
            compute_distances(lane_data, kx, ky)
            viol_results   = check_violation(lane_data)
            frame_violated = any(vr["violated"] for vr in viol_results)
            tracker_states.append(trackers[w_idx].update(frame_violated))

    for i in range(len(wheels), MAX_WHEELS):
        trackers[i].reset()

    vis = draw_frame(frame, lane_data, wheels[:MAX_WHEELS], tracker_states, paused)
    return vis


# ── 모델 / tracker 초기화 ────────────────────────────────
def load_models():
    print("🔍 모델 로딩 중...")
    seg_model = YOLO(SEG_MODEL_PATH)
    wheel_det = WheelDetector()
    trackers  = [ViolationTracker(wheel_id=i) for i in range(MAX_WHEELS)]
    print("✅ 모델 로딩 완료\n")
    return seg_model, wheel_det, trackers


# ── camera 모드 ───────────────────────────────────────────
def run_camera(state):
    seg_model, wheel_det, trackers = load_models()
    print("🎥 카메라 모드 시작. 종료: 'q'")

    for frame in camera_stream():
        vis = process_frame(seg_model, wheel_det, trackers, frame, state)

        if SHOW_WINDOW:
            cv2.imshow("Drone Referee", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("✅ 종료")


# ── video 모드 ────────────────────────────────────────────
def run_video(state):
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
        vis   = process_frame(seg_model, wheel_det, trackers, frame, state)
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
def run_image(state):
    seg_model, wheel_det, trackers = load_models()

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

        for t in trackers:
            t.reset()

        frame = _apply_clahe(frame, clahe)
        vis   = process_frame(seg_model, wheel_det, trackers, frame, state)

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
    state = RefereeState()
    start_input_listener(state)

    if MODE == 'camera':
        run_camera(state)
    elif MODE == 'video':
        run_video(state)
    elif MODE == 'image':
        run_image(state)
    else:
        print(f"❌ 알 수 없는 MODE: '{MODE}' → 'camera' / 'video' / 'image'")

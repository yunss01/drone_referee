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
import time
import threading
import numpy as np
from ultralytics import YOLO

from preprocess import _apply_clahe, CLIP_LIMIT, TILE_SIZE
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

# [camera 모드] 카메라 장치 번호 (ls /dev/video* 로 확인)
CAM_NUM = 0

VIDEO_INPUT  = "dataset/3배_1.MP4"
VIDEO_OUTPUT = "result_video_1.mp4"

IMAGE_INPUT  = "test.jpg"
IMAGE_OUTPUT = "result_images/"

SEG_MODEL_PATH  = "best_seg_rev02.pt"
POSE_MODEL_PATH = "best_referee.pt"
MAX_WHEELS      = 4
SHOW_WINDOW     = False

# ── CSV 로그 설정 (video 모드 전용) ──────────────────────
SAVE_LOG      = False
VIOLATION_LOG = "violation_log.csv"
# ──────────────────────────────────────────────────────────


# ── 재위치 토글 상태 관리 ────────────────────────────────
class RefereeState:
    def __init__(self):
        self._lock   = threading.Lock()
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
    print("💡 터미널에서 Enter 키를 누르면 재위치 모드 ON/OFF 전환")

    def _listen():
        while True:
            try:
                input()
                state.toggle()
            except EOFError:
                break

    t = threading.Thread(target=_listen, daemon=True)
    t.start()


# ── FPS 측정기 ───────────────────────────────────────────
class FPSCounter:
    """
    최근 N개 프레임의 처리 시간을 기반으로 평균 FPS 계산.
    매 print_interval 프레임마다 터미널에 출력.
    """
    def __init__(self, window=30, print_interval=30):
        self.window         = window
        self.print_interval = print_interval
        self.times          = []
        self.frame_count    = 0
        self._last_tick     = None

    def tick(self):
        now = time.time()
        if self._last_tick is not None:
            self.times.append(now - self._last_tick)
            if len(self.times) > self.window:
                self.times.pop(0)
        self._last_tick = now
        self.frame_count += 1

    def fps(self):
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))

    def should_print(self):
        return self.frame_count % self.print_interval == 0


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
        centerline = get_centerline(mask_bin)
        if centerline is None:
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

    for data in lane_data:
        color = (0, 200, 255)
        overlay = vis.copy()
        overlay[data["mask_bin"] > 0] = color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        pt1, pt2 = data["centerline"]
        cv2.line(vis, pt1, pt2, color, 2)

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

        if confirmed:
            cv2.putText(vis, f"VIOLATION {state['duration']*1000:.0f}ms",
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if paused:
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
    return vis, tracker_states


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

    cap = cv2.VideoCapture(CAM_NUM)
    if not cap.isOpened():
        print(f"❌ 카메라 열기 실패: /dev/video{CAM_NUM}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"🎥 카메라 모드 시작 (장치: /dev/video{CAM_NUM}, {w}×{h})")
    print(f"   종료: 'q' 키\n")

    clahe   = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    fps_ctr = FPSCounter(window=30, print_interval=30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  프레임 읽기 실패, 재시도 중...")
                continue

            fps_ctr.tick()
            frame = _apply_clahe(frame, clahe)
            vis, _ = process_frame(seg_model, wheel_det, trackers, frame, state)

            if fps_ctr.should_print():
                print(f"  [camera] {fps_ctr.fps():.1f} fps  "
                      f"(frame {fps_ctr.frame_count})")

            if SHOW_WINDOW:
                cv2.imshow("Drone Referee", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print(f"\n✅ 종료  |  평균 FPS: {fps_ctr.fps():.1f}")


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

    log_f = None
    if SAVE_LOG:
        log_f = open(VIOLATION_LOG, 'w')
        log_f.write("frame,timestamp_sec,wheel_id\n")
        print(f"📝 침범 로그 저장 → {VIOLATION_LOG}")

    clahe   = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    fps_ctr = FPSCounter(window=30, print_interval=30)
    print(f"🎬 영상 모드: {VIDEO_INPUT} ({total}프레임, {fps:.1f}fps, {w}×{h})")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fps_ctr.tick()
        frame = _apply_clahe(frame, clahe)
        vis, tracker_states = process_frame(
            seg_model, wheel_det, trackers, frame, state
        )
        out.write(vis)

        if log_f:
            timestamp = frame_idx / fps
            for s in tracker_states:
                if s["confirmed"]:
                    log_f.write(f"{frame_idx},{timestamp:.3f},{s['wheel_id']}\n")

        frame_idx += 1
        if fps_ctr.should_print():
            progress = frame_idx / total * 100
            print(f"  [{frame_idx}/{total}] {progress:.1f}%  |  "
                  f"처리 속도: {fps_ctr.fps():.1f} fps  "
                  f"(원본: {fps:.1f} fps)")

        if SHOW_WINDOW:
            cv2.imshow("Drone Referee", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if log_f:
        log_f.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print(f"\n✅ 완료 → {VIDEO_OUTPUT}")
    print(f"   전체 평균 처리 속도: {fps_ctr.fps():.1f} fps  (원본: {fps:.1f} fps)")
    if SAVE_LOG:
        print(f"   로그 → {VIOLATION_LOG}")


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
        vis, _ = process_frame(seg_model, wheel_det, trackers, frame, state)

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

"""
main.py  —  드론 심판 시스템 통합 실행

MODE 설정:
  'video'  : 영상 파일 처리 후 결과 영상 저장
  'image'  : 이미지 1장 또는 폴더 전체 처리 후 저장
  'rtmp'   : 유튜브 라이브/RTMP 입력 → 추론 → 유튜브 RTMP 재송출

재위치 토글:
  터미널에서 Enter 키 → 재위치 모드 ON/OFF
  재위치 모드 ON 중에는 침범 판정이 일시정지됨

종료: 'q' 키 (video 모드)
"""

import os
import cv2
import time
import threading
import subprocess
import shutil
import sys
import signal
import atexit
import traceback
import faulthandler
from collections import deque
import numpy as np
from ultralytics import YOLO

try:
    import resource
except ImportError:
    resource = None

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
MODE = 'rtmp'  # 'video' / 'image' / 'rtmp'

VIDEO_INPUT  = "0035_D.mp4"
VIDEO_OUTPUT = "0035_result_video_comp.mp4"

IMAGE_INPUT  = "test.jpg"
IMAGE_OUTPUT = "result_images/"

# ── 라이브 스트림 모드 설정 ───────────────────────────────
# INPUT_STREAM_URL        : 유튜브 라이브 페이지 또는 직접 스트림 URL
# INPUT_STREAM_FORMAT     : yt-dlp 입력 포맷 우선순위
# OUTPUT_STREAM_ENABLED   : 처리 결과를 유튜브 RTMP로 재송출할지 여부
# OUTPUT_STREAM_RTMP_URL  : 유튜브 라이브 송출 주소
# SAVE_STREAM_OUTPUT      : 추론된 결과 영상을 로컬 mp4로 저장할지 여부
# STREAM_OUTPUT_PATH      : 로컬 저장 파일 경로
INPUT_STREAM_URL       = "https://youtube.com/live/--fQWyY7W-k"
INPUT_STREAM_FORMAT    = "bestvideo[height<=720]/best[height<=720]/bestvideo/best"
OUTPUT_STREAM_ENABLED  = True
OUTPUT_STREAM_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/e6m2-vfja-yz2m-sjab-fsj7"
OUTPUT_STREAM_PRESET   = "veryfast"
SAVE_STREAM_OUTPUT     = True
STREAM_OUTPUT_PATH     = "rtmp_result.mp4"
RUNTIME_LOG_PATH       = "logs/drone_referee_runtime.log"

# 유튜브 라이브 화면을 보면서 Enter를 누를 때의 지연 보정값(초)
REPOSITION_TOGGLE_DELAY_SEC = 0.0

SEG_MODEL_PATH  = "model/best_seg_rev03_FP16.engine"
POSE_MODEL_PATH = "model/best_referee_FP16.engine"
MAX_WHEELS      = 4
SHOW_WINDOW     = False
DRAW_LANE_VIS   = False

# ── 프레임 스킵 설정 ──────────────────────────────────────
# N프레임마다 한 번만 추론 (1 = 모든 프레임 추론, 6 = 6프레임마다 1번)
FRAME_SKIP = 1
STREAM_READ_TIMEOUT_SEC = 2.0

# ── CSV 로그 설정 (video 모드 전용) ──────────────────────
SAVE_LOG      = False
VIOLATION_LOG = "violation_log.csv"
# ──────────────────────────────────────────────────────────


_RUNTIME_LOG_FILE = None
_EXIT_RECORDED = False


class TeeStream:
    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self._pending = ""

    def write(self, data):
        if not data:
            return 0

        self.console_stream.write(data)
        self._pending += data

        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.log_stream.write(f"[{timestamp}] {line}\n")

        return len(data)

    def flush(self):
        self.console_stream.flush()
        if self._pending:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.log_stream.write(f"[{timestamp}] {self._pending}")
            self._pending = ""
        self.log_stream.flush()

    def isatty(self):
        return self.console_stream.isatty()


def get_memory_usage_mb():
    if resource is None:
        return None

    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage_kb / (1024 * 1024)
    return usage_kb / 1024.0


def record_exit(reason):
    global _EXIT_RECORDED
    if _EXIT_RECORDED:
        return
    _EXIT_RECORDED = True
    print(f"\n[종료 기록] {reason}", flush=True)


def setup_runtime_logging():
    global _RUNTIME_LOG_FILE

    os.makedirs(os.path.dirname(RUNTIME_LOG_PATH), exist_ok=True)
    _RUNTIME_LOG_FILE = open(RUNTIME_LOG_PATH, "a", encoding="utf-8", buffering=1)

    sys.stdout = TeeStream(sys.__stdout__, _RUNTIME_LOG_FILE)
    sys.stderr = TeeStream(sys.__stderr__, _RUNTIME_LOG_FILE)

    try:
        faulthandler.enable(_RUNTIME_LOG_FILE, all_threads=True)
    except Exception:
        pass

    def _handle_exception(exc_type, exc_value, exc_tb):
        print("\n[치명적 예외] 프로그램이 예외로 종료됩니다.", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        record_exit(f"uncaught exception: {exc_type.__name__}")

    sys.excepthook = _handle_exception

    if hasattr(threading, "excepthook"):
        def _handle_thread_exception(args):
            print(f"\n[스레드 예외] thread={args.thread.name}", flush=True)
            traceback.print_exception(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
            )
        threading.excepthook = _handle_thread_exception

    def _handle_signal(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"SIG{signum}"
        print(f"\n[시그널] {signame}({signum}) 수신", flush=True)
        record_exit(f"signal {signame}({signum})")
        raise SystemExit(128 + signum)

    for signame in ("SIGTERM", "SIGINT", "SIGHUP", "SIGABRT"):
        if hasattr(signal, signame):
            signal.signal(getattr(signal, signame), _handle_signal)

    def _on_exit():
        if not _EXIT_RECORDED:
            record_exit("normal interpreter shutdown")
        if _RUNTIME_LOG_FILE is not None:
            _RUNTIME_LOG_FILE.flush()

    atexit.register(_on_exit)

    print(f"📝 런타임 로그 파일: {RUNTIME_LOG_PATH}", flush=True)


# ── 재위치 토글 상태 관리 ────────────────────────────────
class RefereeState:
    def __init__(self):
        self._lock             = threading.Lock()
        self._paused           = False
        self._scheduled_paused = False
        self._pending_toggles  = []

    def _status_text(self, paused):
        if paused:
            return "⏸  재위치 모드 ON  (침범 판정 일시정지)"
        return "▶  재위치 모드 OFF (침범 판정 재개)"

    def _apply_pending_locked(self, now=None):
        if now is None:
            now = time.time()

        while self._pending_toggles and self._pending_toggles[0][0] <= now:
            _, paused = self._pending_toggles.pop(0)
            self._paused = paused
            print(f"\n[적용] {self._status_text(paused)}\n")

        if not self._pending_toggles:
            self._scheduled_paused = self._paused

    @property
    def paused(self):
        with self._lock:
            self._apply_pending_locked()
            return self._paused

    def toggle(self):
        with self._lock:
            now = time.time()
            self._apply_pending_locked(now)
            self._scheduled_paused = not self._scheduled_paused

            if REPOSITION_TOGGLE_DELAY_SEC <= 0:
                self._paused = self._scheduled_paused
                print(f"\n[토글] {self._status_text(self._paused)}\n")
                return

            effective_at = now + REPOSITION_TOGGLE_DELAY_SEC
            self._pending_toggles.append((effective_at, self._scheduled_paused))

        print(f"\n[예약] {self._status_text(self._scheduled_paused)}")
        print(f"      {REPOSITION_TOGGLE_DELAY_SEC:.1f}초 뒤 적용 예정\n")


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


class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap
        self._cond = threading.Condition()
        self._stop = False
        self._frame = None
        self._frame_id = 0
        self._fail_count = 0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while True:
            with self._cond:
                if self._stop:
                    return

            ret, frame = self.cap.read()

            with self._cond:
                if self._stop:
                    return

                if ret and frame is not None:
                    self._frame = frame
                    self._frame_id += 1
                    self._fail_count = 0
                else:
                    self._fail_count += 1
                self._cond.notify_all()

            if not ret:
                time.sleep(0.01)

    def read_latest(self, last_frame_id, timeout_sec=1.0):
        deadline = time.time() + timeout_sec
        with self._cond:
            while (
                not self._stop
                and self._frame_id <= last_frame_id
                and self._fail_count == 0
            ):
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

            if self._frame_id > last_frame_id and self._frame is not None:
                return self._frame_id, self._frame.copy(), self._fail_count

            return last_frame_id, None, self._fail_count

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class RealtimeOutputWriter:
    def __init__(self, fps, saved_out=None, stream_proc=None, stream_status="disabled"):
        self.interval = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
        self.saved_out = saved_out
        self.stream_proc = stream_proc
        self.stream_status = stream_status
        self.output_frame_count = 0

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False
        self._queue = deque()
        self._last_frame = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def submit_frame(self, frame, repeat_count=1):
        with self._cond:
            self._queue.append([frame.copy(), max(1, int(repeat_count))])
            self._cond.notify_all()

    def _run(self):
        with self._cond:
            while not self._queue and not self._stop:
                self._cond.wait(timeout=0.1)

        next_tick = time.perf_counter()
        while True:
            with self._cond:
                if self._stop:
                    return
                if self._queue:
                    frame, repeats_left = self._queue[0]
                    self._last_frame = frame
                    self._queue[0][1] -= 1
                    if self._queue[0][1] <= 0:
                        self._queue.popleft()
                else:
                    frame = self._last_frame
                stream_proc = self.stream_proc

            if frame is not None:
                if self.saved_out is not None:
                    self.saved_out.write(frame)

                if stream_proc is not None and stream_proc.stdin is not None:
                    try:
                        stream_proc.stdin.write(frame.tobytes())
                    except (BrokenPipeError, OSError):
                        print("⚠️  유튜브 송출 연결이 끊겼습니다. 로컬 저장만 계속 진행합니다.")
                        stop_stream_process(stream_proc)
                        with self._cond:
                            if self.stream_proc is stream_proc:
                                self.stream_proc = None
                                self.stream_status = "disconnected"

                self.output_frame_count += 1

            next_tick += self.interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self.stream_proc is not None:
            stop_stream_process(self.stream_proc)
            self.stream_proc = None

    def pending_frames(self):
        with self._cond:
            return len(self._queue)


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


def draw_dashed_line(vis, pt1, pt2, color, thickness=2, dash_len=18, gap_len=10):
    start = np.array(pt1, dtype=np.float32)
    end   = np.array(pt2, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return

    direction = delta / length
    pos = 0.0
    while pos < length:
        dash_start = start + direction * pos
        dash_end   = start + direction * min(pos + dash_len, length)
        cv2.line(
            vis,
            tuple(np.round(dash_start).astype(int)),
            tuple(np.round(dash_end).astype(int)),
            color,
            thickness,
        )
        pos += dash_len + gap_len


# ── 시각화 ───────────────────────────────────────────────
def draw_frame(frame, lane_data, wheels, tracker_states, paused):
    vis = frame.copy()

    if DRAW_LANE_VIS:
        for data in lane_data:
            color = (0, 200, 255)
            overlay = vis.copy()
            overlay[data["mask_bin"] > 0] = color
            vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
            cv2.drawContours(vis, data["contours"], -1, (255, 255, 255), 1)
            pt1, pt2 = data["centerline"]
            draw_dashed_line(vis, pt1, pt2, (255, 255, 255), thickness=4)
            draw_dashed_line(vis, pt1, pt2, color, thickness=2)

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

    info_text = f"wheels:{len(wheels)}"
    if DRAW_LANE_VIS:
        info_text += f"  lanes:{len(lane_data)}"
    cv2.putText(vis, info_text,
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
    seg_model = YOLO(SEG_MODEL_PATH, task='segment')
    wheel_det = WheelDetector(model_path=POSE_MODEL_PATH)
    trackers  = [ViolationTracker(wheel_id=i) for i in range(MAX_WHEELS)]
    print("✅ 모델 로딩 완료\n")
    return seg_model, wheel_det, trackers


def start_ffmpeg_log_thread(proc):
    if proc.stderr is None:
        return

    def _drain():
        try:
            for raw_line in iter(proc.stderr.readline, b""):
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"[ffmpeg] {line}")
        except Exception as e:
            print(f"[ffmpeg] stderr 읽기 실패: {e}")
        finally:
            try:
                return_code = proc.wait(timeout=0.1)
                print(f"[ffmpeg] 프로세스 종료 code={return_code}")
            except subprocess.TimeoutExpired:
                pass

    threading.Thread(target=_drain, name="ffmpeg-log", daemon=True).start()


def resolve_stream_url(input_url):
    if not input_url:
        print("❌ INPUT_STREAM_URL 설정이 비어 있습니다.")
        return None

    if not input_url.startswith("http"):
        return input_url

    if "youtube.com" not in input_url and "youtu.be" not in input_url:
        return input_url

    print("  🔍 yt-dlp로 유튜브 스트림 URL 추출 중...")

    commands = []
    yt_dlp_bin = shutil.which("yt-dlp")
    if yt_dlp_bin:
        commands.append([yt_dlp_bin, "-g", "-f", INPUT_STREAM_FORMAT, input_url])
    commands.append([
        "python3", "-m", "yt_dlp", "-g", "-f", INPUT_STREAM_FORMAT, input_url
    ])

    last_error = None
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue

        stream_urls = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        if result.returncode == 0 and stream_urls:
            print("  ✅ 스트림 URL 추출 완료\n")
            return stream_urls[0]

        stderr = result.stderr.strip() or "(stderr 없음)"
        last_error = f"{' '.join(cmd[:3])}: {stderr}"

    print("❌ 유튜브 스트림 URL 추출 실패")
    if yt_dlp_bin is None:
        print("   원인: yt-dlp가 설치되어 있지 않습니다.")
        print("   설치: `sudo apt install yt-dlp` 또는 `python3 -m pip install yt-dlp`")
    if last_error:
        print(f"   오류: {last_error}")
    return None


def open_stream_capture(stream_url, retries=10, retry_delay=3.0):
    cap = None
    for attempt in range(1, retries + 1):
        if cap is not None:
            cap.release()

        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        cap.release()

        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        cap.release()

        print(f"  ⏳ 입력 스트림 연결 실패, 재시도 중... ({attempt}/{retries})")
        time.sleep(retry_delay)

    return None


def start_output_stream_process(width, height, fps):
    if not OUTPUT_STREAM_ENABLED:
        return None

    if (
        not OUTPUT_STREAM_RTMP_URL
        or "YOUR_STREAM_KEY" in OUTPUT_STREAM_RTMP_URL
    ):
        print("⚠️  OUTPUT_STREAM_ENABLED=True 이지만 OUTPUT_STREAM_RTMP_URL 설정이 비어 있습니다.")
        print("   로컬 저장만 계속 진행합니다.")
        return None

    # YouTube 권장에 맞춰 2초 간격으로 키프레임을 고정한다.
    gop_size = max(1, int(round(fps * 2)))

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.3f}",
        "-i", "-",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", OUTPUT_STREAM_PRESET,
        "-tune", "zerolatency",
        "-g", str(gop_size),
        "-keyint_min", str(gop_size),
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-shortest",
        "-f", "flv",
        OUTPUT_STREAM_RTMP_URL,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("⚠️  ffmpeg를 찾을 수 없습니다. 로컬 저장만 계속 진행합니다.")
        return None
    except Exception as e:
        print(f"⚠️  라이브 송출 시작 실패: {e}")
        print("   로컬 저장만 계속 진행합니다.")
        return None

    print(f"📡 라이브 송출 시작 → {OUTPUT_STREAM_RTMP_URL}")
    print(f"   키프레임 간격: 약 {gop_size / fps:.1f}초 ({gop_size} 프레임)")
    start_ffmpeg_log_thread(proc)
    return proc


def stop_stream_process(proc):
    if proc is None:
        return

    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass

    try:
        proc.wait(timeout=5)
        print(f"[ffmpeg] stop 후 종료 code={proc.returncode}")
    except Exception:
        print("[ffmpeg] stop timeout -> kill()")
        proc.kill()

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

    # out_fps = fps / FRAME_SKIP # 원본과 같은 시간 길이로 재생
    out_fps = fps / FRAME_SKIP # 원본과 같은 속도로 재생 (30fps 유지)
    out = cv2.VideoWriter(VIDEO_OUTPUT,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          out_fps, (w, h))

    log_f = None
    if SAVE_LOG:
        log_f = open(VIOLATION_LOG, 'w')
        log_f.write("frame,timestamp_sec,wheel_id\n")
        print(f"📝 침범 로그 저장 → {VIOLATION_LOG}")

    clahe   = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    fps_ctr = FPSCounter(window=30, print_interval=30)
    print(f"🎬 영상 모드: {VIDEO_INPUT} ({total}프레임, {fps:.1f}fps, {w}×{h})")
    print(f"   프레임 스킵: {FRAME_SKIP} → 추론 {total//FRAME_SKIP}프레임, "
          f"결과 영상 {out_fps:.1f}fps\n")

    frame_idx   = 0
    infer_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # FRAME_SKIP마다 한 번만 추론
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

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

        infer_count += 1
        frame_idx   += 1

        if fps_ctr.should_print():
            progress = frame_idx / total * 100
            print(f"  [{frame_idx}/{total}] {progress:.1f}%  |  "
                  f"추론 속도: {fps_ctr.fps():.1f} fps  "
                  f"(원본: {fps:.1f} fps, skip={FRAME_SKIP})")

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
    print(f"   추론 프레임 수: {infer_count} / {total}")
    print(f"   추론 평균 속도: {fps_ctr.fps():.1f} fps")
    print(f"   결과 영상 fps:  {out_fps:.1f} fps")
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


# ── rtmp 모드 ─────────────────────────────────────────────
def run_rtmp(state):
    """
    유튜브 라이브 또는 RTMP 스트림을 입력으로 받아 추론하고,
    결과를 유튜브 RTMP로 재송출하거나 로컬 mp4로 저장한다.
    """
    seg_model, wheel_det, trackers = load_models()

    print(f"📡 스트림 모드 입력: {INPUT_STREAM_URL}")
    print(f"   종료: Ctrl+C\n")

    stream_url = resolve_stream_url(INPUT_STREAM_URL)
    if not stream_url:
        return

    cap = open_stream_capture(stream_url, retries=10, retry_delay=3.0)
    if cap is None or not cap.isOpened():
        print(f"❌ 스트림에 연결할 수 없습니다.")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0
    output_fps = fps / FRAME_SKIP
    output_size = (w, h)

    print(f"✅ 스트림 연결 성공: {w}×{h} @ {fps:.1f}fps")
    print(f"   프레임 스킵: {FRAME_SKIP}\n")

    saved_out = None
    if SAVE_STREAM_OUTPUT:
        out_dir = os.path.dirname(STREAM_OUTPUT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        saved_out = cv2.VideoWriter(
            STREAM_OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*'mp4v'),
            output_fps,
            output_size,
        )
        if not saved_out.isOpened():
            print(f"⚠️  결과 저장 파일을 열 수 없습니다: {STREAM_OUTPUT_PATH}")
            saved_out.release()
            saved_out = None

    stream_proc = start_output_stream_process(w, h, output_fps)
    if not OUTPUT_STREAM_ENABLED:
        stream_status = "disabled"
    elif stream_proc is None:
        stream_status = "start_failed"
    else:
        stream_status = "streaming"

    print(f"   로컬 저장: {'ON' if saved_out else 'OFF'}")
    if SAVE_STREAM_OUTPUT:
        print(f"   저장 경로: {STREAM_OUTPUT_PATH}")
    print(f"   유튜브 송출: {'ON' if stream_proc else 'OFF'}")
    print(f"   출력 FPS: {output_fps:.1f} (새 프레임이 늦으면 직전 프레임 유지)\n")

    clahe     = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    frame_reader = LatestFrameReader(cap).start()
    output_writer = RealtimeOutputWriter(
        output_fps,
        saved_out=saved_out,
        stream_proc=stream_proc,
        stream_status=stream_status,
    ).start()
    fps_ctr   = FPSCounter(window=30, print_interval=30)
    last_frame_id = 0
    last_output_frame_id = 0
    frame_idx = 0
    fail_cnt  = 0
    infer_count = 0

    try:
        while True:
            frame_id, frame, reader_fail_cnt = frame_reader.read_latest(
                last_frame_id,
                timeout_sec=STREAM_READ_TIMEOUT_SEC,
            )
            if frame is None:
                fail_cnt = max(fail_cnt + 1, reader_fail_cnt)
                if fail_cnt >= 30:
                    print("⚠️  입력 스트림이 끊겨 재연결을 시도합니다.")
                    frame_reader.stop()
                    cap.release()
                    cap = open_stream_capture(stream_url, retries=5, retry_delay=2.0)
                    if cap is None or not cap.isOpened():
                        print("❌ 입력 스트림 재연결 실패")
                        break

                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
                    cap_fps = cap.get(cv2.CAP_PROP_FPS)
                    if cap_fps and not np.isnan(cap_fps) and cap_fps > 0:
                        fps = cap_fps
                    frame_reader = LatestFrameReader(cap).start()
                    last_frame_id = 0
                    last_output_frame_id = 0
                    fail_cnt = 0
                    print("✅ 입력 스트림 재연결 성공")
                    continue
                time.sleep(0.02)
                continue

            last_frame_id = frame_id
            fail_cnt = 0

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            fps_ctr.tick()
            frame = _apply_clahe(frame, clahe)
            vis, _ = process_frame(
                seg_model, wheel_det, trackers, frame, state
            )
            vis_for_output = vis
            if (vis.shape[1], vis.shape[0]) != output_size:
                vis_for_output = cv2.resize(vis, output_size)

            repeat_count = max(1, frame_id - last_output_frame_id)
            output_writer.submit_frame(vis_for_output, repeat_count=repeat_count)
            last_output_frame_id = frame_id

            frame_idx += 1
            infer_count += 1

            if fps_ctr.should_print():
                rss_mb = get_memory_usage_mb()
                rss_text = f"{rss_mb:.1f}MB" if rss_mb is not None else "n/a"
                print(
                    f"  [rtmp] 추론 {fps_ctr.fps():.1f} fps  "
                    f"(frame {fps_ctr.frame_count}, processed={infer_count}, "
                    f"emitted={output_writer.output_frame_count}, "
                    f"last_repeat={repeat_count}, queue={output_writer.pending_frames()}, "
                    f"rss={rss_text}, source={w}x{h})"
                )

            if SHOW_WINDOW:
                cv2.imshow("Drone Referee - RTMP", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        frame_reader.stop()
        cap.release()
        output_writer.stop()
        if saved_out:
            saved_out.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print(f"\n✅ 종료  |  추론 평균 FPS: {fps_ctr.fps():.1f}")
    print(f"   처리 프레임 수: {infer_count}")
    print(f"   출력 프레임 수: {output_writer.output_frame_count}")
    if SAVE_STREAM_OUTPUT and saved_out is not None:
        print(f"   결과 저장 → {STREAM_OUTPUT_PATH}")
    stream_status_text = {
        "disabled": "사용 안 함",
        "start_failed": "시작 실패",
        "streaming": "종료 시점까지 송출 유지",
        "disconnected": "중간에 연결 끊김",
    }[output_writer.stream_status]
    print(f"   유튜브 송출 상태: {stream_status_text}")


# ── 진입점 ───────────────────────────────────────────────
if __name__ == "__main__":
    setup_runtime_logging()
    state = RefereeState()
    start_input_listener(state)

    if MODE == 'video':
        run_video(state)
    elif MODE == 'image':
        run_image(state)
    elif MODE == 'rtmp':
        run_rtmp(state)
    else:
        print(f"❌ 알 수 없는 MODE: '{MODE}' → 'video' / 'image' / 'rtmp'")

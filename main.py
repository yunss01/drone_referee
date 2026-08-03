"""
main.py  —  드론 심판 시스템 통합 실행

MODE 설정:
  'video'  : 영상 파일 처리 후 결과 영상 저장
  'image'  : 이미지 1장 또는 폴더 전체 처리 후 저장
  'rtmp'   : 유튜브/RTMP 입력을 디스크 세그먼트로 스풀 후 순차 추론 → 유튜브 RTMP 재송출

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
INPUT_STREAM_URL       = "https://youtube.com/live/--fQWyY7W-k" # 드론 영상 유튜브 라이브 링크
INPUT_STREAM_FORMAT    = (
    "bestvideo[height<=1080][vcodec*=avc1]/"
    "best[height<=1080][vcodec*=avc1]/"
    "bestvideo[height<=1080]/best[height<=1080]/bestvideo/best"
)
OUTPUT_STREAM_ENABLED  = True
OUTPUT_STREAM_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/e6m2-vfja-yz2m-sjab-fsj7" # 성균관대학교노승윤 계정
SAVE_STREAM_OUTPUT     = False
STREAM_OUTPUT_PATH     = "rtmp_result.mp4"
RUNTIME_LOG_PATH       = "logs/drone_referee_runtime.log"
SPOOL_ROOT_DIR         = "spool_sessions"
SPOOL_SEGMENT_SECONDS  = 10
KEEP_INPUT_SEGMENTS    = False
DELETE_INPUT_SEGMENTS_AFTER_PROCESS = True
SPOOL_POLL_INTERVAL_SEC = 0.5

# RTMP relay 설정 (디스크 스풀 송출 + 자동 재연결)
# 이 시스템의 유튜브 라이브는 실시간 시청용이 아니라 나중에 확인하는 아카이브다.
# 따라서 지연이 생기더라도 한 프레임도 버리지 않는 것이 최우선이다.
RTMP_SPOOL_PART_MB = 4
RTMP_SPOOL_WARN_GB = 20
RTMP_RECONNECT_DELAY_SEC = 3.0
RTMP_RELAY_CHUNK_BYTES = 188 * 64
# 종료 시 남은 송출 대기분을 끝까지 보낼지 여부.
# True면 Ctrl+C 후에도 스풀을 다 비울 때까지 송출을 계속한다.
# (그 동안 Ctrl+C를 한 번 더 누르면 강제 종료)
DRAIN_SPOOL_ON_EXIT = True

STREAM_VIDEO_ENCODER   = "h264_nvenc"
CPU_FALLBACK_VIDEO_ENCODER = "libx264"
NVENC_PRESET           = "p5"
X264_PRESET            = "veryfast"

# 유튜브 라이브 화면을 보면서 Enter를 누를 때의 지연 보정값(초)
REPOSITION_TOGGLE_DELAY_SEC = 0.0

SEG_MODEL_PATH  = "model/best_seg_rev03_FP16.engine"
POSE_MODEL_PATH = "model/best_referee_FP16.engine"
MAX_WHEELS      = 4
SHOW_WINDOW     = False
DRAW_LANE_VIS   = True # 점선 표시 여부

# ── 프레임 스킵 설정 ──────────────────────────────────────
# N프레임마다 한 번만 추론 (1 = 모든 프레임 추론, 6 = 6프레임마다 1번)
FRAME_SKIP = 2

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
        # SIGINT는 KeyboardInterrupt로 올려보내 정상 종료 절차(남은 송출
        # 대기분 전송)를 타게 한다. 그 단계에서 Ctrl+C를 한 번 더 누르면
        # 그때는 강제 종료된다.
        if signum == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
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


def start_ffmpeg_log_thread(proc, name="ffmpeg"):
    if proc.stderr is None:
        return

    def _drain():
        try:
            for raw_line in iter(proc.stderr.readline, b""):
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"[{name}] {line}")
        except Exception as e:
            print(f"[{name}] stderr 읽기 실패: {e}")
        finally:
            try:
                return_code = proc.wait(timeout=0.1)
                print(f"[{name}] 프로세스 종료 code={return_code}")
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


def pick_stream_bitrate(width, height, fps):
    pixels = width * height
    fps_scale = 1.5 if fps > 30.5 else 1.0

    if pixels >= 3840 * 2160:
        bitrate_mbps = 22.0
        maxrate_mbps = 28.0
        cq = 19
    elif pixels >= 2560 * 1440:
        bitrate_mbps = 12.0
        maxrate_mbps = 16.0
        cq = 20
    elif pixels >= 1920 * 1080:
        bitrate_mbps = 8.0
        maxrate_mbps = 10.0
        cq = 20
    elif pixels >= 1280 * 720:
        bitrate_mbps = 5.0
        maxrate_mbps = 7.0
        cq = 21
    else:
        bitrate_mbps = 3.0
        maxrate_mbps = 4.5
        cq = 22

    bitrate_mbps *= fps_scale
    maxrate_mbps *= fps_scale
    bufsize_mbps = bitrate_mbps * 2.0

    return {
        "b:v": f"{bitrate_mbps:.1f}M",
        "maxrate": f"{maxrate_mbps:.1f}M",
        "bufsize": f"{bufsize_mbps:.1f}M",
        "cq": str(cq),
    }


def build_video_encoder_args(width, height, fps, encoder_name=STREAM_VIDEO_ENCODER):
    bitrate = pick_stream_bitrate(width, height, fps)

    if encoder_name == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", NVENC_PRESET,
            "-rc", "vbr",
            "-cq", bitrate["cq"],
            "-b:v", bitrate["b:v"],
            "-maxrate", bitrate["maxrate"],
            "-bufsize", bitrate["bufsize"],
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ], bitrate

    return [
        "-c:v", CPU_FALLBACK_VIDEO_ENCODER,
        "-preset", X264_PRESET,
        "-tune", "zerolatency",
        "-b:v", bitrate["b:v"],
        "-maxrate", bitrate["maxrate"],
        "-bufsize", bitrate["bufsize"],
        "-pix_fmt", "yuv420p",
    ], bitrate


def stop_named_stream_process(name, proc):
    """DiskSpoolRelay가 쓰는 인터페이스. main.py의 stop_stream_process 래퍼."""
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
        print(f"[ffmpeg:{name}] stop 후 종료 code={proc.returncode}")
    except Exception:
        print(f"[ffmpeg:{name}] stop timeout -> kill()")
        proc.kill()


def start_named_ffmpeg_log_thread(name, proc):
    start_ffmpeg_log_thread(proc, name=f"ffmpeg:{name}")


class DiskSpoolRelay:
    """mpegts 바이트 스트림을 디스크에 스풀하면서 RTMP로 송출한다.

    구조:  상류 ffmpeg -> [writer 스레드] -> 디스크 스풀 파트 파일
                       -> [sender 스레드] -> ffmpeg push -> RTMP

    설계 원칙 (이 시스템은 실시간 시청용이 아니라 아카이브용 라이브):
    - 한 프레임도 버리지 않는다. 네트워크가 느리면 지연될 뿐, 유실은 없다.
    - 상류(디스크 녹화/인코딩)는 네트워크 상태와 무관하게 절대 블로킹되지 않는다.
    - push가 죽으면 자동 재연결하고, 현재 파트를 처음부터 다시 보내
      파이프/소켓에 남아 사라졌을 수 있는 구간까지 확실히 재전송한다.
      (유실보다 약간의 중복이 낫다)
    - 전송이 끝난 파트 파일은 즉시 삭제해 디스크를 회수한다.
    """

    def __init__(self, name, source_stream, rtmp_url, spool_dir,
                 part_bytes=None, reconnect_delay=None):
        self.name = name
        self.source = source_stream
        self.rtmp_url = rtmp_url
        self.spool_dir = spool_dir
        self.part_bytes = part_bytes or (RTMP_SPOOL_PART_MB * 1024 * 1024)
        self.reconnect_delay = reconnect_delay or RTMP_RECONNECT_DELAY_SEC

        os.makedirs(self.spool_dir, exist_ok=True)

        self._stopping = threading.Event()      # 즉시 중단(강제 종료)
        self._source_done = threading.Event()   # 상류 입력 종료
        self._sender_done = threading.Event()   # 스풀을 전부 송출 완료
        self._lock = threading.Lock()

        self._write_index = 0        # writer가 쓰고 있는 파트 번호
        self._send_index = 0         # sender가 보내고 있는 파트 번호
        self._written_bytes = 0      # 누적 스풀 기록량
        self._sent_bytes = 0         # 누적 전송 완료량
        self._push_proc = None
        self._threads = []
        self._warned_backlog = False

        self.reconnect_count = 0
        self.status = "pending"

    # ── 외부 API ──────────────────────────────────────────
    def start(self):
        proc = self._spawn_push()
        if proc is None:
            self.status = "start_failed"
            return False

        self._push_proc = proc
        self.status = "streaming"

        for target, tname in (
            (self._writer_loop, f"{self.name}-spool"),
            (self._sender_loop, f"{self.name}-sender"),
        ):
            t = threading.Thread(target=target, name=tname, daemon=True)
            t.start()
            self._threads.append(t)
        return True

    def pending_bytes(self):
        """아직 유튜브로 못 보낸 분량(바이트)."""
        with self._lock:
            return max(0, self._written_bytes - self._sent_bytes)

    def close_source(self):
        """상류 입력만 닫는다. 이미 스풀된 분량은 계속 송출된다."""
        try:
            self.source.close()
        except Exception:
            pass

    def drain(self, progress_cb=None, poll_sec=2.0):
        """남은 스풀을 전부 송출할 때까지 대기한다.
        상류 ffmpeg는 호출 전에 이미 종료되어 있어야 한다(그래야 EOF가 온다).
        progress_cb(pending_bytes)가 주어지면 주기적으로 호출한다.
        """
        while not self._stopping.is_set():
            if self._sender_done.is_set():
                break
            if progress_cb is not None:
                progress_cb(self.pending_bytes())
            time.sleep(poll_sec)

    def stop(self):
        self._stopping.set()
        self.close_source()
        for t in self._threads:
            t.join(timeout=3)

        proc = self._push_proc
        self._push_proc = None
        if proc is not None:
            stop_named_stream_process(f"{self.name}-push", proc)

        self._cleanup_spool()

    def summary(self):
        pending_mb = self.pending_bytes() / (1024 * 1024)
        sent_mb = self._sent_bytes / (1024 * 1024)
        return (
            f"전송 {sent_mb:.0f}MB, 미전송 {pending_mb:.1f}MB, "
            f"재연결 {self.reconnect_count}회"
        )

    # ── 내부 구현 ─────────────────────────────────────────
    def _part_path(self, index):
        return os.path.join(self.spool_dir, f"part_{index:08d}.ts")

    def _spawn_push(self):
        try:
            return subprocess.Popen(
                [
                    "ffmpeg", "-loglevel", "error",
                    "-f", "mpegts", "-i", "pipe:0",
                    "-c", "copy", "-f", "flv",
                    self.rtmp_url,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            print(f"⚠️  ffmpeg를 찾을 수 없어 {self.name} 송출을 시작하지 못했습니다.")
            return None
        except Exception as e:
            print(f"⚠️  {self.name} 송출 프로세스 시작 실패: {e}")
            return None

    def _writer_loop(self):
        """상류에서 계속 읽어 디스크 파트 파일로 쌓는다.
        네트워크와 완전히 무관하므로 상류가 막히는 일이 없다.
        """
        f = None
        part_written = 0
        try:
            while not self._stopping.is_set():
                chunk = self.source.read(RTMP_RELAY_CHUNK_BYTES)
                if not chunk:
                    break

                if f is None:
                    f = open(self._part_path(self._write_index), "wb")
                    part_written = 0

                f.write(chunk)
                f.flush()
                part_written += len(chunk)
                with self._lock:
                    self._written_bytes += len(chunk)

                if part_written >= self.part_bytes:
                    f.close()
                    f = None
                    with self._lock:
                        self._write_index += 1

                self._check_backlog()
        except Exception as e:
            if not self._stopping.is_set():
                print(f"⚠️  {self.name} 스풀 기록 종료: {e}")
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            with self._lock:
                # 마지막 파트도 sender가 넘어갈 수 있도록 인덱스를 진행시킨다.
                self._write_index += 1
            self._source_done.set()

    def _check_backlog(self):
        pending = self.pending_bytes()
        warn_bytes = RTMP_SPOOL_WARN_GB * 1024 * 1024 * 1024
        if pending >= warn_bytes and not self._warned_backlog:
            self._warned_backlog = True
            print(
                f"⚠️  {self.name} 송출 대기분이 {pending / (1024**3):.1f}GB를 넘었습니다. "
                f"네트워크 업로드가 계속 부족한 상태입니다."
            )
        elif pending < warn_bytes / 2:
            self._warned_backlog = False

    def _sender_loop(self):
        """파트 파일을 순서대로 push로 흘린다. 전송 끝난 파트는 삭제."""
        f = None
        try:
            while not self._stopping.is_set():
                if f is None:
                    path = self._part_path(self._send_index)
                    if not os.path.exists(path):
                        if self._source_done.is_set():
                            with self._lock:
                                if self._send_index >= self._write_index:
                                    return
                        time.sleep(0.05)
                        continue
                    f = open(path, "rb")

                chunk = f.read(RTMP_RELAY_CHUNK_BYTES)

                if not chunk:
                    # 이 파트를 다 읽었다. 다음 파트가 있으면 넘어간다.
                    with self._lock:
                        part_complete = self._send_index < self._write_index
                        source_done = self._source_done.is_set()
                    if part_complete or source_done:
                        f.close()
                        f = None
                        try:
                            os.remove(self._part_path(self._send_index))
                        except OSError:
                            pass
                        self._send_index += 1
                        continue
                    # 아직 상류가 이 파트에 더 쓰는 중 -> 잠깐 기다렸다 재시도
                    time.sleep(0.05)
                    continue

                if not self._ensure_push_alive():
                    if self._stopping.is_set():
                        return
                    # 재연결 실패 -> 현재 파트를 처음부터 다시 보낸다.
                    f.seek(0)
                    continue

                proc = self._push_proc
                if proc is None or proc.stdin is None:
                    continue
                try:
                    proc.stdin.write(chunk)
                    with self._lock:
                        self._sent_bytes += len(chunk)
                except (BrokenPipeError, OSError, ValueError, AttributeError):
                    print(f"⚠️  {self.name} 송출 연결이 끊겼습니다. 재연결 후 이어서 보냅니다.")
                    self._kill_push()
                    # 파이프/소켓에 남아 사라졌을 수 있는 구간까지 확실히
                    # 재전송하기 위해 현재 파트를 처음부터 다시 보낸다.
                    with self._lock:
                        self._sent_bytes -= f.tell()
                    f.seek(0)
        except Exception as e:
            if not self._stopping.is_set():
                print(f"⚠️  {self.name} 송출 스레드 종료: {e}")
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            self._sender_done.set()

    def _ensure_push_alive(self):
        proc = self._push_proc
        if proc is not None and proc.poll() is None:
            return True

        if proc is not None:
            print(
                f"⚠️  {self.name} 송출 프로세스가 종료되었습니다 "
                f"(code={proc.returncode}). {self.reconnect_delay:.0f}초 후 재연결합니다."
            )
            self._kill_push()

        if self._stopping.wait(timeout=self.reconnect_delay):
            return False

        new_proc = self._spawn_push()
        if new_proc is None:
            self.status = "disconnected"
            return False

        self._push_proc = new_proc
        self.reconnect_count += 1
        self.status = "streaming"
        start_named_ffmpeg_log_thread(f"{self.name}-push", new_proc)
        print(f"🔁 {self.name} 송출 재연결 완료 (누적 {self.reconnect_count}회)")
        return True

    def _kill_push(self):
        proc = self._push_proc
        self._push_proc = None
        if proc is None:
            return
        # 순서 주의: 다른 스레드가 stdin에 write 중일 수 있으므로
        # 프로세스를 먼저 죽여 그 write가 EPIPE로 풀리게 한 뒤 닫는다.
        # (반대로 하면 write 중인 파일 객체를 닫게 되어 데드락이 난다)
        try:
            proc.kill()
        except Exception:
            pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

    def _cleanup_spool(self):
        """남은 파트 파일 정리. 미전송분이 있으면 남겨두고 경로를 알린다."""
        try:
            remaining = [
                name for name in os.listdir(self.spool_dir)
                if name.startswith("part_") and name.endswith(".ts")
            ]
        except OSError:
            return

        if remaining and not self._sender_done.is_set():
            left = sum(
                os.path.getsize(os.path.join(self.spool_dir, n))
                for n in remaining
            )
            print(
                f"⚠️  {self.name} 미전송 {left / (1024**2):.1f}MB가 남았습니다. "
                f"스풀 보존: {self.spool_dir}"
            )
            return

        try:
            for name in remaining:
                os.remove(os.path.join(self.spool_dir, name))
            os.rmdir(self.spool_dir)
        except OSError:
            pass


def drain_relays_on_exit(*relays):
    """종료 요청 후에도 남은 송출 대기분을 끝까지 밀어낸다.
    이 시스템의 라이브는 아카이브 용도이므로, 이미 촬영/처리된 분량은
    시간이 걸리더라도 전부 유튜브로 올려야 한다.
    이 대기 중 Ctrl+C를 한 번 더 누르면 강제 종료된다.
    """
    active = [r for r in relays if r is not None and r.pending_bytes() > 0]
    if not active:
        return

    total_mb = sum(r.pending_bytes() for r in active) / (1024 * 1024)
    print("\n" + "─" * 60)
    print(f"📤 남은 송출 대기분 {total_mb:.1f}MB를 마저 보내는 중입니다.")
    print("   유튜브 업로드가 끝날 때까지 프로그램을 유지합니다.")
    print("   (강제 종료하려면 Ctrl+C를 한 번 더 누르세요)")
    print("─" * 60)

    started_at = time.time()
    last_report = 0.0
    last_pending = None
    stalled_since = None

    try:
        while True:
            # 완료 판정은 sender 스레드의 정상 종료(_sender_done)로 한다.
            # 바이트 누계는 재전송 때문에 오차가 남을 수 있어 기준으로 못 쓴다.
            if all(r._sender_done.is_set() for r in active):
                break

            pending_map = {r.name: r.pending_bytes() for r in active}
            pending_total = sum(pending_map.values())

            # 송출 스레드가 정상 완료(_sender_done)가 아닌 채로 죽었다면
            # 더 보낼 방법이 없으므로 중단한다.
            stuck = [
                r for r in active
                if not r._sender_done.is_set()
                and not any(
                    t.is_alive() for t in r._threads
                    if t.name.endswith("-sender")
                )
            ]
            if stuck:
                names = ", ".join(r.name for r in stuck)
                print(f"⚠️  {names} 송출 스레드가 중단되어 남은 분량을 보낼 수 없습니다.")
                break

            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - started_at
                detail = ", ".join(
                    f"{name} {b / (1024 * 1024):.1f}MB"
                    for name, b in pending_map.items()
                )
                if last_pending is not None and elapsed > 0:
                    drained = last_pending - pending_total
                    rate = drained / max(1e-6, now - last_report)
                    if rate > 1024:
                        eta_sec = pending_total / rate
                        eta = f", 예상 {eta_sec / 60:.1f}분 남음"
                    else:
                        eta = ", 진행 없음(네트워크 확인 필요)"
                else:
                    eta = ""
                print(f"   [송출 대기] {detail}{eta}")

                if last_pending is not None and pending_total >= last_pending:
                    stalled_since = stalled_since or now
                    if now - stalled_since >= 120:
                        print(
                            "⚠️  2분 넘게 전송이 진행되지 않고 있습니다. "
                            "네트워크를 확인하거나 Ctrl+C로 종료하세요."
                        )
                        stalled_since = now
                else:
                    stalled_since = None

                last_pending = pending_total
                last_report = now

            time.sleep(1.0)
    except KeyboardInterrupt:
        remain = sum(r.pending_bytes() for r in active) / (1024 * 1024)
        print(f"\n[강제 종료] 미전송 {remain:.1f}MB가 남았습니다.")
        return

    print("✅ 남은 송출 대기분을 모두 전송했습니다.")


def start_output_stream_process(width, height, fps, spool_dir):
    """처리 결과를 유튜브로 송출한다.

    구조: python(프레임) -> 인코딩 전용 ffmpeg(mpegts, pipe:1)
                          -> DiskSpoolRelay(디스크 스풀 + 자동 재연결) -> RTMP

    python이 stdin.write로 프레임을 넣는 대상은 네트워크와 무관한
    로컬 인코더뿐이므로, 업로드가 느려져도 추론 루프가 블로킹되지 않는다.
    송출 대기분은 디스크에 쌓여 한 프레임도 유실되지 않는다.
    """
    if not OUTPUT_STREAM_ENABLED:
        return None, None, "disabled"

    if (
        not OUTPUT_STREAM_RTMP_URL
        or "YOUR_STREAM_KEY" in OUTPUT_STREAM_RTMP_URL
    ):
        print("⚠️  OUTPUT_STREAM_ENABLED=True 이지만 OUTPUT_STREAM_RTMP_URL 설정이 비어 있습니다.")
        print("   로컬 저장만 계속 진행합니다.")
        return None, None, "start_failed"

    # YouTube 권장에 맞춰 2초 간격으로 키프레임을 고정한다.
    gop_size = max(1, int(round(fps * 2)))
    video_args, bitrate = build_video_encoder_args(width, height, fps)

    encode_cmd = [
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
    ] + video_args + [
        "-g", str(gop_size),
        "-keyint_min", str(gop_size),
        "-sc_threshold", "0",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-shortest",
        "-f", "mpegts",
        "pipe:1",
    ]

    try:
        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("⚠️  ffmpeg를 찾을 수 없습니다. 로컬 저장만 계속 진행합니다.")
        return None, None, "start_failed"
    except Exception as e:
        print(f"⚠️  라이브 송출 인코더 시작 실패: {e}")
        print("   로컬 저장만 계속 진행합니다.")
        return None, None, "start_failed"

    start_named_ffmpeg_log_thread("output-encode", encode_proc)

    relay = DiskSpoolRelay("처리본", encode_proc.stdout,
                           OUTPUT_STREAM_RTMP_URL, spool_dir)
    if not relay.start():
        print("⚠️  송출 relay 시작 실패. 로컬 저장만 계속 진행합니다.")
        encode_proc.kill()
        return None, None, "start_failed"

    start_named_ffmpeg_log_thread("output-push", relay._push_proc)

    print(f"📡 라이브 송출 시작 → {OUTPUT_STREAM_RTMP_URL}")
    print(
        f"   encoder={STREAM_VIDEO_ENCODER}  bitrate={bitrate['b:v']}  "
        f"maxrate={bitrate['maxrate']}  bufsize={bitrate['bufsize']}  cq={bitrate['cq']}"
    )
    print(f"   키프레임 간격: 약 {gop_size / fps:.1f}초 ({gop_size} 프레임)")
    print(f"   송출 스풀: {spool_dir} (무손실, 끊기면 자동 재연결)")
    return encode_proc, relay, "streaming"


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


def stream_status_text(status):
    return {
        "disabled": "사용 안 함",
        "start_failed": "시작 실패",
        "streaming": "종료 시점까지 송출 유지",
        "disconnected": "중간에 연결 끊김",
    }.get(status, status)


def make_spool_session_dirs():
    session_name = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SPOOL_ROOT_DIR, session_name)
    segments_dir = os.path.join(session_dir, "input_segments")
    markers_dir = os.path.join(session_dir, "processed_markers")
    stream_spool_dir = os.path.join(session_dir, "stream_spool")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)
    os.makedirs(stream_spool_dir, exist_ok=True)
    return session_dir, segments_dir, markers_dir, stream_spool_dir


def build_spool_record_command(stream_url, segments_dir):
    segment_pattern = os.path.join(segments_dir, "input_%06d.mkv")
    return [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_at_eof", "1",
        "-reconnect_delay_max", "5",
        "-rw_timeout", "15000000",
        "-fflags", "+genpts",
        "-i", stream_url,
        "-map", "0:v:0",
        "-an",
        "-c:v", "copy",
        "-f", "segment",
        "-segment_time", str(SPOOL_SEGMENT_SECONDS),
        "-reset_timestamps", "1",
        "-segment_format", "matroska",
        segment_pattern,
    ]


def start_spool_recorder(stream_url, segments_dir):
    cmd = build_spool_record_command(stream_url, segments_dir)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("❌ ffmpeg를 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"❌ 입력 스풀 레코더 시작 실패: {e}")
        return None

    print(f"📼 입력 세그먼트 저장 시작 → {segments_dir}")
    print(f"   세그먼트 길이: {SPOOL_SEGMENT_SECONDS}초")
    print(
        f"   입력 세그먼트 최종 보존: {'ON' if KEEP_INPUT_SEGMENTS else 'OFF'} "
        f"(처리 후 삭제: {'ON' if DELETE_INPUT_SEGMENTS_AFTER_PROCESS else 'OFF'})"
    )
    start_ffmpeg_log_thread(proc, name="ffmpeg:spool-input")
    return proc


def marker_path_for(segment_path, markers_dir):
    stem = os.path.splitext(os.path.basename(segment_path))[0]
    return os.path.join(markers_dir, f"{stem}.done")


def list_processable_segments(segments_dir, markers_dir, recorder_alive):
    segment_files = sorted(
        os.path.join(segments_dir, name)
        for name in os.listdir(segments_dir)
        if name.startswith("input_") and name.endswith(".mkv")
    )

    if recorder_alive:
        candidates = segment_files[:-1]
    else:
        candidates = segment_files

    ready = []
    for path in candidates:
        marker_path = marker_path_for(path, markers_dir)
        if os.path.exists(marker_path):
            continue
        try:
            if os.path.getsize(path) <= 0:
                continue
        except OSError:
            continue
        ready.append(path)
    return ready


def probe_segment(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def mark_segment_processed(segment_path, markers_dir):
    marker_path = marker_path_for(segment_path, markers_dir)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("done\n")

    if DELETE_INPUT_SEGMENTS_AFTER_PROCESS and not KEEP_INPUT_SEGMENTS:
        try:
            os.remove(segment_path)
        except FileNotFoundError:
            pass

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
    유튜브 라이브/VOD 또는 RTMP 입력을 디스크 세그먼트로 먼저 저장한 뒤,
    완료된 세그먼트를 순서대로 추론해서 유튜브 RTMP로 재송출하거나 로컬 mp4로 저장한다.
    """
    print(f"📡 스트림 모드 입력: {INPUT_STREAM_URL}")
    print(f"   종료: Ctrl+C\n")

    stream_url = resolve_stream_url(INPUT_STREAM_URL)
    if not stream_url:
        return

    if KEEP_INPUT_SEGMENTS and DELETE_INPUT_SEGMENTS_AFTER_PROCESS:
        print("❌ KEEP_INPUT_SEGMENTS=True 와 DELETE_INPUT_SEGMENTS_AFTER_PROCESS=True는 동시에 사용할 수 없습니다.")
        return

    (session_dir, segments_dir, markers_dir,
     stream_spool_dir) = make_spool_session_dirs()
    print(f"📁 세션 폴더: {session_dir}")

    recorder_proc = start_spool_recorder(stream_url, segments_dir)
    if recorder_proc is None:
        return

    seg_model, wheel_det, trackers = load_models()

    saved_out = None
    stream_proc = None
    stream_relay = None
    stream_status = "disabled"
    output_size = None
    output_fps = 0.0
    output_frame_count = 0

    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    fps_ctr = FPSCounter(window=30, print_interval=30)
    frame_idx = 0
    infer_count = 0
    total_processed_segments = 0
    total_processed_frames = 0
    last_wait_log_at = 0.0

    try:
        while True:
            recorder_alive = recorder_proc.poll() is None
            pending_segments = list_processable_segments(
                segments_dir,
                markers_dir,
                recorder_alive,
            )

            if not pending_segments:
                if not recorder_alive:
                    break

                now = time.time()
                if now - last_wait_log_at >= 10.0:
                    saved_segments = len([
                        name for name in os.listdir(segments_dir)
                        if name.startswith("input_") and name.endswith(".mkv")
                    ])
                    processed_segments = len([
                        name for name in os.listdir(markers_dir)
                        if name.endswith(".done")
                    ])
                    print(
                        f"  [spool] 대기 중... "
                        f"(saved={saved_segments}, processed={processed_segments}, session={os.path.basename(session_dir)})"
                    )
                    last_wait_log_at = now
                time.sleep(SPOOL_POLL_INTERVAL_SEC)
                continue

            for segment_path in pending_segments:
                meta = probe_segment(segment_path)
                if meta is None:
                    print(f"⚠️  세그먼트 메타데이터 확인 실패: {segment_path}")
                    mark_segment_processed(segment_path, markers_dir)
                    continue

                if output_size is None:
                    output_size = (meta["width"], meta["height"])
                    output_fps = meta["fps"] / FRAME_SKIP
                    if not output_fps or np.isnan(output_fps) or output_fps <= 0:
                        output_fps = 30.0

                    print(f"✅ 스풀 처리 시작: {meta['width']}×{meta['height']} @ {meta['fps']:.1f}fps")
                    print(f"   프레임 스킵: {FRAME_SKIP}")
                    print(f"   처리 결과 출력 FPS: {output_fps:.1f}")

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

                    stream_proc, stream_relay, stream_status = start_output_stream_process(
                        output_size[0],
                        output_size[1],
                        output_fps,
                        stream_spool_dir,
                    )

                    print(f"   로컬 저장: {'ON' if saved_out else 'OFF'}")
                    if SAVE_STREAM_OUTPUT:
                        print(f"   저장 경로: {STREAM_OUTPUT_PATH}")
                    print(f"   유튜브 송출: {'ON' if stream_proc else 'OFF'}\n")

                print(
                    f"🎞️  세그먼트 처리 시작: {os.path.basename(segment_path)} "
                    f"({meta['width']}×{meta['height']} @ {meta['fps']:.1f}fps, frames={meta['frames']})"
                )

                cap = cv2.VideoCapture(segment_path)
                if not cap.isOpened():
                    print(f"⚠️  세그먼트를 열 수 없습니다: {segment_path}")
                    mark_segment_processed(segment_path, markers_dir)
                    continue

                segment_written = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

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

                    if saved_out is not None:
                        saved_out.write(vis_for_output)

                    if stream_proc is not None and stream_proc.stdin is not None:
                        try:
                            stream_proc.stdin.write(vis_for_output.tobytes())
                        except (BrokenPipeError, OSError):
                            # 네트워크 문제는 relay가 흡수/재연결하므로,
                            # 여기까지 오는 것은 로컬 인코더가 죽은 경우다.
                            print("⚠️  송출 인코더가 종료되었습니다. 로컬 저장만 계속 진행합니다.")
                            if stream_relay is not None:
                                stream_relay.stop()
                                stream_relay = None
                            stop_stream_process(stream_proc)
                            stream_proc = None
                            stream_status = "disconnected"

                    output_frame_count += 1
                    frame_idx += 1
                    infer_count += 1
                    total_processed_frames += 1
                    segment_written += 1

                    if fps_ctr.should_print():
                        rss_mb = get_memory_usage_mb()
                        rss_text = f"{rss_mb:.1f}MB" if rss_mb is not None else "n/a"
                        saved_segments = len([
                            name for name in os.listdir(segments_dir)
                            if name.startswith("input_") and name.endswith(".mkv")
                        ])
                        processed_segments = len([
                            name for name in os.listdir(markers_dir)
                            if name.endswith(".done")
                        ])
                        backlog = max(0, saved_segments - processed_segments)
                        print(
                            f"  [rtmp-spool] 추론 {fps_ctr.fps():.1f} fps  "
                            f"(processed={infer_count}, written={output_frame_count}, "
                            f"segments={total_processed_segments}, backlog={backlog}, "
                            f"rss={rss_text}, source={output_size[0]}x{output_size[1]})"
                        )

                    if SHOW_WINDOW:
                        cv2.imshow("Drone Referee - RTMP", vis_for_output)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                cap.release()
                mark_segment_processed(segment_path, markers_dir)
                total_processed_segments += 1

                rss_mb = get_memory_usage_mb()
                rss_text = f"{rss_mb:.1f}MB" if rss_mb is not None else "n/a"
                saved_segments = len([
                    name for name in os.listdir(segments_dir)
                    if name.startswith("input_") and name.endswith(".mkv")
                ])
                processed_segments = len([
                    name for name in os.listdir(markers_dir)
                    if name.endswith(".done")
                ])
                backlog = max(0, saved_segments - processed_segments)
                print(
                    f"  [rtmp-spool] segment={os.path.basename(segment_path)} done  "
                    f"(segment_frames={segment_written}, total_frames={total_processed_frames}, "
                    f"segments_done={total_processed_segments}, backlog={backlog}, "
                    f"infer={fps_ctr.fps():.1f}fps, rss={rss_text})"
                )

    except KeyboardInterrupt:
        print("\n[중단] 사용자 요청으로 스풀 처리를 종료합니다.")
    finally:
        if recorder_proc.poll() is None:
            stop_stream_process(recorder_proc)
        if stream_proc is not None:
            # 인코더 입력을 닫아 남은 프레임을 스풀로 흘려보낸다.
            try:
                if stream_proc.stdin:
                    stream_proc.stdin.close()
            except Exception:
                pass

        if DRAIN_SPOOL_ON_EXIT:
            drain_relays_on_exit(stream_relay)

        if stream_relay is not None:
            stream_relay.stop()
        if stream_proc is not None:
            stop_stream_process(stream_proc)
            stream_proc = None
        if saved_out:
            saved_out.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print(f"\n✅ 종료  |  추론 평균 FPS: {fps_ctr.fps():.1f}")
    print(f"   처리 세그먼트 수: {total_processed_segments}")
    print(f"   처리 프레임 수: {infer_count}")
    print(f"   출력 프레임 수: {output_frame_count}")
    if SAVE_STREAM_OUTPUT and saved_out is not None:
        print(f"   결과 저장 → {STREAM_OUTPUT_PATH}")
    print(f"   유튜브 송출 상태: {stream_status_text(stream_status)}")
    if stream_relay is not None:
        print(f"     └ {stream_relay.summary()}")
    print(f"   세션 폴더: {session_dir}")


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

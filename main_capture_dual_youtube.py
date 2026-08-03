"""
main_capture_dual_youtube.py

캡처보드(USB 카메라) 입력을 ffmpeg 하나가 단독 점유해서
1) 원본 영상을 유튜브 라이브 1로 송출하고
2) 같은 입력을 디스크 세그먼트로 스풀한 뒤
3) 완료된 세그먼트를 순서대로 추론해서 유튜브 라이브 2로 송출한다.

이 파일은 main.py를 import하지 않는 독립 실행 파일이다.
"""

import os
import sys
import time
import atexit
import signal
import threading
import subprocess
import traceback
import faulthandler
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import resource
except ImportError:
    resource = None

import detector as detector_module
import lane_checker as lane_checker_module
import violation_tracker as violation_tracker_module
from detector import WheelDetector
from preprocess import _apply_clahe
from violation_tracker import ViolationTracker


# ──────────────────────────────────────────────────────────
#  캡처보드 입력 설정
# ──────────────────────────────────────────────────────────
CAM_NUM = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30.0
CAMERA_FOURCC = "MJPG"

SHOW_WINDOW = False
FRAME_SKIP = 2
PROCESS_LONG_SIDE = 1280
DRAW_LANE_VIS = True # 점선 표시 여부
REPOSITION_TOGGLE_DELAY_SEC = 0.0

# 모델 / 추론 설정
SEG_MODEL_PATH = "model/best_seg_rev03_FP16.engine"
POSE_MODEL_PATH = "model/best_referee_FP16.engine"
MAX_WHEELS = 4

# CLAHE 설정
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (4, 4)

# 차선 segmentation 설정
SEG_CONF_THRESH = 0.5
SEG_IMGSZ = 1280
LANE_MIN_AREA = 50
LANE_MAX_HALF_WIDTH = 50
LANE_CENTERLINE_MIN_ASPECT = 0.0
LANE_EXCLUDE_CLASSES = ["crosswalk"]
LANE_PRINT_CLASSES = False

# 바퀴 pose 설정
POSE_DEVICE = "cuda:0"
POSE_CONF_THRESH = 0.5
POSE_KP_THRESH = 0.7

# 침범 시간 설정
VIOLATION_HOLD_SEC = 0.1

# 원본 라이브 송출
RAW_STREAM_ENABLED = True
RAW_STREAM_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/9hdv-mff8-h2kv-uyhh-cv8u"  # 자동화연구실 계정
SAVE_RAW_OUTPUT = False
RAW_OUTPUT_PATH = "capture_raw_output.mp4"

# 처리 결과 라이브 송출
PROCESSED_STREAM_ENABLED = True
PROCESSED_STREAM_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/e6m2-vfja-yz2m-sjab-fsj7"  # 성균관대학교노승윤 계정
SAVE_PROCESSED_OUTPUT = False
PROCESSED_OUTPUT_PATH = "capture_processed_output.mp4"

# 입력 스풀 설정
SPOOL_ROOT_DIR = "spool_capture_sessions"
SPOOL_SEGMENT_SECONDS = 10
KEEP_INPUT_SEGMENTS = False
DELETE_INPUT_SEGMENTS_AFTER_PROCESS = True
SPOOL_POLL_INTERVAL_SEC = 0.5

# RTMP relay 설정 (디스크 스풀 송출 + 자동 재연결)
# RTMP_SPOOL_PART_MB       : 송출 스풀을 나눌 파트 파일 크기. 재연결 시 현재 파트를 처음부터 다시 보내므로 작을수록 중복 전송이 줄고, 클수록 파일 수가 준다.
# RTMP_SPOOL_WARN_GB       : 송출 대기분이 이보다 커지면 경고를 띄운다.
# RTMP_RECONNECT_DELAY_SEC : push 프로세스가 죽었을 때 재연결까지 대기 시간.
# RTMP_RELAY_CHUNK_BYTES   : 한 번에 읽고 쓰는 크기(mpegts 188B 배수).
RTMP_SPOOL_PART_MB = 4
RTMP_SPOOL_WARN_GB = 20
RTMP_RECONNECT_DELAY_SEC = 3.0
RTMP_RELAY_CHUNK_BYTES = 188 * 64
# 종료 시 남은 송출 대기분을 끝까지 보낼지 여부.
# True면 Ctrl+C 후에도 스풀을 다 비울 때까지 송출을 계속한다.
# (그 동안 Ctrl+C를 한 번 더 누르면 강제 종료)
DRAIN_SPOOL_ON_EXIT = True

# 출력 인코더 설정
STREAM_VIDEO_ENCODER = "h264_nvenc"
CPU_FALLBACK_VIDEO_ENCODER = "libx264"
NVENC_PRESET = "p5"
X264_PRESET = "veryfast"

RUNTIME_LOG_PATH = "logs/drone_referee_capture_dual.log"
# ──────────────────────────────────────────────────────────


_RUNTIME_LOG_FILE = None
_EXIT_RECORDED = False


def apply_module_overrides():
    detector_module.DEVICE = POSE_DEVICE
    detector_module.CONF_THRESH = POSE_CONF_THRESH
    detector_module.KP_THRESH = POSE_KP_THRESH

    lane_checker_module.CONF_THRESH = SEG_CONF_THRESH
    lane_checker_module.IMGSZ = SEG_IMGSZ
    lane_checker_module.MIN_AREA = LANE_MIN_AREA
    lane_checker_module.MAX_HALF_WIDTH = LANE_MAX_HALF_WIDTH
    lane_checker_module.CENTERLINE_MIN_ASPECT = LANE_CENTERLINE_MIN_ASPECT
    lane_checker_module.EXCLUDE_CLASSES = list(LANE_EXCLUDE_CLASSES)
    lane_checker_module.PRINT_CLASSES = LANE_PRINT_CLASSES

    violation_tracker_module.VIOLATION_HOLD_SEC = VIOLATION_HOLD_SEC


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


class RefereeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._paused = False
        self._scheduled_paused = False
        self._pending_toggles = []

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


def start_input_listener(state):
    print("💡 터미널에서 Enter 키를 누르면 재위치 모드 ON/OFF 전환")

    def _listen():
        while True:
            try:
                input()
                state.toggle()
            except EOFError:
                break

    thread = threading.Thread(target=_listen, daemon=True)
    thread.start()


class FPSCounter:
    def __init__(self, window=30, print_interval=30):
        self.window = window
        self.print_interval = print_interval
        self.times = []
        self.frame_count = 0
        self._last_tick = None

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


def build_lane_data(seg_model, frame):
    instance_masks = lane_checker_module.run_segmentation(seg_model, frame)
    lane_data = []

    for mask_bool in instance_masks:
        mask_bin = (mask_bool * 255).astype(np.uint8)
        contours = lane_checker_module.filter_small_contours(mask_bin)
        if not contours:
            continue

        half_width = lane_checker_module.get_instance_half_width(contours)
        centerline = lane_checker_module.get_centerline(mask_bin)
        if centerline is None:
            continue

        lane_data.append({
            "mask_bin": mask_bin,
            "contours": contours,
            "half_width": half_width,
            "centerline": centerline,
        })

    return lane_data


def draw_dashed_line(vis, pt1, pt2, color, thickness=2, dash_len=18, gap_len=10):
    start = np.array(pt1, dtype=np.float32)
    end = np.array(pt2, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return

    direction = delta / length
    pos = 0.0
    while pos < length:
        dash_start = start + direction * pos
        dash_end = start + direction * min(pos + dash_len, length)
        cv2.line(
            vis,
            tuple(np.round(dash_start).astype(int)),
            tuple(np.round(dash_end).astype(int)),
            color,
            thickness,
        )
        pos += dash_len + gap_len


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
        kx, ky = wheel["keypoint"]
        confirmed = state["confirmed"]
        violated = state["violated"]

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
            cv2.putText(
                vis,
                f"VIOLATION {state['duration'] * 1000:.0f}ms",
                (x1, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    if paused:
        cv2.putText(
            vis,
            "REPOSITIONING",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 200, 255),
            3,
        )
    else:
        status = "VIOLATION" if any_confirmed else "NORMAL"
        status_color = (0, 0, 255) if any_confirmed else (0, 255, 0)
        cv2.putText(
            vis,
            status,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            status_color,
            3,
        )

    info_text = f"wheels:{len(wheels)}"
    if DRAW_LANE_VIS:
        info_text += f"  lanes:{len(lane_data)}"
    cv2.putText(
        vis,
        info_text,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return vis


def process_frame(seg_model, wheel_det, trackers, frame, state):
    lane_data = build_lane_data(seg_model, frame)
    wheels = wheel_det.predict(frame)
    paused = state.paused

    tracker_states = []
    for w_idx, wheel in enumerate(wheels[:MAX_WHEELS]):
        if paused:
            trackers[w_idx].reset()
            tracker_states.append({
                "wheel_id": w_idx,
                "violated": False,
                "confirmed": False,
                "duration": 0.0,
            })
        else:
            kx, ky = wheel["keypoint"]
            lane_checker_module.compute_distances(lane_data, kx, ky)
            viol_results = lane_checker_module.check_violation(lane_data)
            frame_violated = any(vr["violated"] for vr in viol_results)
            tracker_states.append(trackers[w_idx].update(frame_violated))

    for idx in range(len(wheels), MAX_WHEELS):
        trackers[idx].reset()

    vis = draw_frame(frame, lane_data, wheels[:MAX_WHEELS], tracker_states, paused)
    return vis, tracker_states


def load_models():
    apply_module_overrides()
    print("🔍 모델 로딩 중...")
    seg_model = YOLO(SEG_MODEL_PATH, task="segment")
    wheel_det = WheelDetector(model_path=POSE_MODEL_PATH)
    trackers = [ViolationTracker(wheel_id=i) for i in range(MAX_WHEELS)]
    print("✅ 모델 로딩 완료\n")
    return seg_model, wheel_det, trackers


def _stream_status_text(status):
    return {
        "disabled": "사용 안 함",
        "start_failed": "시작 실패",
        "streaming": "송출 중",
        "disconnected": "중간에 연결 끊김",
    }.get(status, status)


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


def start_named_ffmpeg_log_thread(name, proc):
    if proc.stderr is None:
        return

    def _drain():
        try:
            for raw_line in iter(proc.stderr.readline, b""):
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"[ffmpeg:{name}] {line}")
        except Exception as e:
            print(f"[ffmpeg:{name}] stderr 읽기 실패: {e}")
        finally:
            try:
                return_code = proc.wait(timeout=0.1)
                print(f"[ffmpeg:{name}] 프로세스 종료 code={return_code}")
            except subprocess.TimeoutExpired:
                pass

    thread = threading.Thread(target=_drain, name=f"ffmpeg-{name}", daemon=True)
    thread.start()


def stop_named_stream_process(name, proc):
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


def start_output_stream_process(name, enabled, rtmp_url, width, height, fps,
                                spool_dir):
    if not enabled:
        return None, "disabled"

    if not rtmp_url or "YOUR_" in rtmp_url:
        print(f"⚠️  {name} 송출 URL이 비어 있거나 placeholder 상태입니다.")
        print("   로컬 저장만 계속 진행합니다.")
        return None, "start_failed"

    gop_size = max(1, int(round(fps * 2)))
    video_args, bitrate = build_video_encoder_args(width, height, fps)

    # 인코딩 전용 프로세스: 네트워크에 관여하지 않고 mpegts를 pipe:1로만 내보낸다.
    # 파이썬은 이 프로세스의 stdin에만 프레임을 쓰므로, RTMP가 느려져도
    # 추론 루프가 블로킹되지 않는다.
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
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print(f"⚠️  ffmpeg를 찾을 수 없습니다. {name} 스트림은 건너뜁니다.")
        return None, None, "start_failed"
    except Exception as e:
        print(f"⚠️  {name} 라이브 송출 시작 실패: {e}")
        return None, None, "start_failed"

    start_named_ffmpeg_log_thread(f"{name}-encode", proc)

    relay = DiskSpoolRelay(name, proc.stdout, rtmp_url, spool_dir)
    if not relay.start():
        print(f"⚠️  {name} 송출 relay 시작 실패. 로컬 저장만 계속 진행합니다.")
        proc.kill()
        return None, None, "start_failed"

    start_named_ffmpeg_log_thread(f"{name}-push", relay._push_proc)

    print(f"📡 {name} 송출 시작 → {rtmp_url}")
    print(
        f"   encoder={STREAM_VIDEO_ENCODER}  bitrate={bitrate['b:v']}  "
        f"maxrate={bitrate['maxrate']}  bufsize={bitrate['bufsize']}  cq={bitrate['cq']}"
    )
    print(f"   키프레임 간격: 약 {gop_size / fps:.1f}초 ({gop_size} 프레임)")
    print(f"   송출 스풀: {spool_dir} (무손실, 끊기면 자동 재연결)")
    return proc, relay, "streaming"


def open_named_video_writer(enabled, out_path, fps, size, name):
    if not enabled:
        return None

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        print(f"⚠️  {name} 저장 파일을 열 수 없습니다: {out_path}")
        writer.release()
        return None

    print(f"💾 {name} 저장 → {out_path}")
    return writer


def _round_even(value):
    value = max(2, int(round(value)))
    if value % 2:
        value += 1
    return value


def compute_processing_size(src_w, src_h, long_side=PROCESS_LONG_SIDE):
    if src_w <= 0 or src_h <= 0:
        return (1280, 720)

    longest = max(src_w, src_h)
    if longest <= long_side:
        return (_round_even(src_w), _round_even(src_h))

    scale = long_side / float(longest)
    dst_w = _round_even(src_w * scale)
    dst_h = _round_even(src_h * scale)
    return (dst_w, dst_h)


def make_spool_session_dirs():
    session_name = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SPOOL_ROOT_DIR, session_name)
    segments_dir = os.path.join(session_dir, "input_segments")
    markers_dir = os.path.join(session_dir, "processed_markers")
    raw_spool_dir = os.path.join(session_dir, "stream_spool_raw")
    processed_spool_dir = os.path.join(session_dir, "stream_spool_processed")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)
    os.makedirs(raw_spool_dir, exist_ok=True)
    os.makedirs(processed_spool_dir, exist_ok=True)
    return (session_dir, segments_dir, markers_dir,
            raw_spool_dir, processed_spool_dir)


def list_segment_files(segments_dir):
    return sorted(
        os.path.join(segments_dir, name)
        for name in os.listdir(segments_dir)
        if name.startswith("input_") and name.endswith(".mkv")
    )


def marker_path_for(segment_path, markers_dir):
    stem = os.path.splitext(os.path.basename(segment_path))[0]
    return os.path.join(markers_dir, f"{stem}.done")


def list_processable_segments(segments_dir, markers_dir, recorder_alive):
    segment_files = list_segment_files(segments_dir)

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
        fps = CAMERA_FPS or 30.0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def mark_segment_processed(segment_path, markers_dir):
    marker_path = marker_path_for(segment_path, markers_dir)
    with open(marker_path, "w", encoding="utf-8") as file:
        file.write("done\n")

    if DELETE_INPUT_SEGMENTS_AFTER_PROCESS and not KEEP_INPUT_SEGMENTS:
        try:
            os.remove(segment_path)
        except FileNotFoundError:
            pass


def spool_counters(segments_dir, markers_dir):
    saved_segments = 0
    backlog = 0

    for path in list_segment_files(segments_dir):
        saved_segments += 1
        marker_path = marker_path_for(path, markers_dir)
        if os.path.exists(marker_path):
            continue
        try:
            if os.path.getsize(path) > 0:
                backlog += 1
        except OSError:
            pass

    processed_segments = len([
        name for name in os.listdir(markers_dir)
        if name.endswith(".done")
    ])
    return saved_segments, processed_segments, backlog


def camera_input_format_name():
    code = (CAMERA_FOURCC or "").strip().upper()
    if not code:
        return None

    return {
        "MJPG": "mjpeg",
        "YUYV": "yuyv422",
        "YUY2": "yuyv422",
        "H264": "h264",
        "NV12": "nv12",
    }.get(code, code.lower())


def build_camera_spool_command(segments_dir):
    capture_path = f"/dev/video{CAM_NUM}"
    segment_pattern = os.path.join(segments_dir, "input_%06d.mkv")

    if SAVE_RAW_OUTPUT:
        raw_dir = os.path.dirname(RAW_OUTPUT_PATH)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)

    raw_status = "disabled"
    tee_outputs = [
        (
            "[f=segment:"
            f"segment_time={SPOOL_SEGMENT_SECONDS}:"
            "reset_timestamps=1:"
            "segment_format=matroska]"
            f"{segment_pattern}"
        )
    ]

    if RAW_STREAM_ENABLED:
        if not RAW_STREAM_RTMP_URL or "YOUR_" in RAW_STREAM_RTMP_URL:
            print("⚠️  원본 송출 URL이 비어 있거나 placeholder 상태입니다.")
            print("   입력 세그먼트 저장만 계속 진행합니다.")
            raw_status = "start_failed"
        else:
            # RTMP를 tee에서 직접 물지 않고 pipe:1로만 내보낸다.
            # 실제 유튜브 송출은 별도 프로세스(mbuffer + ffmpeg)가 담당해서
            # 네트워크가 느려져도 이 레코더(디스크 세그먼트 저장)가 막히지 않게 한다.
            tee_outputs.append("[f=mpegts]pipe:1")
            raw_status = "pending"

    if SAVE_RAW_OUTPUT:
        tee_outputs.append(
            "[f=mp4:movflags=+faststart:onfail=ignore]"
            f"{RAW_OUTPUT_PATH}"
        )

    configured_width = CAMERA_WIDTH or 1920
    configured_height = CAMERA_HEIGHT or 1080
    configured_fps = CAMERA_FPS or 30.0
    gop_size = max(1, int(round(configured_fps * 2)))
    video_args, bitrate = build_video_encoder_args(
        configured_width,
        configured_height,
        configured_fps,
    )

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-fflags", "+genpts",
        "-thread_queue_size", "2048",
        "-f", "v4l2",
    ]

    input_format = camera_input_format_name()
    if input_format:
        cmd += ["-input_format", input_format]
    if CAMERA_FPS:
        cmd += ["-framerate", f"{CAMERA_FPS:.3f}"]
    if CAMERA_WIDTH and CAMERA_HEIGHT:
        cmd += ["-video_size", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"]

    cmd += [
        "-i", capture_path,
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
        "-flags", "+global_header",
        "-f", "tee",
        "|".join(tee_outputs),
    ]

    return cmd, raw_status, bitrate, gop_size


def start_raw_stream_relay(recorder_proc, spool_dir):
    """recorder_proc의 stdout(mpegts pipe)을 RtmpRelay로 넘겨 원본을 송출한다.
    relay가 파이썬 메모리에 버퍼링하며 읽어내므로, 네트워크가 느려지거나
    끊겨도 recorder_proc(디스크 세그먼트 저장)는 전혀 영향을 받지 않는다.
    """
    if recorder_proc.stdout is None:
        return None, "start_failed"

    relay = DiskSpoolRelay("원본", recorder_proc.stdout, RAW_STREAM_RTMP_URL, spool_dir)
    if not relay.start():
        return relay, "start_failed"

    start_named_ffmpeg_log_thread("원본-push", relay._push_proc)
    return relay, "streaming"


def format_relay_pending(*relays):
    """실행 중 송출 대기분(아직 유튜브로 못 보낸 분량)을 한 줄로 요약한다.
    네트워크가 나빠지면 이 값이 커지고, 회복되면 줄어든다.
    재연결이 일어나지 않는 종류의 장애(패킷 DROP 등)는 이 값으로만
    관측되므로, 테스트 시 이 수치를 보면 된다.
    """
    parts = []
    for relay in relays:
        if relay is None:
            continue
        pending_mb = relay.pending_bytes() / (1024 * 1024)
        tag = f"{relay.name} {pending_mb:.1f}MB"
        if relay.reconnect_count:
            tag += f"/재연결{relay.reconnect_count}"
        parts.append(tag)
    if not parts:
        return ""
    return "  송출대기[" + ", ".join(parts) + "]"


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


def start_camera_spool_recorder(segments_dir, raw_spool_dir):
    cmd, raw_status, bitrate, gop_size = build_camera_spool_command(segments_dir)

    needs_pipe = raw_status == "pending"
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if needs_pipe else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("❌ ffmpeg를 찾을 수 없습니다.")
        return None, "start_failed", None
    except Exception as e:
        print(f"❌ 캡처 입력 스풀 레코더 시작 실패: {e}")
        return None, "start_failed", None

    print(f"📼 캡처 입력 스풀 시작 → /dev/video{CAM_NUM}")
    print(
        f"   캡처 설정: {CAMERA_WIDTH}×{CAMERA_HEIGHT} @ {CAMERA_FPS:.1f}fps"
        + (f" ({CAMERA_FOURCC})" if CAMERA_FOURCC else "")
    )
    print(f"   입력 세그먼트 저장: {segments_dir}")
    print(f"   세그먼트 길이: {SPOOL_SEGMENT_SECONDS}초")
    print(
        f"   encoder={STREAM_VIDEO_ENCODER}  bitrate={bitrate['b:v']}  "
        f"maxrate={bitrate['maxrate']}  bufsize={bitrate['bufsize']}  cq={bitrate['cq']}"
    )
    print(f"   키프레임 간격: 약 {gop_size / (CAMERA_FPS or 30.0):.1f}초 ({gop_size} 프레임)")
    print(
        f"   입력 세그먼트 최종 보존: {'ON' if KEEP_INPUT_SEGMENTS else 'OFF'} "
        f"(처리 후 삭제: {'ON' if DELETE_INPUT_SEGMENTS_AFTER_PROCESS else 'OFF'})"
    )
    if SAVE_RAW_OUTPUT:
        print(f"   원본 저장 경로: {RAW_OUTPUT_PATH}")
    start_named_ffmpeg_log_thread("capture-spool", proc)

    time.sleep(0.5)
    if proc.poll() is not None:
        print("❌ 캡처 입력 스풀 레코더가 시작 직후 종료되었습니다.")
        print("   런타임 로그와 ffmpeg stderr 로그를 확인해 주세요.")
        return None, "start_failed", None

    raw_relay = None
    if needs_pipe:
        raw_relay, raw_status = start_raw_stream_relay(proc, raw_spool_dir)

    if RAW_STREAM_ENABLED:
        print(f"   원본 송출 상태: {_stream_status_text(raw_status)}")
        if raw_status == "streaming":
            print(
                f"   원본 송출 주소: {RAW_STREAM_RTMP_URL} "
                f"(무손실 디스크 스풀, 끊기면 자동 재연결)"
            )

    return proc, raw_status, raw_relay


def run_camera_dual_stream(state):
    if KEEP_INPUT_SEGMENTS and DELETE_INPUT_SEGMENTS_AFTER_PROCESS:
        print("❌ KEEP_INPUT_SEGMENTS=True 와 DELETE_INPUT_SEGMENTS_AFTER_PROCESS=True는 동시에 사용할 수 없습니다.")
        return

    print(f"📡 캡처 입력 모드: /dev/video{CAM_NUM}")
    print("   구조: 캡처 -> 원본 송출 + 디스크 스풀 -> 세그먼트 FIFO 추론 -> 처리본 송출")
    print("   종료: Ctrl+C\n")

    (session_dir, segments_dir, markers_dir,
     raw_spool_dir, processed_spool_dir) = make_spool_session_dirs()
    print(f"📁 세션 폴더: {session_dir}")

    recorder_proc, raw_status, raw_relay = start_camera_spool_recorder(
        segments_dir, raw_spool_dir
    )
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

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE,
    )
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
                    saved_segments, processed_segments, backlog = spool_counters(
                        segments_dir,
                        markers_dir,
                    )
                    print(
                        f"  [capture-spool] 대기 중... "
                        f"(saved={saved_segments}, processed={processed_segments}, "
                        f"backlog={backlog}, session={os.path.basename(session_dir)})"
                        + format_relay_pending(raw_relay, stream_relay)
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
                    output_size = compute_processing_size(meta["width"], meta["height"])
                    output_fps = meta["fps"] / FRAME_SKIP
                    if not output_fps or np.isnan(output_fps) or output_fps <= 0:
                        output_fps = CAMERA_FPS or 30.0

                    print(f"✅ 스풀 처리 시작: {meta['width']}×{meta['height']} @ {meta['fps']:.1f}fps")
                    print(f"   프레임 스킵: {FRAME_SKIP}")
                    print(
                        f"   처리 입력/송출 해상도: {output_size[0]}×{output_size[1]} "
                        f"(long-side {PROCESS_LONG_SIDE})"
                    )
                    print(f"   처리 결과 출력 FPS: {output_fps:.1f}")

                    saved_out = open_named_video_writer(
                        SAVE_PROCESSED_OUTPUT,
                        PROCESSED_OUTPUT_PATH,
                        output_fps,
                        output_size,
                        "처리본",
                    )
                    stream_proc, stream_relay, stream_status = start_output_stream_process(
                        "처리본",
                        PROCESSED_STREAM_ENABLED,
                        PROCESSED_STREAM_RTMP_URL,
                        output_size[0],
                        output_size[1],
                        output_fps,
                        processed_spool_dir,
                    )

                    print(f"   처리본 송출 상태: {_stream_status_text(stream_status)}")
                    print(f"   처리본 저장: {'ON' if saved_out else 'OFF'}")
                    print(
                        f"   설정: draw_lane_vis={DRAW_LANE_VIS}, "
                        f"lane_min_area={LANE_MIN_AREA}, seg_imgsz={SEG_IMGSZ}, "
                        f"pose_device={POSE_DEVICE}, violation_hold={VIOLATION_HOLD_SEC}s\n"
                    )

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
                    if (frame.shape[1], frame.shape[0]) != output_size:
                        proc_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
                    else:
                        proc_frame = frame

                    proc_frame = _apply_clahe(proc_frame, clahe)
                    vis, _ = process_frame(
                        seg_model,
                        wheel_det,
                        trackers,
                        proc_frame,
                        state,
                    )

                    if (vis.shape[1], vis.shape[0]) != output_size:
                        vis = cv2.resize(vis, output_size)

                    if saved_out is not None:
                        saved_out.write(vis)

                    if stream_proc is not None and stream_proc.stdin is not None:
                        try:
                            stream_proc.stdin.write(vis.tobytes())
                        except (BrokenPipeError, OSError):
                            # 여기까지 오는 것은 네트워크가 아니라 로컬 인코더가
                            # 죽은 경우다(네트워크 문제는 relay가 흡수/재연결한다).
                            print("⚠️  처리본 인코더가 종료되었습니다. 로컬 저장만 계속 진행합니다.")
                            if stream_relay is not None:
                                stream_relay.stop()
                                stream_relay = None
                            stop_named_stream_process("처리본-encode", stream_proc)
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
                        saved_segments, processed_segments, backlog = spool_counters(
                            segments_dir,
                            markers_dir,
                        )
                        print(
                            f"  [capture-spool] 추론 {fps_ctr.fps():.1f} fps  "
                            f"(processed={infer_count}, written={output_frame_count}, "
                            f"segments={total_processed_segments}, backlog={backlog}, "
                            f"saved={saved_segments}, done={processed_segments}, "
                            f"rss={rss_text}, source={meta['width']}x{meta['height']}, "
                            f"proc={output_size[0]}x{output_size[1]})"
                            + format_relay_pending(raw_relay, stream_relay)
                        )

                    if SHOW_WINDOW:
                        cv2.imshow("Capture Dual Stream - Processed", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt

                cap.release()
                mark_segment_processed(segment_path, markers_dir)
                total_processed_segments += 1

                rss_mb = get_memory_usage_mb()
                rss_text = f"{rss_mb:.1f}MB" if rss_mb is not None else "n/a"
                saved_segments, processed_segments, backlog = spool_counters(
                    segments_dir,
                    markers_dir,
                )
                print(
                    f"  [capture-spool] segment={os.path.basename(segment_path)} done  "
                    f"(segment_frames={segment_written}, total_frames={total_processed_frames}, "
                    f"segments_done={total_processed_segments}, backlog={backlog}, "
                    f"saved={saved_segments}, done={processed_segments}, "
                    f"infer={fps_ctr.fps():.1f}fps, rss={rss_text})"
                    + format_relay_pending(raw_relay, stream_relay)
                )

    except KeyboardInterrupt:
        print("\n[중단] 사용자 요청으로 캡처 스풀 처리를 종료합니다.")
    finally:
        if recorder_proc.poll() is None:
            stop_named_stream_process("capture-spool", recorder_proc)
        if stream_proc is not None:
            # 인코더 입력을 닫아 남은 프레임을 스풀로 흘려보낸다.
            try:
                if stream_proc.stdin:
                    stream_proc.stdin.close()
            except Exception:
                pass

        if DRAIN_SPOOL_ON_EXIT:
            drain_relays_on_exit(raw_relay, stream_relay)

        if raw_relay is not None:
            raw_relay.stop()
        if stream_relay is not None:
            stream_relay.stop()
        if stream_proc is not None:
            stop_named_stream_process("처리본-encode", stream_proc)
        if saved_out is not None:
            saved_out.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print(f"\n✅ 종료 | 추론 평균 FPS: {fps_ctr.fps():.1f}")
    print(f"   처리 세그먼트 수: {total_processed_segments}")
    print(f"   처리 프레임 수: {infer_count}")
    print(f"   처리본 출력 프레임 수: {output_frame_count}")
    print(f"   원본 송출 상태: {_stream_status_text(raw_status)}")
    if raw_relay is not None:
        print(f"     └ {raw_relay.summary()}")
    print(f"   처리본 송출 상태: {_stream_status_text(stream_status)}")
    if stream_relay is not None:
        print(f"     └ {stream_relay.summary()}")
    if SAVE_PROCESSED_OUTPUT and saved_out is not None:
        print(f"   처리본 저장 → {PROCESSED_OUTPUT_PATH}")
    print(f"   세션 폴더: {session_dir}")


if __name__ == "__main__":
    setup_runtime_logging()
    state = RefereeState()
    start_input_listener(state)
    run_camera_dual_stream(state)

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
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160
CAMERA_FPS = 30.0
CAMERA_FOURCC = "MJPG"

SHOW_WINDOW = False
FRAME_SKIP = 1
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


def start_output_stream_process(name, enabled, rtmp_url, width, height, fps):
    if not enabled:
        return None, "disabled"

    if not rtmp_url or "YOUR_" in rtmp_url:
        print(f"⚠️  {name} 송출 URL이 비어 있거나 placeholder 상태입니다.")
        print("   로컬 저장만 계속 진행합니다.")
        return None, "start_failed"

    gop_size = max(1, int(round(fps * 2)))
    video_args, bitrate = build_video_encoder_args(width, height, fps)
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
        "-f", "flv",
        rtmp_url,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print(f"⚠️  ffmpeg를 찾을 수 없습니다. {name} 스트림은 건너뜁니다.")
        return None, "start_failed"
    except Exception as e:
        print(f"⚠️  {name} 라이브 송출 시작 실패: {e}")
        return None, "start_failed"

    print(f"📡 {name} 송출 시작 → {rtmp_url}")
    print(
        f"   encoder={STREAM_VIDEO_ENCODER}  bitrate={bitrate['b:v']}  "
        f"maxrate={bitrate['maxrate']}  bufsize={bitrate['bufsize']}  cq={bitrate['cq']}"
    )
    print(f"   키프레임 간격: 약 {gop_size / fps:.1f}초 ({gop_size} 프레임)")
    start_named_ffmpeg_log_thread(name, proc)
    return proc, "streaming"


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
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(markers_dir, exist_ok=True)
    return session_dir, segments_dir, markers_dir


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

    if recorder_alive and len(segment_files) > 1:
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
            tee_outputs.append(f"[f=flv:onfail=ignore]{RAW_STREAM_RTMP_URL}")
            raw_status = "streaming"

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


def start_camera_spool_recorder(segments_dir):
    cmd, raw_status, bitrate, gop_size = build_camera_spool_command(segments_dir)

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("❌ ffmpeg를 찾을 수 없습니다.")
        return None, "start_failed"
    except Exception as e:
        print(f"❌ 캡처 입력 스풀 레코더 시작 실패: {e}")
        return None, "start_failed"

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
    if RAW_STREAM_ENABLED:
        print(f"   원본 송출 상태: {_stream_status_text(raw_status)}")
        if raw_status == "streaming":
            print(f"   원본 송출 주소: {RAW_STREAM_RTMP_URL}")
    if SAVE_RAW_OUTPUT:
        print(f"   원본 저장 경로: {RAW_OUTPUT_PATH}")
    start_named_ffmpeg_log_thread("capture-spool", proc)

    time.sleep(0.5)
    if proc.poll() is not None:
        print("❌ 캡처 입력 스풀 레코더가 시작 직후 종료되었습니다.")
        print("   런타임 로그와 ffmpeg stderr 로그를 확인해 주세요.")
        return None, "start_failed"

    return proc, raw_status


def run_camera_dual_stream(state):
    if KEEP_INPUT_SEGMENTS and DELETE_INPUT_SEGMENTS_AFTER_PROCESS:
        print("❌ KEEP_INPUT_SEGMENTS=True 와 DELETE_INPUT_SEGMENTS_AFTER_PROCESS=True는 동시에 사용할 수 없습니다.")
        return

    print(f"📡 캡처 입력 모드: /dev/video{CAM_NUM}")
    print("   구조: 캡처 -> 원본 송출 + 디스크 스풀 -> 세그먼트 FIFO 추론 -> 처리본 송출")
    print("   종료: Ctrl+C\n")

    session_dir, segments_dir, markers_dir = make_spool_session_dirs()
    print(f"📁 세션 폴더: {session_dir}")

    recorder_proc, raw_status = start_camera_spool_recorder(segments_dir)
    if recorder_proc is None:
        return

    seg_model, wheel_det, trackers = load_models()

    saved_out = None
    stream_proc = None
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
                    stream_proc, stream_status = start_output_stream_process(
                        "처리본",
                        PROCESSED_STREAM_ENABLED,
                        PROCESSED_STREAM_RTMP_URL,
                        output_size[0],
                        output_size[1],
                        output_fps,
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
                            print("⚠️  처리본 유튜브 송출 연결이 끊겼습니다. 로컬 저장만 계속 진행합니다.")
                            stop_named_stream_process("processed", stream_proc)
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
                )

    except KeyboardInterrupt:
        print("\n[중단] 사용자 요청으로 캡처 스풀 처리를 종료합니다.")
    finally:
        if recorder_proc.poll() is None:
            stop_named_stream_process("capture-spool", recorder_proc)
        if stream_proc is not None:
            stop_named_stream_process("processed", stream_proc)
        if saved_out is not None:
            saved_out.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print(f"\n✅ 종료 | 추론 평균 FPS: {fps_ctr.fps():.1f}")
    print(f"   처리 세그먼트 수: {total_processed_segments}")
    print(f"   처리 프레임 수: {infer_count}")
    print(f"   처리본 출력 프레임 수: {output_frame_count}")
    print(f"   원본 송출 상태: {_stream_status_text(raw_status)}")
    print(f"   처리본 송출 상태: {_stream_status_text(stream_status)}")
    if SAVE_PROCESSED_OUTPUT and saved_out is not None:
        print(f"   처리본 저장 → {PROCESSED_OUTPUT_PATH}")
    print(f"   세션 폴더: {session_dir}")


if __name__ == "__main__":
    setup_runtime_logging()
    state = RefereeState()
    start_input_listener(state)
    run_camera_dual_stream(state)

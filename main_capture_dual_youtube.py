"""
main_capture_dual_youtube.py

캡처보드(USB 카메라) 입력을 main.py 하나가 단독 점유해서
1) 원본 영상을 유튜브 라이브 1로 송출
2) 추론 결과 영상을 유튜브 라이브 2로 송출
3) 필요하면 원본/결과 영상을 로컬에도 저장

기존 main.py는 유튜브 입력 -> 유튜브 출력 구조로 그대로 둔다.
이 파일은 캡처보드 직접 입력 전용 실행 파일이다.
"""

import os
import time
import threading
import subprocess
from collections import deque

import cv2
import numpy as np

import main as shared_main


# ──────────────────────────────────────────────────────────
#  캡처보드 입력 설정
# ──────────────────────────────────────────────────────────
CAM_NUM = 0
CAMERA_BACKEND = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160
CAMERA_FPS = 30.0
CAMERA_FOURCC = "MJPG"
CAMERA_READ_TIMEOUT_SEC = 2.0
CAMERA_REOPEN_RETRIES = 10
CAMERA_REOPEN_DELAY_SEC = 2.0

# 추론은 최신 프레임 기준으로만 수행하고, 출력은 원본 시간축을 유지한다.
FRAME_SKIP = 1
SHOW_WINDOW = False
PROCESS_LONG_SIDE = 1280

# 원본 라이브 송출
RAW_STREAM_ENABLED = True
RAW_STREAM_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/9hdv-mff8-h2kv-uyhh-cv8u" # 자동화연구실 계정
SAVE_RAW_OUTPUT = False
RAW_OUTPUT_PATH = "capture_raw_output.mp4"

# 처리 결과 라이브 송출
PROCESSED_STREAM_ENABLED = True
PROCESSED_STREAM_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/e6m2-vfja-yz2m-sjab-fsj7" # 성균관대학교노승윤 계정
SAVE_PROCESSED_OUTPUT = False
PROCESSED_OUTPUT_PATH = "capture_processed_output.mp4"

STREAM_VIDEO_ENCODER = "h264_nvenc"
CPU_FALLBACK_VIDEO_ENCODER = "libx264"
NVENC_PRESET = "p5"
X264_PRESET = "veryfast"
RUNTIME_LOG_PATH = "logs/drone_referee_capture_dual.log"
# ──────────────────────────────────────────────────────────


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

    threading.Thread(target=_drain, name=f"ffmpeg-{name}", daemon=True).start()


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
                    frame, _repeats_left = self._queue[0]
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
                        stop_named_stream_process("writer", stream_proc)
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
            stop_named_stream_process("writer", self.stream_proc)
            self.stream_proc = None

    def pending_frames(self):
        with self._cond:
            return len(self._queue)


def build_output_target(name, stream_enabled, stream_url, save_enabled, save_path, fps, size):
    saved_out = open_named_video_writer(save_enabled, save_path, fps, size, name)
    stream_proc, stream_status = start_output_stream_process(
        name, stream_enabled, stream_url, size[0], size[1], fps
    )

    if saved_out is None and stream_proc is None:
        print(f"ℹ️  {name}: 로컬 저장/라이브 송출 모두 비활성 또는 시작 실패")
        return None, None, stream_status

    writer = RealtimeOutputWriter(
        fps,
        saved_out=saved_out,
        stream_proc=stream_proc,
        stream_status=stream_status,
    ).start()
    return writer, saved_out, stream_status


def apply_camera_settings(cap):
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if CAMERA_FOURCC:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
        except Exception:
            pass

    if CAMERA_WIDTH:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    if CAMERA_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if CAMERA_FPS:
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)


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


def open_camera_capture(retries=CAMERA_REOPEN_RETRIES, retry_delay=CAMERA_REOPEN_DELAY_SEC):
    cap = None
    for attempt in range(1, retries + 1):
        if cap is not None:
            cap.release()

        if CAMERA_BACKEND:
            cap = cv2.VideoCapture(CAM_NUM, CAMERA_BACKEND)
        else:
            cap = cv2.VideoCapture(CAM_NUM)

        apply_camera_settings(cap)
        if cap.isOpened():
            return cap

        cap.release()
        cap = cv2.VideoCapture(CAM_NUM)
        apply_camera_settings(cap)
        if cap.isOpened():
            return cap

        cap.release()
        print(f"  ⏳ 카메라 연결 실패, 재시도 중... ({attempt}/{retries})")
        time.sleep(retry_delay)

    return None


class CameraCaptureDistributor:
    def __init__(self, cap, raw_writer=None):
        self.cap = cap
        self.raw_writer = raw_writer
        self._cond = threading.Condition()
        self._stop = False
        self._frame = None
        self._frame_id = 0
        self._fail_count = 0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while True:
            with self._cond:
                if self._stop:
                    return

            ret, frame = self.cap.read()

            if ret and frame is not None and self.raw_writer is not None:
                self.raw_writer.submit_frame(frame, repeat_count=1)

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


def close_output_target(name, writer, saved_out):
    if writer is not None:
        if writer.stream_proc is not None:
            stop_named_stream_process(name, writer.stream_proc)
            writer.stream_proc = None
        writer.stop()

    if saved_out is not None:
        saved_out.release()


def run_camera_dual_stream(state):
    shared_main.DRAW_LANE_VIS = False
    seg_model, wheel_det, trackers = shared_main.load_models()

    cap = open_camera_capture()
    if cap is None or not cap.isOpened():
        print(f"❌ 캡처보드/카메라를 열 수 없습니다: /dev/video{CAM_NUM}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAMERA_WIDTH or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAMERA_HEIGHT or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = CAMERA_FPS or 30.0

    raw_output_size = (w, h)
    processed_output_size = compute_processing_size(w, h)
    raw_output_fps = fps
    processed_output_fps = fps

    print(f"🎥 캡처 입력 시작: /dev/video{CAM_NUM} ({w}×{h} @ {fps:.1f}fps)")
    print(f"   프레임 스킵: {FRAME_SKIP}")
    print(f"   원본 송출 해상도: {raw_output_size[0]}×{raw_output_size[1]}")
    print(
        f"   처리 입력/송출 해상도: {processed_output_size[0]}×{processed_output_size[1]} "
        f"(모델 imgsz 기준 long-side {PROCESS_LONG_SIDE})"
    )
    print("   처리본은 원본 시간축에 맞춰 반복 프레임을 보정")

    raw_writer, raw_saved_out, raw_status = build_output_target(
        "원본",
        RAW_STREAM_ENABLED,
        RAW_STREAM_RTMP_URL,
        SAVE_RAW_OUTPUT,
        RAW_OUTPUT_PATH,
        raw_output_fps,
        raw_output_size,
    )
    processed_writer, processed_saved_out, processed_status = build_output_target(
        "처리본",
        PROCESSED_STREAM_ENABLED,
        PROCESSED_STREAM_RTMP_URL,
        SAVE_PROCESSED_OUTPUT,
        PROCESSED_OUTPUT_PATH,
        processed_output_fps,
        processed_output_size,
    )

    print(f"   원본 송출 상태: {_stream_status_text(raw_status)}")
    print(f"   처리본 송출 상태: {_stream_status_text(processed_status)}\n")

    clahe = cv2.createCLAHE(
        clipLimit=shared_main.CLIP_LIMIT,
        tileGridSize=shared_main.TILE_SIZE,
    )
    capture_reader = CameraCaptureDistributor(cap, raw_writer=raw_writer).start()
    fps_ctr = shared_main.FPSCounter(window=30, print_interval=30)
    last_frame_id = 0
    last_processed_frame_id = 0
    infer_count = 0

    try:
        while True:
            frame_id, frame, reader_fail_cnt = capture_reader.read_latest(
                last_frame_id,
                timeout_sec=CAMERA_READ_TIMEOUT_SEC,
            )
            if frame is None:
                if reader_fail_cnt >= 30:
                    print("⚠️  카메라 입력이 끊겨 재연결을 시도합니다.")
                    capture_reader.stop()
                    cap.release()
                    cap = open_camera_capture(retries=5, retry_delay=2.0)
                    if cap is None or not cap.isOpened():
                        print("❌ 카메라 재연결 실패")
                        break

                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
                    raw_output_size = (w, h)
                    processed_output_size = compute_processing_size(w, h)
                    capture_reader = CameraCaptureDistributor(cap, raw_writer=raw_writer).start()
                    last_frame_id = 0
                    last_processed_frame_id = 0
                    print(
                        "✅ 카메라 재연결 성공 "
                        f"(raw={raw_output_size[0]}x{raw_output_size[1]}, "
                        f"processed={processed_output_size[0]}x{processed_output_size[1]})"
                    )
                    continue
                time.sleep(0.01)
                continue

            last_frame_id = frame_id

            if FRAME_SKIP > 1 and frame_id % FRAME_SKIP != 0:
                continue

            fps_ctr.tick()
            if (frame.shape[1], frame.shape[0]) != processed_output_size:
                proc_frame = cv2.resize(frame, processed_output_size, interpolation=cv2.INTER_AREA)
            else:
                proc_frame = frame

            proc_frame = shared_main._apply_clahe(proc_frame, clahe)
            vis, _ = shared_main.process_frame(
                seg_model, wheel_det, trackers, proc_frame, state
            )

            if (vis.shape[1], vis.shape[0]) != processed_output_size:
                vis = cv2.resize(vis, processed_output_size)

            repeat_count = max(1, frame_id - last_processed_frame_id)
            if processed_writer is not None:
                processed_writer.submit_frame(vis, repeat_count=repeat_count)
            last_processed_frame_id = frame_id
            infer_count += 1

            if fps_ctr.should_print():
                rss_mb = shared_main.get_memory_usage_mb()
                rss_text = f"{rss_mb:.1f}MB" if rss_mb is not None else "n/a"
                raw_queue = raw_writer.pending_frames() if raw_writer else 0
                proc_queue = processed_writer.pending_frames() if processed_writer else 0
                raw_emitted = raw_writer.output_frame_count if raw_writer else 0
                proc_emitted = processed_writer.output_frame_count if processed_writer else 0
                print(
                    f"  [capture] 추론 {fps_ctr.fps():.1f} fps  "
                    f"(processed={infer_count}, repeat={repeat_count}, "
                    f"raw_emit={raw_emitted}, raw_q={raw_queue}, "
                    f"proc_emit={proc_emitted}, proc_q={proc_queue}, "
                    f"rss={rss_text}, source={w}x{h}, "
                    f"proc={processed_output_size[0]}x{processed_output_size[1]})"
                )

            if SHOW_WINDOW:
                cv2.imshow("Capture Dual Stream - Processed", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        capture_reader.stop()
        cap.release()
        close_output_target("원본", raw_writer, raw_saved_out)
        close_output_target("처리본", processed_writer, processed_saved_out)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print(f"\n✅ 종료 | 추론 평균 FPS: {fps_ctr.fps():.1f}")
    print(f"   처리 프레임 수: {infer_count}")
    if raw_writer is not None:
        print(f"   원본 출력 프레임 수: {raw_writer.output_frame_count}")
        print(f"   원본 송출 상태: {_stream_status_text(raw_writer.stream_status)}")
    else:
        print(f"   원본 송출 상태: {_stream_status_text(raw_status)}")
    if processed_writer is not None:
        print(f"   처리본 출력 프레임 수: {processed_writer.output_frame_count}")
        print(f"   처리본 송출 상태: {_stream_status_text(processed_writer.stream_status)}")
    else:
        print(f"   처리본 송출 상태: {_stream_status_text(processed_status)}")


if __name__ == "__main__":
    shared_main.RUNTIME_LOG_PATH = RUNTIME_LOG_PATH
    shared_main.REPOSITION_TOGGLE_DELAY_SEC = 0.0
    shared_main.DRAW_LANE_VIS = False
    shared_main.setup_runtime_logging()

    state = shared_main.RefereeState()
    shared_main.start_input_listener(state)
    run_camera_dual_stream(state)

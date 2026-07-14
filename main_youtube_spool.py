"""
main_youtube_spool.py

유튜브 라이브/VOD 입력을 디스크 세그먼트로 먼저 저장한 뒤,
완료된 세그먼트를 순서대로 추론해서 유튜브 RTMP로 송출한다.

목표:
1) 입력 프레임을 메모리 큐에서 버리지 않는다.
2) 추론이 느리면 지연은 누적되더라도, 저장된 세그먼트를 순서대로 처리한다.
3) 기존 main.py는 건드리지 않고 별도 실행 파일로 유지한다.
"""

import os
import re
import time
import threading
import subprocess
from pathlib import Path

import cv2
import numpy as np

import main as shared_main


# ──────────────────────────────────────────────────────────
#  입력 스트림 설정
# ──────────────────────────────────────────────────────────
INPUT_STREAM_URL = "https://youtube.com/live/--fQWyY7W-k"
INPUT_STREAM_FORMAT = (
    "bestvideo[height<=1080][vcodec*=avc1]/"
    "best[height<=1080][vcodec*=avc1]/"
    "bestvideo[height<=1080]/best[height<=1080]/bestvideo/best"
)

# ──────────────────────────────────────────────────────────
#  스풀(입력 저장) 설정
# ──────────────────────────────────────────────────────────
SPOOL_ROOT_DIR = "spool_sessions"
SESSION_NAME = ""  # 비우면 시작 시각으로 자동 생성
SPOOL_SEGMENT_SECONDS = 10
KEEP_INPUT_SEGMENTS = False
DELETE_INPUT_SEGMENTS_AFTER_PROCESS = True
SPOOL_POLL_INTERVAL_SEC = 0.5

# ──────────────────────────────────────────────────────────
#  처리 결과 출력 설정
# ──────────────────────────────────────────────────────────
OUTPUT_STREAM_ENABLED = True
OUTPUT_STREAM_RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/e6m2-vfja-yz2m-sjab-fsj7" # 성균관대학교노승윤 계정
SAVE_PROCESSED_OUTPUT = False
PROCESSED_OUTPUT_PATH = ""  # 비우면 세션 폴더 안에 자동 생성

STREAM_VIDEO_ENCODER = "h264_nvenc"
CPU_FALLBACK_VIDEO_ENCODER = "libx264"
NVENC_PRESET = "p5"
X264_PRESET = "veryfast"

DRAW_LANE_VIS = True
SHOW_WINDOW = False
RUNTIME_LOG_PATH = "logs/drone_referee_spool_runtime.log"
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


def stop_named_process(name, proc):
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


def make_session_dir():
    session_name = SESSION_NAME.strip()
    if not session_name:
        session_name = time.strftime("%Y%m%d_%H%M%S")

    session_dir = Path(SPOOL_ROOT_DIR) / session_name
    segments_dir = session_dir / "input_segments"
    markers_dir = session_dir / "processed_markers"
    session_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    markers_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, segments_dir, markers_dir


def build_spool_record_command(stream_url, segments_dir: Path):
    segment_pattern = str(segments_dir / "input_%06d.mkv")
    return [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
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


def start_spool_recorder(stream_url, segments_dir: Path):
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
    start_named_ffmpeg_log_thread("spool-input", proc)
    return proc


def segment_index_from_path(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return -1
    return int(match.group(1))


def marker_path_for(segment_path: Path, markers_dir: Path):
    return markers_dir / f"{segment_path.stem}.done"


def list_processable_segments(segments_dir: Path, markers_dir: Path, recorder_alive: bool):
    segment_files = sorted(segments_dir.glob("input_*.mkv"))
    if recorder_alive and len(segment_files) > 1:
        candidates = segment_files[:-1]
    else:
        candidates = segment_files

    ready = []
    for path in candidates:
        marker_path = marker_path_for(path, markers_dir)
        if marker_path.exists():
            continue
        try:
            if path.stat().st_size <= 0:
                continue
        except FileNotFoundError:
            continue
        ready.append(path)
    return ready


def probe_segment(path: Path):
    cap = cv2.VideoCapture(str(path))
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


def open_processed_writer(path, fps, size):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        print(f"⚠️  처리 결과 저장 파일을 열 수 없습니다: {path}")
        writer.release()
        return None

    print(f"💾 처리 결과 저장 → {path}")
    return writer


def start_output_stream_process(width, height, fps):
    if not OUTPUT_STREAM_ENABLED:
        return None, "disabled"

    if not OUTPUT_STREAM_RTMP_URL or "YOUR_" in OUTPUT_STREAM_RTMP_URL:
        print("⚠️  OUTPUT_STREAM_RTMP_URL이 비어 있거나 placeholder 상태입니다.")
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
        return None, "start_failed"
    except Exception as e:
        print(f"⚠️  처리본 라이브 송출 시작 실패: {e}")
        return None, "start_failed"

    print(f"📡 처리본 송출 시작 → {OUTPUT_STREAM_RTMP_URL}")
    print(
        f"   encoder={STREAM_VIDEO_ENCODER}  bitrate={bitrate['b:v']}  "
        f"maxrate={bitrate['maxrate']}  bufsize={bitrate['bufsize']}  cq={bitrate['cq']}"
    )
    print(f"   키프레임 간격: 약 {gop_size / fps:.1f}초 ({gop_size} 프레임)")
    start_named_ffmpeg_log_thread("processed-output", proc)
    return proc, "streaming"


class OutputTarget:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.output_path = PROCESSED_OUTPUT_PATH.strip()
        self.saved_out = None
        self.stream_proc = None
        self.stream_status = "disabled"
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0

    def initialize(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        size = (width, height)

        if SAVE_PROCESSED_OUTPUT:
            out_path = self.output_path
            if not out_path:
                out_path = str(self.session_dir / "processed_output.mp4")
            self.saved_out = open_processed_writer(out_path, fps, size)

        self.stream_proc, self.stream_status = start_output_stream_process(width, height, fps)

        print(f"   처리본 로컬 저장: {'ON' if self.saved_out else 'OFF'}")
        print(f"   처리본 유튜브 송출: {_stream_status_text(self.stream_status)}\n")

    def write(self, frame):
        if self.saved_out is not None:
            self.saved_out.write(frame)

        if self.stream_proc is not None and self.stream_proc.stdin is not None:
            try:
                self.stream_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                print("⚠️  처리본 유튜브 송출 연결이 끊겼습니다. 로컬 저장만 계속 진행합니다.")
                stop_named_process("processed-output", self.stream_proc)
                self.stream_proc = None
                self.stream_status = "disconnected"

        self.frame_count += 1

    def close(self):
        if self.saved_out is not None:
            self.saved_out.release()
            self.saved_out = None

        if self.stream_proc is not None:
            stop_named_process("processed-output", self.stream_proc)
            self.stream_proc = None


def mark_segment_processed(segment_path: Path, markers_dir: Path):
    marker_path = marker_path_for(segment_path, markers_dir)
    marker_path.write_text("done\n", encoding="utf-8")

    if DELETE_INPUT_SEGMENTS_AFTER_PROCESS and not KEEP_INPUT_SEGMENTS:
        try:
            segment_path.unlink()
        except FileNotFoundError:
            pass


def process_segment_file(
    segment_path: Path,
    seg_model,
    wheel_det,
    trackers,
    state,
    output_target: OutputTarget,
    fps_ctr,
):
    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        print(f"⚠️  세그먼트를 열 수 없습니다: {segment_path}")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or output_target.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or output_target.height
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = output_target.fps or 30.0

    if output_target.width == 0 or output_target.height == 0:
        output_target.initialize(width, height, fps)

    clahe = cv2.createCLAHE(
        clipLimit=shared_main.CLIP_LIMIT,
        tileGridSize=shared_main.TILE_SIZE,
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_ctr.tick()
        frame = shared_main._apply_clahe(frame, clahe)
        vis, _ = shared_main.process_frame(seg_model, wheel_det, trackers, frame, state)

        if (vis.shape[1], vis.shape[0]) != (output_target.width, output_target.height):
            vis = cv2.resize(vis, (output_target.width, output_target.height))

        output_target.write(vis)
        frame_count += 1

        if SHOW_WINDOW:
            cv2.imshow("Drone Referee - Spool", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    return frame_count


def run_spool_pipeline(state):
    if KEEP_INPUT_SEGMENTS and DELETE_INPUT_SEGMENTS_AFTER_PROCESS:
        raise ValueError("KEEP_INPUT_SEGMENTS=True 와 DELETE_INPUT_SEGMENTS_AFTER_PROCESS=True는 동시에 사용할 수 없습니다.")

    shared_main.INPUT_STREAM_FORMAT = INPUT_STREAM_FORMAT
    shared_main.DRAW_LANE_VIS = DRAW_LANE_VIS

    session_dir, segments_dir, markers_dir = make_session_dir()
    print(f"📁 세션 폴더: {session_dir}")

    stream_url = shared_main.resolve_stream_url(INPUT_STREAM_URL)
    if not stream_url:
        return

    recorder_proc = start_spool_recorder(stream_url, segments_dir)
    if recorder_proc is None:
        return

    seg_model, wheel_det, trackers = shared_main.load_models()
    output_target = OutputTarget(session_dir)
    fps_ctr = shared_main.FPSCounter(window=30, print_interval=30)
    total_processed_frames = 0
    total_processed_segments = 0
    last_wait_log_at = 0.0

    try:
        while True:
            recorder_alive = recorder_proc.poll() is None
            pending_segments = list_processable_segments(segments_dir, markers_dir, recorder_alive)

            if not pending_segments:
                if not recorder_alive:
                    break

                now = time.time()
                if now - last_wait_log_at >= 10.0:
                    total_segments = len(list(segments_dir.glob("input_*.mkv")))
                    done_segments = len(list(markers_dir.glob("*.done")))
                    print(
                        f"  [spool] 대기 중... "
                        f"(saved={total_segments}, processed={done_segments}, session={session_dir.name})"
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

                print(
                    f"🎞️  세그먼트 처리 시작: {segment_path.name} "
                    f"({meta['width']}×{meta['height']} @ {meta['fps']:.1f}fps, frames={meta['frames']})"
                )

                segment_frames = process_segment_file(
                    segment_path,
                    seg_model,
                    wheel_det,
                    trackers,
                    state,
                    output_target,
                    fps_ctr,
                )
                mark_segment_processed(segment_path, markers_dir)

                total_processed_frames += segment_frames
                total_processed_segments += 1

                rss_mb = shared_main.get_memory_usage_mb()
                rss_text = f"{rss_mb:.1f}MB" if rss_mb is not None else "n/a"
                total_segments = len(list(segments_dir.glob("input_*.mkv")))
                done_segments = len(list(markers_dir.glob("*.done")))
                backlog = max(0, total_segments - done_segments)
                print(
                    f"  [spool] segment={segment_path.name} done  "
                    f"(processed_frames={segment_frames}, total_frames={total_processed_frames}, "
                    f"segments_done={total_processed_segments}, backlog={backlog}, "
                    f"infer={fps_ctr.fps():.1f}fps, out={output_target.frame_count}, rss={rss_text})"
                )

                if SHOW_WINDOW and cv2.getWindowProperty("Drone Referee - Spool", cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[중단] 사용자 요청으로 스풀 처리를 종료합니다.")
    finally:
        if recorder_proc.poll() is None:
            stop_named_process("spool-input", recorder_proc)
        output_target.close()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print(f"\n✅ 스풀 종료 | 처리 세그먼트 수: {total_processed_segments}")
    print(f"   처리 프레임 수: {total_processed_frames}")
    print(f"   추론 평균 FPS: {fps_ctr.fps():.1f}")
    print(f"   처리본 출력 프레임 수: {output_target.frame_count}")
    print(f"   처리본 송출 상태: {_stream_status_text(output_target.stream_status)}")
    print(f"   세션 폴더: {session_dir}")


if __name__ == "__main__":
    shared_main.RUNTIME_LOG_PATH = RUNTIME_LOG_PATH
    shared_main.REPOSITION_TOGGLE_DELAY_SEC = 0.0
    shared_main.DRAW_LANE_VIS = DRAW_LANE_VIS
    shared_main.setup_runtime_logging()

    state = shared_main.RefereeState()
    shared_main.start_input_listener(state)
    run_spool_pipeline(state)

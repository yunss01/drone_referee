"""로컬 영상에서 기존 판정과 가림 보완/이탈 판정을 비교한다. 송출하지 않는다."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from dash_referee import DashConfig, DashReferee, margin, paint_geometry, supports_point
import main_capture_dual_youtube as pipeline


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def load_pose_cache(directory, videos):
    settings = {row["key"]: row["value"] for row in read_rows(directory / "settings.csv")}
    if digest(ROOT / pipeline.POSE_MODEL_PATH) != settings["sha256:POSE_MODEL_PATH"]:
        raise ValueError("바퀴 모델이 기존 검출 캐시와 다릅니다.")
    metadata = {row["video"]: row for row in read_rows(directory / "summary.csv")}
    for video in videos:
        info = metadata[video.name]
        if info["status"] != "completed" or int(info["frame_skip"]) != 1:
            raise ValueError("전체 프레임을 처리한 캐시만 사용할 수 있습니다.")
        if video.stat().st_size != int(info["input_size_bytes"]) or video.stat().st_mtime_ns != int(info["input_mtime_ns"]):
            raise ValueError(f"입력 영상이 캐시 생성 당시와 다릅니다: {video}")
    cache = {}
    for row in read_rows(directory / "wheels.csv"):
        box = tuple(int(row[key]) for key in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
        wheel = dict(bbox=box, keypoint=(int(row["keypoint_x"]), int(row["keypoint_y"])),
                     kp_valid=row["kp_valid"] == "1", conf=float(row["bbox_confidence"]))
        cache.setdefault((row["video"], int(row["frame_index"])), []).append((int(row["wheel_index"]), wheel))
    return {key: [wheel for _, wheel in sorted(value)] for key, value in cache.items()}, metadata


class CsvOutput:
    def __init__(self, path, fields):
        self.file = path.open("w", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.file, fields, extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def video_writer(path, size, fps):
    width, height = size
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def run_video(video, model, detector, pose_cache, cache_metadata, args, config, output, tables):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("영상 FPS가 유효하지 않습니다.")
    size = pipeline.compute_processing_size(int(cap.get(3)), int(cap.get(4)))
    if cache_metadata is not None:
        info = cache_metadata[video.name]
        if size != (int(info["processing_width"]), int(info["processing_height"])):
            raise ValueError("처리 크기가 캐시와 다릅니다.")
    engine = DashReferee(config)
    clahe = cv2.createCLAHE(clipLimit=pipeline.CLAHE_CLIP_LIMIT, tileGridSize=pipeline.CLAHE_TILE_SIZE)
    old_starts, last_positive = {}, {}
    counts = dict(video=video.name, frames=0, wheels=0, baseline_raw=0,
                  observed_components_raw=0, temporal_raw=0, baseline_confirmed=0,
                  temporal_confirmed=0, added_vs_baseline=0, removed_vs_baseline=0,
                  unknown=0, motion_valid_frames=0, entered=0, exited=0, lost=0,
                  reacquired=0, contact_started=0, contact_ended=0,
                  intrusion=0, departure=0, departure_active_wheels=0,
                  segmentation_elapsed_sec=0.0, temporal_elapsed_sec=0.0,
                  status="error")
    writer = video_writer(output / f"{video.stem}_comparison.mp4", (size[0] * 2, size[1]), fps / args.frame_skip) if args.save_video else None
    started = last_report = time.monotonic()
    frame_index = -1
    try:
        while True:
            ok, original = cap.read()
            if not ok:
                if frame_index + 1 != total:
                    raise RuntimeError(f"영상 조기 종료: {frame_index + 1}/{total}")
                counts["status"] = "completed"
                break
            frame_index += 1
            if frame_index % args.frame_skip:
                continue
            if args.max_frames is not None and counts["frames"] >= args.max_frames:
                counts["status"] = "limited"
                break
            frame = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
            processed = pipeline._apply_clahe(frame, clahe)
            infer_started = time.perf_counter()
            lanes = pipeline.build_lane_data(model, processed)
            counts["segmentation_elapsed_sec"] += time.perf_counter() - infer_started
            wheels = (pose_cache.get((video.name, frame_index), []) if pose_cache is not None
                      else detector.predict(processed))
            timestamp = frame_index / fps
            temporal_started = time.perf_counter()
            result = engine.update(processed, lanes, wheels[:pipeline.MAX_WHEELS], timestamp)
            counts["temporal_elapsed_sec"] += time.perf_counter() - temporal_started
            counts["frames"] += 1
            counts["motion_valid_frames"] += int(result["motion_valid"])
            current_labels = engine.paint.current_observed_labels
            observed = np.uint8(current_labels > 0) * 255
            current_ids = np.unique(current_labels[current_labels > 0])
            current = paint_geometry(observed, current_labels, current_ids, config.max_half_width)
            old_states = []
            common = dict(video=video.name, frame_index=frame_index, timestamp_sec=timestamp)
            for key in set(old_starts) - {wheel["track_id"] for wheel in result["wheels"]}:
                old_starts.pop(key, None)
            for wheel, state in zip(result["wheels"], result["states"]):
                key = wheel["track_id"]
                pipeline.lane_checker_module.compute_distances(lanes, *wheel["keypoint"])
                baseline = any(v["violated"] for v in pipeline.lane_checker_module.check_violation(lanes))
                if baseline:
                    old_starts.setdefault(key, timestamp)
                    last_positive[key] = timestamp
                else:
                    old_starts.pop(key, None)
                old_confirmed = baseline and timestamp - old_starts[key] + 1e-9 >= config.entry_hold_sec
                observed_raw = any(margin(wheel["keypoint"], lane) < 0 and supports_point(
                    wheel["keypoint"], lane, config.support_tolerance_px) for lane in current)
                assessable = state.get("assessable", state["evidence"] != "unknown")
                temporal = None if not assessable and not state["violated"] else state["violated"]
                old_states.append(dict(wheel_id=key, violated=baseline, confirmed=old_confirmed,
                                       duration=timestamp - old_starts[key] if old_confirmed else 0))
                row = dict(**common, wheel_id=key, detection_index=wheel["detection_index"],
                           keypoint_x=wheel["keypoint"][0], keypoint_y=wheel["keypoint"][1],
                           baseline_raw=int(baseline), baseline_confirmed=int(old_confirmed),
                           observed_components_raw=int(observed_raw), temporal_raw="" if temporal is None else int(temporal),
                           temporal_confirmed=int(state["confirmed"]), naive_hold_raw=int(timestamp - last_positive.get(key, -np.inf) <= 0.2),
                           phase=state.get("phase", ""), evidence=state["evidence"], lane_id=state["lane_id"], hit_lane_id=state["hit_lane_id"],
                           intrusion_confirmed=int(state.get("intrusion_confirmed", state["confirmed"])),
                           departure_detected=int(state.get("departure_detected", False)),
                           departure_active=int(state.get("departure_active", False)),
                           departure_recorded=int(state.get("departure_recorded", False)),
                           departure_latched=int(state.get("departure_latched", False)),
                           departure_anchor_id=state.get("departure_anchor_id"),
                           contact_source=state["contact_source"], motion_valid=int(result["motion_valid"]),
                           motion_reason=result["motion_reason"], motion_error_px=result["motion_error_px"])
                tables["wheels"].write(row)
                counts["wheels"] += 1
                for field in ["baseline_raw", "baseline_confirmed", "observed_components_raw", "temporal_confirmed"]:
                    counts[field] += int(row[field])
                counts["temporal_raw"] += int(temporal is True)
                counts["unknown"] += int(temporal is None)
                counts["departure_active_wheels"] += row["departure_active"]
                counts["added_vs_baseline"] += int(temporal is True and not baseline)
                counts["removed_vs_baseline"] += int(temporal is False and baseline)
            for event in result["events"]:
                if event.get("event") == "departure" and "paint_mask" in event:
                    evidence_dir = output / "evidence"
                    evidence_dir.mkdir(exist_ok=True)
                    evidence_file = evidence_dir / f"{video.stem}_{frame_index}_W{event['wheel_id']}_A{event['paint_anchor_id']}.png"
                    cv2.imwrite(str(evidence_file), event["paint_mask"])
                    event["evidence_mask"] = str(evidence_file.relative_to(output))
                tables["events"].write({**common, **event})
                counts[event["event"]] += 1
            for item in result.get("departure_diagnostics", []):
                tables["departure_trace"].write({**common, **item})
            tables["frames"].write(dict(**common, detected_wheels=len(result["wheels"]),
                                        baseline_raw=int(any(s["violated"] for s in old_states)),
                                        temporal_raw=int(any(s["violated"] for s in result["states"])),
                                        temporal_confirmed=int(any(s["confirmed"] for s in result["states"])),
                                        unknown_wheels=sum(not s.get("assessable", s["evidence"] != "unknown") for s in result["states"]),
                                        departure_detected=int(any(s.get("departure_detected") for s in result["states"])),
                                        departure_active=int(any(s.get("departure_active") for s in result["states"])),
                                        departure_latched=int(any(s.get("departure_latched") for s in result["states"])),
                                        motion_valid=int(result["motion_valid"]), motion_reason=result["motion_reason"],
                                        recovered_pixels=result["recovered_pixels"]))
            if writer is not None:
                before = pipeline.draw_frame(frame, lanes, result["wheels"], old_states, False)
                after = pipeline.draw_frame(frame, result["lanes"], result["wheels"], result["states"], False)
                for image, text in [(before, "FRAME ONLY (same wheel IDs / video timer)"),
                                    (after, "PAINT MEMORY + LINE OUT (finite paint / screen-down side)")]:
                    cv2.rectangle(image, (0, size[1] - 30), (size[0], size[1]), (20, 20, 20), -1)
                    cv2.putText(image, text, (10, size[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                writer.stdin.write(np.hstack([before, after]).tobytes())
            if time.monotonic() - last_report >= 10:
                print(f"{video.name} {frame_index + 1}/{total} "
                      f"{counts['frames'] / (time.monotonic() - started):.1f}fps "
                      f"added={counts['added_vs_baseline']} unknown={counts['unknown']}", flush=True)
                last_report = time.monotonic()
    finally:
        cap.release()
        if writer is not None:
            writer.stdin.close()
            if writer.wait(timeout=60) != 0:
                raise RuntimeError("비교 영상 인코딩 실패")
        counts["elapsed_sec"] = round(time.monotonic() - started, 3)
        tables["summary"].write(counts)
    print(json.dumps(counts, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "dataset/편집본")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "특허용 자료들/dash_tracking_results")
    parser.add_argument("--pose-cache", type=Path, help="기존 clipping 실행 폴더: 같은 바퀴 검출을 재사용")
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=0.2)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.frame_skip < 1 or args.hold_seconds < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("프레임 스킵/시간/제한 값이 유효하지 않습니다.")
    videos = args.videos or sorted(args.input_dir.glob("*.mp4"), key=lambda p: (len(p.stem), p.stem))
    videos = [p.resolve() for p in videos]
    if not videos:
        parser.error("입력 영상이 없습니다.")
    cv2.setNumThreads(2)
    pipeline.apply_module_overrides()
    from ultralytics import YOLO
    model = YOLO(str(ROOT / pipeline.SEG_MODEL_PATH), task="segment")
    pose_cache = cache_metadata = None
    detector = None
    if args.pose_cache:
        pose_cache, cache_metadata = load_pose_cache(args.pose_cache, videos)
    else:
        detector = pipeline.WheelDetector(str(ROOT / pipeline.POSE_MODEL_PATH))
    output = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    config = DashConfig(entry_hold_sec=args.hold_seconds)
    metadata = dict(config=asdict(config), frame_skip=args.frame_skip,
                    show_restored_paint=pipeline.SHOW_RESTORED_PAINT,
                    independent_ground_truth=False, comparison="운영 기준과의 차이이며 실제 FN/FP 수가 아님",
                    pose_cache=str(args.pose_cache) if args.pose_cache else None,
                    videos={str(p):dict(bytes=p.stat().st_size, sha256=digest(p)) for p in videos},
                    departure_config=asdict(DashReferee(config).departure.config),
                    model_sha256={name:digest(ROOT / name) for name in
                                  [pipeline.SEG_MODEL_PATH, pipeline.POSE_MODEL_PATH]},
                    departure_definition="화면 아래쪽만 이탈. 바퀴 주변 실제 도색의 유한 선분 내부이고 접지점의 x열에 도색 픽셀이 있을 때, 도색 아래 경계+오차 여유를 넘어간 현재 접지점을 이탈로 판정. 과거 기록은 현재 상태와 분리.",
                    source_sha256={name:digest(ROOT / name) for name in ["dash_referee.py", "departure_checker.py", "main.py", "main_capture_dual_youtube.py", "lane_checker.py", "detector.py", "preprocess.py", "tools/evaluate_dash_referee.py"]})
    if args.pose_cache:
        metadata["pose_cache_sha256"] = {name:digest(args.pose_cache / name) for name in
                                          ["settings.csv", "summary.csv", "wheels.csv"]}
    for name in metadata["source_sha256"]:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (output / "settings.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    fields = dict(
        wheels="video frame_index timestamp_sec wheel_id detection_index keypoint_x keypoint_y baseline_raw baseline_confirmed observed_components_raw temporal_raw temporal_confirmed intrusion_confirmed departure_detected departure_active departure_recorded departure_latched departure_anchor_id naive_hold_raw phase evidence lane_id hit_lane_id contact_source motion_valid motion_reason motion_error_px",
        events="video frame_index timestamp_sec event time_sec wheel_id lane_id contact_started_sec paint_anchor_id keypoint_x keypoint_y paint_column_x paint_top_y paint_bottom_y clearance_px required_margin_px entry_margin_px edge_error_px locally_verified verified_columns decision_rule along_px length_px axis_ax axis_ay axis_bx axis_by evidence bbox_x1 bbox_y1 bbox_x2 bbox_y2 paint_origin_x paint_origin_y model_age_sec model_error_px evidence_mask",
        frames="video frame_index timestamp_sec detected_wheels baseline_raw temporal_raw temporal_confirmed departure_detected departure_active departure_latched unknown_wheels motion_valid motion_reason recovered_pixels",
        departure_trace="video frame_index timestamp_sec wheel_id paint_anchor_id lane_id finite_column along_px length_px near_wheel permitted paint_column_x paint_top_y paint_bottom_y clearance_px required_margin_px entry_margin_px edge_error_px locally_verified verified_columns decision_rule is_out selected model_age_sec model_error_px",
        summary="video frames wheels baseline_raw observed_components_raw temporal_raw baseline_confirmed temporal_confirmed added_vs_baseline removed_vs_baseline unknown motion_valid_frames intrusion departure departure_active_wheels entered exited lost reacquired contact_started contact_ended segmentation_elapsed_sec temporal_elapsed_sec status elapsed_sec")
    tables = {name:CsvOutput(output / f"{name}.csv", names.split()) for name, names in fields.items()}
    print("결과 폴더:", output, flush=True)
    try:
        for video in videos:
            run_video(video, model, detector, pose_cache, cache_metadata, args, config, output, tables)
    finally:
        for table in tables.values():
            table.close()


if __name__ == "__main__":
    main()

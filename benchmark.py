"""
benchmark.py — 각 처리 단계별 시간 측정 및 모델 비교

원본 PT 모델 vs TensorRT FP16 엔진의 단계별 처리 시간과 FPS를 비교한다.

사용법:
    python3 benchmark.py --img test_image.png --runs 30

출력 예시:
    ┌─────────────────────────────────────────┐
    │         Benchmark Results               │
    │  PT model vs TensorRT FP16             │
    ├──────────────────┬──────────┬──────────┤
    │ Stage            │ PT (ms)  │ TRT (ms) │
    ├──────────────────┼──────────┼──────────┤
    │ CLAHE            │    1.2   │    1.2   │
    │ Seg inference    │  210.3   │   90.1   │
    │ Mask processing  │    8.4   │    8.4   │
    │ PCA centerline   │    2.1   │    2.1   │
    │ Pose inference   │   95.2   │   47.3   │
    │ Violation check  │    0.3   │    0.3   │
    │ Draw             │   12.1   │   12.1   │
    ├──────────────────┼──────────┼──────────┤
    │ Total            │  329.6   │  161.5   │
    │ FPS              │    3.0   │    6.2   │
    └──────────────────┴──────────┴──────────┘
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

from preprocess import _apply_clahe, CLIP_LIMIT, TILE_SIZE
from detector import WheelDetector
from lane_checker import (
    run_segmentation, filter_small_contours,
    get_instance_half_width, get_centerline,
    compute_distances, check_violation,
)

# ──────────────────────────────────────────────────────────
#  비교할 모델 경로 설정
# ──────────────────────────────────────────────────────────
PT_SEG_PATH   = "model/best_seg_rev03.pt"        # 원본 PT
PT_POSE_PATH  = "model/best_referee.pt"           # 원본 PT

TRT_SEG_PATH  = "model/best_seg_rev03_FP16.engine"   # TensorRT FP16
TRT_POSE_PATH = "model/best_referee_FP16.engine"      # TensorRT FP16
# ──────────────────────────────────────────────────────────

WARMUP_RUNS = 3   # 워밍업 (첫 몇 번은 느리므로 제외)


def measure_stages(seg_model, wheel_det, frame, clahe, runs=30):
    """각 단계별 처리 시간을 측정하여 평균값(ms) 반환"""

    times = {
        "clahe":     [],
        "seg":       [],
        "mask":      [],
        "pca":       [],
        "pose":      [],
        "violation": [],
        "draw":      [],
    }

    # 워밍업
    for _ in range(WARMUP_RUNS):
        f = _apply_clahe(frame, clahe)
        run_segmentation(seg_model, f)
        wheel_det.predict(f)

    for _ in range(runs):
        # 1. CLAHE
        t0 = time.perf_counter()
        f = _apply_clahe(frame, clahe)
        times["clahe"].append((time.perf_counter() - t0) * 1000)

        # 2. Seg 추론
        t0 = time.perf_counter()
        instance_masks = run_segmentation(seg_model, f)
        times["seg"].append((time.perf_counter() - t0) * 1000)

        # 3. 마스크 후처리 (erosion + contour filter)
        t0 = time.perf_counter()
        lane_data = []
        for mask_bool in instance_masks:
            mask_bin = (mask_bool * 255).astype(np.uint8)
            contours = filter_small_contours(mask_bin)
            if not contours:
                continue
            half_width = get_instance_half_width(contours)
            lane_data.append({
                "mask_bin":   mask_bin,
                "contours":   contours,
                "half_width": half_width,
                "centerline": None,
            })
        times["mask"].append((time.perf_counter() - t0) * 1000)

        # 4. PCA 중심선 추출
        t0 = time.perf_counter()
        valid_lanes = []
        for data in lane_data:
            cl = get_centerline(data["mask_bin"])
            if cl is None:
                continue
            data["centerline"] = cl
            valid_lanes.append(data)
        times["pca"].append((time.perf_counter() - t0) * 1000)

        # 5. Pose 추론 (바퀴 접지점)
        t0 = time.perf_counter()
        wheels = wheel_det.predict(f)
        times["pose"].append((time.perf_counter() - t0) * 1000)

        # 6. 침범 판정
        t0 = time.perf_counter()
        for w in wheels:
            kx, ky = w["keypoint"]
            compute_distances(valid_lanes, kx, ky)
            check_violation(valid_lanes)
        times["violation"].append((time.perf_counter() - t0) * 1000)

        # 7. 시각화 (draw)
        t0 = time.perf_counter()
        vis = f.copy()
        for data in valid_lanes:
            overlay = vis.copy()
            overlay[data["mask_bin"] > 0] = (0, 200, 255)
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        for w in wheels:
            x1, y1, x2, y2 = w["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, w["keypoint"], 5, (0, 0, 255), -1)
        times["draw"].append((time.perf_counter() - t0) * 1000)

    return {
        k: {"mean": np.mean(v), "std": np.std(v)}
        for k, v in times.items()
    }


def print_table(pt_times, trt_times):
    labels = {
        "clahe":     "CLAHE preprocessing",
        "seg":       "Seg inference",
        "mask":      "Mask post-processing",
        "pca":       "PCA centerline",
        "pose":      "Pose inference",
        "violation": "Violation check",
        "draw":      "Visualization",
    }

    pt_total  = sum(v["mean"] for v in pt_times.values())
    trt_total = sum(v["mean"] for v in trt_times.values())
    pt_fps    = 1000 / pt_total
    trt_fps   = 1000 / trt_total

    col1 = 22   # Stage
    col2 = 16   # PT mean ± std
    col3 = 16   # TRT mean ± std
    col4 = 8    # Speedup

    header_w = col1 + col2 + col3 + col4 + 9
    sep  = "─" * header_w
    line = f"{'─'*col1}┼{'─'*col2}┼{'─'*col3}┼{'─'*col4}"

    print(f"\n┌{sep}┐")
    print(f"│{'Benchmark: PT model vs TensorRT FP16':^{header_w}}│")
    print(f"├{'─'*col1}┬{'─'*col2}┬{'─'*col3}┬{'─'*col4}┤")
    print(f"│{'Stage':<{col1}}│{'PT mean±std (ms)':^{col2}}│{'TRT mean±std (ms)':^{col3}}│{'Speedup':^{col4}}│")
    print(f"├{line}┤")

    for key, label in labels.items():
        pt_m  = pt_times[key]["mean"]
        pt_s  = pt_times[key]["std"]
        trt_m = trt_times[key]["mean"]
        trt_s = trt_times[key]["std"]
        speedup = pt_m / trt_m if trt_m > 0 else 1.0
        sp_str = f"{speedup:.2f}x" if speedup > 1.05 else "  -"
        pt_str  = f"{pt_m:.1f} ± {pt_s:.1f}"
        trt_str = f"{trt_m:.1f} ± {trt_s:.1f}"
        print(f"│{label:<{col1}}│{pt_str:^{col2}}│{trt_str:^{col3}}│{sp_str:^{col4}}│")

    print(f"├{line}┤")

    pt_total_str  = f"{pt_total:.1f}"
    trt_total_str = f"{trt_total:.1f}"
    pt_fps_str    = f"{pt_fps:.1f} fps"
    trt_fps_str   = f"{trt_fps:.1f} fps"
    speedup_str   = f"{pt_total/trt_total:.2f}x"

    print(f"│{'Total (mean)':<{col1}}│{pt_total_str:^{col2}}│{trt_total_str:^{col3}}│{speedup_str:^{col4}}│")
    print(f"│{'FPS':<{col1}}│{pt_fps_str:^{col2}}│{trt_fps_str:^{col3}}│{'':^{col4}}│")
    print(f"└{'─'*col1}┴{'─'*col2}┴{'─'*col3}┴{'─'*col4}┘")

    print(f"\n  PT  모델: {pt_total:.1f}ms/frame → {pt_fps:.1f} fps")
    print(f"  TRT 모델: {trt_total:.1f}ms/frame → {trt_fps:.1f} fps")
    print(f"  전체 속도 향상: {pt_total/trt_total:.2f}배\n")


def main():
    parser = argparse.ArgumentParser(description="단계별 처리 시간 벤치마크")
    parser.add_argument("--img",  required=True, help="테스트 이미지 경로")
    parser.add_argument("--runs", type=int, default=30, help="반복 측정 횟수")
    args = parser.parse_args()

    frame = cv2.imread(args.img)
    if frame is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {args.img}")
        return

    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)

    print(f"📂 이미지: {args.img}  ({frame.shape[1]}×{frame.shape[0]})")
    print(f"🔁 반복 횟수: {args.runs} (워밍업 {WARMUP_RUNS}회 제외)")
    print()

    # ── PT 모델 측정 ───────────────────────────────────────
    print("🔍 [1/2] PT 모델 로딩 중...")
    pt_seg  = YOLO(PT_SEG_PATH,  task='segment')
    pt_pose = WheelDetector(model_path=PT_POSE_PATH)
    print("  측정 중...")
    pt_times = measure_stages(pt_seg, pt_pose, frame, clahe, args.runs)
    print("  ✅ PT 측정 완료")
    del pt_seg, pt_pose

    print()

    # ── TensorRT FP16 모델 측정 ────────────────────────────
    print("🔍 [2/2] TensorRT FP16 모델 로딩 중...")
    trt_seg  = YOLO(TRT_SEG_PATH,  task='segment')
    trt_pose = WheelDetector(model_path=TRT_POSE_PATH)
    print("  측정 중...")
    trt_times = measure_stages(trt_seg, trt_pose, frame, clahe, args.runs)
    print("  ✅ TRT 측정 완료")

    # ── 결과 출력 ──────────────────────────────────────────
    print_table(pt_times, trt_times)


if __name__ == "__main__":
    main()

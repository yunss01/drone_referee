"""
visualize_pca.py — PCA 중심선 추출 과정 시각화
점선 마스크 하나를 골라서 픽셀 분포 → 장축 방향 → 중심선을 단계별로 보여줌

사용법:
    python3 visualize_pca.py --img test_image.png --model model/best_seg_rev03.engine --save pca_vis.jpg
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO


CONF_THRESH = 0.5
IMGSZ       = 1280
MIN_AREA    = 200
EXCLUDE_CLASSES = ['crosswalk']


def run_segmentation(model, frame):
    results = model.predict(
        frame, conf=CONF_THRESH, device=0,
        verbose=False, imgsz=IMGSZ, retina_masks=True,
    )
    res = results[0]
    if res.masks is None:
        return []

    masks = []
    for i, m in enumerate(res.masks.data):
        cls_name = res.names[int(res.boxes.cls[i].item())]
        if cls_name in EXCLUDE_CLASSES:
            continue
        mask_np = m.cpu().numpy()
        mask_resized = cv2.resize(
            mask_np.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_resized = cv2.erode(mask_resized.astype(np.uint8), kernel).astype(bool)
        masks.append(mask_resized)
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",    required=True)
    parser.add_argument("--model",  required=True)
    parser.add_argument("--save",   default="pca_vis.jpg")
    parser.add_argument("--idx",    type=int, default=0, help="시각화할 마스크 인덱스")
    args = parser.parse_args()

    model = YOLO(args.model, task='segment')
    frame = cv2.imread(args.img)
    if frame is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {args.img}")
        return

    masks = run_segmentation(model, frame)

    # 면적 기준 필터 + 유효한 마스크만 추출
    valid_masks = []
    for mask_bool in masks:
        mask_bin = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
        if contours:
            valid_masks.append(mask_bin)

    if not valid_masks:
        print("❌ 유효한 마스크가 없습니다.")
        return

    # 가장 큰 마스크 선택 (또는 --idx로 지정)
    idx = min(args.idx, len(valid_masks) - 1)
    mask_bin = valid_masks[idx]
    print(f"✅ 마스크 {idx} 선택 (전체 {len(valid_masks)}개)")

    # 마스크 bounding box로 crop
    ys, xs = np.where(mask_bin > 0)
    pad = 30
    y1c = max(0, ys.min() - pad)
    y2c = min(frame.shape[0], ys.max() + pad)
    x1c = max(0, xs.min() - pad)
    x2c = min(frame.shape[1], xs.max() + pad)

    crop_frame = frame[y1c:y2c, x1c:x2c].copy()
    crop_mask  = mask_bin[y1c:y2c, x1c:x2c]

    # 확대 (논문용으로 크게)
    scale = 3
    H, W  = crop_frame.shape[:2]
    crop_frame = cv2.resize(crop_frame, (W*scale, H*scale), interpolation=cv2.INTER_LINEAR)
    crop_mask  = cv2.resize(crop_mask,  (W*scale, H*scale), interpolation=cv2.INTER_NEAREST)

    # PCA 계산 (확대된 좌표 기준)
    ys2, xs2 = np.where(crop_mask > 0)
    pts = np.column_stack([xs2, ys2]).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    center = mean[0]
    axis   = eigenvectors[0]  # 장축
    perp   = eigenvectors[1]  # 단축

    projections = (pts - center) @ axis
    pt1 = (center + axis * projections.min()).astype(int)
    pt2 = (center + axis * projections.max()).astype(int)

    # ── 3단계 이미지 생성 ──────────────────────────────────

    # 1단계: 원본 crop + 마스크 오버레이
    img1 = crop_frame.copy()
    overlay = img1.copy()
    overlay[crop_mask > 0] = (0, 215, 255)
    img1 = cv2.addWeighted(img1, 0.5, overlay, 0.5, 0)
    cv2.putText(img1, "1. Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 2단계: 픽셀 분포 + 장축/단축 방향 화살표
    img2 = crop_frame.copy()
    overlay2 = img2.copy()
    overlay2[crop_mask > 0] = (0, 215, 255)
    img2 = cv2.addWeighted(img2, 0.5, overlay2, 0.5, 0)

    # 장축 화살표
    arrow_len = int(projections.max() - projections.min()) // 2
    ax_end = (center + axis * arrow_len).astype(int)
    ax_start = (center - axis * arrow_len).astype(int)
    cv2.arrowedLine(img2, tuple(ax_start), tuple(ax_end),
                    (0, 0, 255), 2, tipLength=0.1)

    # 단축 화살표
    perp_proj = (pts - center) @ perp
    perp_len = int((perp_proj.max() - perp_proj.min()) // 2)
    px_end   = (center + perp * perp_len).astype(int)
    px_start = (center - perp * perp_len).astype(int)
    cv2.arrowedLine(img2, tuple(px_start), tuple(px_end),
                    (255, 0, 0), 2, tipLength=0.15)

    # 중심점
    cv2.circle(img2, tuple(center.astype(int)), 5, (255, 255, 0), -1)

    cv2.putText(img2, "2. PCA Axes", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img2, "Major axis", tuple(ax_end + np.array([5, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img2, "Minor axis", tuple(px_end + np.array([5, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 3단계: 중심선 결과
    img3 = crop_frame.copy()
    overlay3 = img3.copy()
    overlay3[crop_mask > 0] = (0, 215, 255)
    img3 = cv2.addWeighted(img3, 0.5, overlay3, 0.5, 0)
    cv2.line(img3, tuple(pt1), tuple(pt2), (0, 0, 255), 3)
    cv2.circle(img3, tuple(pt1), 6, (0, 0, 255), -1)
    cv2.circle(img3, tuple(pt2), 6, (0, 0, 255), -1)
    cv2.putText(img3, "3. Centerline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── 3단계 가로로 이어붙이기 ───────────────────────────
    # 높이 맞추기
    h_max = max(img1.shape[0], img2.shape[0], img3.shape[0])

    def pad_h(img, h):
        if img.shape[0] < h:
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    img1 = pad_h(img1, h_max)
    img2 = pad_h(img2, h_max)
    img3 = pad_h(img3, h_max)

    # 구분선 추가
    divider = np.zeros((h_max, 4, 3), dtype=np.uint8)
    result  = np.hstack([img1, divider, img2, divider, img3])

    cv2.imwrite(args.save, result)
    print(f"💾 저장: {args.save}")
    print(f"   크기: {result.shape[1]}×{result.shape[0]}")


if __name__ == "__main__":
    main()

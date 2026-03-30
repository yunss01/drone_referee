import cv2
import os
import numpy as np
from detector import WheelDetector
 
# ──────────────────────────────────────────
IMAGE_DIR  = "test/"
SAVE_DIR   = "test_results/"
ROI_DIR    = "test_results/roi_crops/"
SHOW_IMAGE = False
SAVE_IMAGE = True
 
ROI_RADIUS = 4      # 접지점 기준 ±4px → 8x8 영역
 
# 흰색 HSV 기준
# 그림자 등으로 차선이 어둡게 찍혀도 감지하도록 V 하한을 낮춤
WHITE_V_MIN = 140   # 기존 180 → 140 (그림자 낀 차선까지 포함)
WHITE_S_MAX = 50    # 채도 상한 (회색~흰색 계열만 허용)
# ──────────────────────────────────────────
 
 
def check_white(roi: np.ndarray):
    if roi.size == 0:
        return 0.0, 0, 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv,
                             (0, 0, WHITE_V_MIN),
                             (180, WHITE_S_MAX, 255))
    white_count = int(white_mask.sum() / 255)
    total_count = roi.shape[0] * roi.shape[1]
    white_ratio = white_count / total_count if total_count > 0 else 0.0
    return white_ratio, white_count, total_count
 
 
def process(original_frame, wheels, frame_name):
    r = ROI_RADIUS
    roi_infos = []
    draw_frame = original_frame.copy()
    h, w = original_frame.shape[:2]
 
    for i, wheel in enumerate(wheels):
        x1, y1, x2, y2 = wheel["bbox"]
        kx, ky          = wheel["keypoint"]
        kp_valid        = wheel["kp_valid"]
 
        # ── ROI: 접지점 ±4px ────────────────────────────
        roi_x1 = max(0, kx - r)
        roi_x2 = min(w, kx + r)
        roi_y1 = max(0, ky)
        roi_y2 = min(h, ky + r*2)
 
        # ── 원본에서 크롭 먼저 ──────────────────────────
        roi_crop = original_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
 
        # ── 흰색 비율 계산 ──────────────────────────────
        white_ratio, white_count, total = check_white(roi_crop)
 
        roi_infos.append({
            "wheel_idx":   i,
            "keypoint":    (kx, ky),
            "kp_valid":    kp_valid,
            "white_ratio": white_ratio,
            "white_count": white_count,
            "total":       total,
            "roi_crop":    roi_crop,
        })
 
        # ── 시각화 ──────────────────────────────────────
        color = (0, 255, 0) if kp_valid else (0, 215, 255)
        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(draw_frame, (kx, ky), 4, (0, 0, 255), -1)
        cv2.rectangle(draw_frame, (roi_x1, roi_y1), (roi_x2, roi_y2),
                      (0, 165, 255), 1)
 
        label = f"conf:{wheel['conf']:.2f} {'KP' if kp_valid else 'FB'}"
        cv2.putText(draw_frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(draw_frame, f"W:{white_ratio:.0%}",
                    (roi_x1, roi_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
 
    cv2.putText(draw_frame, f"wheels: {len(wheels)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
 
    return draw_frame, roi_infos
 
 
def save_roi_crops(roi_infos, frame_name):
    for info in roi_infos:
        crop = info["roi_crop"]
        if crop.size == 0:
            continue
        enlarged = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_NEAREST)
        save_name = (f"{frame_name}_wheel{info['wheel_idx']}"
                     f"_W{info['white_ratio']:.0%}.jpg")
        cv2.imwrite(os.path.join(ROI_DIR, save_name), enlarged)
 
 
def main():
    img_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not img_files:
        print(f"⚠️  이미지 없음: {IMAGE_DIR}")
        return
 
    if SAVE_IMAGE:
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(ROI_DIR, exist_ok=True)
 
    detector = WheelDetector()
    print(f"📂 총 {len(img_files)}장 | ROI=±{ROI_RADIUS}px | V_MIN={WHITE_V_MIN}")
 
    for idx, fname in enumerate(img_files):
        original = cv2.imread(os.path.join(IMAGE_DIR, fname))
        if original is None:
            print(f"  스킵: {fname}")
            continue
 
        wheels = detector.predict(original)
        frame_name = os.path.splitext(fname)[0]
        result_frame, roi_infos = process(original, wheels, frame_name)
 
        print(f"\n[{idx+1}/{len(img_files)}] {fname} → 바퀴 {len(wheels)}개")
        for info in roi_infos:
            kx, ky = info["keypoint"]
            print(f"  wheel{info['wheel_idx']} | 접지점:({kx},{ky}) "
                  f"| 흰색:{info['white_ratio']:.1%} "
                  f"({info['white_count']}/{info['total']}px) "
                  f"| {'KP' if info['kp_valid'] else 'FB'}")
 
        if SAVE_IMAGE:
            cv2.imwrite(
                os.path.join(SAVE_DIR, f"result_{idx:05d}_{fname}"),
                result_frame)
            save_roi_crops(roi_infos, frame_name)
 
        if SHOW_IMAGE:
            cv2.imshow("Test Result", result_frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                print("🛑 중단")
                break
 
    cv2.destroyAllWindows()
    print(f"\n✅ 완료 → {SAVE_DIR}")
 
 
if __name__ == '__main__':
    main()
 

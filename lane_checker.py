"""
lane_checker.py  —  Step 1~5
차선 segmentation instance별 중심선 추출 + 접지점 거리 계산

사용법:
    # 기본 (접지점 하드코딩 테스트)
    python3 lane_checker.py --img test.jpg --model best.pt --save result.jpg

    # 접지점 직접 지정 (픽셀 좌표)
    python3 lane_checker.py --img test.jpg --model best.pt --kx 450 --ky 480 --save result.jpg
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────
#  설정값
# ──────────────────────────────────────────────────────────
CONF_THRESH    = 0.5    # seg 모델 신뢰도 임계값
IMGSZ          = 1280   # 추론 해상도 1280->960으로 다운

MIN_AREA             = 50    # 이보다 작은 마스크 조각은 노이즈로 제거 (px²)
MAX_HALF_WIDTH       = 50    # 반폭 상한 클리핑 (px)
CENTERLINE_MIN_ASPECT = 0.0  # 0 이하이면 span/width 비율 필터 비활성화

# ① 결과에서 제외할 클래스명 (모델의 class 이름과 정확히 일치해야 함)
#    모델의 클래스 목록을 모를 때: 아래 PRINT_CLASSES = True 로 설정하면
#    추론 시 검출된 클래스명을 터미널에 출력해줌
EXCLUDE_CLASSES = ['crosswalk']
PRINT_CLASSES   = False   # True로 바꾸면 클래스명 확인 가능

# 테스트용 기본 접지점 (--kx --ky 로 덮어씌울 수 있음)
DEFAULT_KX = 450
DEFAULT_KY = 480
# ──────────────────────────────────────────────────────────

COLORS = [
    (0,   200, 255),
    (0,   255, 100),
    (255, 100,   0),
    (200,   0, 255),
    (0,   180, 180),
]

def get_color(idx):
    return COLORS[idx % len(COLORS)]


# ── Step 1: seg 추론 ─────────────────────────────────────
def run_segmentation(model, frame):
    """
    EXCLUDE_CLASSES에 포함된 클래스(예: crosswalk)는 마스크에서 제외.
    반환: list of np.ndarray (bool, H×W)
    """
    results = model.predict(
        frame,
        conf=CONF_THRESH,
        device=0,
        verbose=False,
        imgsz=IMGSZ,
        retina_masks=True,
    )
    res = results[0]
    if res.masks is None:
        return []

    if PRINT_CLASSES:
        detected = [res.names[int(c)] for c in res.boxes.cls]
        print(f"[클래스 확인] 검출된 클래스: {detected}")

    masks = []
    for i, m in enumerate(res.masks.data):
        # 클래스 필터링
        cls_id   = int(res.boxes.cls[i].item())
        cls_name = res.names[cls_id]
        if cls_name in EXCLUDE_CLASSES:
            continue

        mask_np = m.cpu().numpy()
        mask_resized = cv2.resize(
            mask_np.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        
        # 마스크 경계에서 2px씩 깎아내는 침식 적용
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
        mask_resized = cv2.erode(mask_resized.astype(np.uint8), kernel).astype(bool)
        
        masks.append(mask_resized)
    return masks


# ── Step 2: 노이즈 제거 ──────────────────────────────────
def filter_small_contours(binary_mask, min_area=MIN_AREA):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contours if cv2.contourArea(c) >= min_area]


# ── Step 3: 반폭 계산 ────────────────────────────────────
def calc_half_width(contour):
    """minAreaRect 짧은 변 / 2 = 반폭"""
    _, (w, h), _ = cv2.minAreaRect(contour)
    return min(w, h) / 2.0

def get_instance_half_width(contours, max_hw=MAX_HALF_WIDTH):
    """instance 내 조각들의 반폭 최댓값, 상한 클리핑"""
    candidates = [calc_half_width(c) for c in contours]
    if not candidates:
        return 0.0
    return min(max(candidates), max_hw)


# ── Step 4: instance 중심선 추출 (minAreaRect 장축) ──────
def get_centerline(mask_bin, min_aspect=CENTERLINE_MIN_ASPECT):
    """
    마스크 내부 픽셀 전체로 PCA → 장축 방향 중심선 반환
    mask_bin: uint8 H×W (0/255)
    반환: (pt1, pt2) 또는 None
    """
    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 5:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    center  = mean[0]
    axis    = eigenvectors[0]   # 장축
    perp    = eigenvectors[1]   # 단축

    projections = (pts - center) @ axis
    span  = projections.max() - projections.min()
    width = ((pts - center) @ perp).ptp()

    if width == 0:
        return None

    if min_aspect > 0 and span / width < min_aspect:
        return None

    pt1 = (center + axis * projections.min()).astype(int)
    pt2 = (center + axis * projections.max()).astype(int)

    return tuple(pt1), tuple(pt2)

# ── Step 5: 접지점 ↔ 중심선 거리 계산 ───────────────────
def point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0:
        return np.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return np.hypot(px - proj_x, py - proj_y)


def compute_distances(lane_data, kx, ky):
    for data in lane_data:
        pt1, pt2 = data["centerline"]
        dist = point_to_segment_dist(kx, ky,
                                     pt1[0], pt1[1],
                                     pt2[0], pt2[1])
        data["dist_to_kp"] = dist


# ── Step 6: 침범 판정 ────────────────────────────────────
def check_violation(lane_data):
    results = []
    for i, data in enumerate(lane_data):
        dist     = data["dist_to_kp"]
        half_w   = data["half_width"]
        violated = dist < half_w
        results.append({
            "lane_idx"  : i,
            "dist"      : dist,
            "half_width": half_w,
            "violated"  : violated,
        })
    return results


# ── 시각화 ───────────────────────────────────────────────
def draw_results(frame, lane_data, kx, ky, violation_results):
    vis = frame.copy()

    for i, (data, vr) in enumerate(zip(lane_data, violation_results)):
        color      = get_color(i)
        violated   = vr["violated"]
        draw_color = (0, 0, 255) if violated else color

        overlay = vis.copy()
        overlay[data["mask_bin"] > 0] = draw_color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        pt1, pt2 = data["centerline"]
        cv2.line(vis, pt1, pt2, draw_color, 2)

        tx, ty = pt1
        label = (f"hw={data['half_width']:.1f} "
                 f"d={vr['dist']:.1f} "
                 f"{'[침범]' if violated else 'OK'}")
        cv2.putText(vis, label, (tx, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1)

    cv2.circle(vis, (kx, ky), 8, (0, 0, 255), -1)
    cv2.circle(vis, (kx, ky), 8, (255, 255, 255), 2)
    cv2.putText(vis, f"KP({kx},{ky})",
                (kx + 10, ky - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    any_viol = any(vr["violated"] for vr in violation_results)
    status   = "VIOLATION" if any_viol else "NORMAL"
    s_color  = (0, 0, 255) if any_viol else (0, 255, 0)
    cv2.putText(vis, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, s_color, 3)
    cv2.putText(vis, f"lanes: {len(lane_data)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


# ── 메인 파이프라인 ──────────────────────────────────────
def process_frame(model, frame, kx, ky):
    instance_masks = run_segmentation(model, frame)

    lane_data = []
    for mask_bool in instance_masks:
        mask_bin = (mask_bool * 255).astype(np.uint8)
        contours = filter_small_contours(mask_bin)
        if not contours:
            continue

        half_width = get_instance_half_width(contours)
        largest    = max(contours, key=cv2.contourArea)
        centerline = get_centerline(mask_bin)
        if centerline is None:   # 비율 불안정 → skip
            continue

        lane_data.append({
            "mask_bin"  : mask_bin,
            "contours"  : contours,
            "half_width": half_width,
            "centerline": centerline,
        })

    compute_distances(lane_data, kx, ky)
    violation_results = check_violation(lane_data)
    vis = draw_results(frame, lane_data, kx, ky, violation_results)
    return vis, lane_data, violation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",   required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save",  default=None)
    args = parser.parse_args()

    model = YOLO(args.model, task='segment')
    frame = cv2.imread(args.img)
    if frame is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {args.img}")
        return

    instance_masks = run_segmentation(model, frame)
    vis = frame.copy()

    lane_count = 0
    for mask_bool in instance_masks:
        mask_bin = (mask_bool * 255).astype(np.uint8)
        contours = filter_small_contours(mask_bin)
        if not contours:
            continue

        centerline = get_centerline(mask_bin)
        if centerline is None:
            continue

        # 노란색 마스크 오버레이
        overlay = vis.copy()
        overlay[mask_bin > 0] = (0, 215, 255)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # 중심선
        pt1, pt2 = centerline
        cv2.line(vis, pt1, pt2, (0, 215, 255), 2)
        lane_count += 1

    cv2.putText(vis, f"lanes: {lane_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    print(f"✅ 검출된 차선 수: {lane_count}")

    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"💾 저장: {args.save}")
    else:
        cv2.imshow("Lane Checker", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

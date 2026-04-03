"""
COCO JSON (Roboflow polygon) → CLRNet 학습 포맷 변환 스크립트

입력: Roboflow에서 내보낸 COCO JSON (_annotations.coco.json)
출력: CLRNet이 요구하는 JSON Lines 파일 (train.json / valid.json / test.json)

변환 시 이미지를 90도 시계방향으로 회전한다고 가정:
  원본: 1920×1080
  회전 후: 1080×1920 (너비×높이)
  h_samples: 0~1920 (회전 후 높이)
"""

import json
from pathlib import Path

# ──────────────────────────────────────────
#  설정값
# ──────────────────────────────────────────
DATASET_ROOT  = "/home/sukja/drone_Referee/dataset/drone_lane"
ORIG_W        = 1920   # 원본 너비
ORIG_H        = 1080   # 원본 높이
H_SAMPLE_STEP = 10
# ──────────────────────────────────────────


def rotate_polygon_90cw(segmentation, orig_w, orig_h):
    """polygon 좌표를 90도 시계방향 회전
    원본 (x, y) → 회전 후 (orig_h - 1 - y, x)
    회전 후 이미지 크기: width=orig_h, height=orig_w
    """
    coords = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    rotated = []
    for i in range(0, len(coords) - 1, 2):
        x, y = coords[i], coords[i + 1]
        new_x = orig_h - 1 - y
        new_y = x
        rotated.extend([new_x, new_y])
    return [rotated]


def polygon_to_centerline(segmentation: list, h_samples: list) -> list:
    """polygon 좌표 → h_samples 각 y에서의 중심 x좌표 추출
    해당 y에 polygon이 없으면 -2 반환
    """
    coords = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]

    y_vals = [p[1] for p in points]
    y_min = min(y_vals)
    y_max = max(y_vals)

    result = []
    for y in h_samples:
        if y < y_min or y > y_max:
            result.append(-2)
            continue

        x_at_y = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]

            if min(y1, y2) <= y <= max(y1, y2):
                if y1 == y2:
                    x_at_y.extend([x1, x2])
                else:
                    t = (y - y1) / (y2 - y1)
                    x = x1 + t * (x2 - x1)
                    x_at_y.append(x)

        if x_at_y:
            center_x = (max(x_at_y) + min(x_at_y)) / 2
            result.append(round(center_x, 1))
        else:
            result.append(-2)

    return result


def convert_split(split: str):
    coco_path = Path(DATASET_ROOT) / split / "_annotations.coco.json"
    out_path  = Path(DATASET_ROOT) / f"{split}.json"

    if not coco_path.exists():
        print(f"⚠️  {coco_path} 없음, 스킵")
        return

    with open(coco_path) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    id_to_anns = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # 회전 후 높이 = 원본 너비 = ORIG_W
    h_samples = list(range(0, ORIG_W, H_SAMPLE_STEP))

    lines = []
    skip_count = 0

    for img_id, file_name in id_to_file.items():
        anns = id_to_anns.get(img_id, [])

        if not anns:
            skip_count += 1
            continue

        lanes = []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg:
                continue

            seg_rotated = rotate_polygon_90cw(seg, ORIG_W, ORIG_H)
            lane_xs = polygon_to_centerline(seg_rotated, h_samples)

            valid_count = sum(1 for x in lane_xs if x != -2)
            if valid_count < 3:
                continue

            lanes.append(lane_xs)

        if not lanes:
            skip_count += 1
            continue

        entry = {
            "lanes":     lanes,
            "h_samples": h_samples,
            "raw_file":  f"{split}/images/{file_name}"
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ {split}: {len(lines)}개 이미지 변환 완료 → {out_path}")
    if skip_count:
        print(f"   ⚠️  {skip_count}개 이미지 스킵")


def main():
    print(f"📂 데이터셋 루트: {DATASET_ROOT}")
    print(f"   원본 크기: {ORIG_W}×{ORIG_H}")
    print(f"   회전 후:   {ORIG_H}×{ORIG_W} (너비×높이)")
    print(f"   h_sample: 0~{ORIG_W}, 간격: {H_SAMPLE_STEP}px")
    print()

    convert_split("train")
    convert_split("valid")
    convert_split("test")

    print()
    print("🎉 변환 완료!")


if __name__ == "__main__":
    main()

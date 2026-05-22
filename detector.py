import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ──────────────────────────────────────────
#  설정값
# ──────────────────────────────────────────
DEVICE      = "cuda:0"           # 추론 장치 (GPU 없으면 "cpu")
CONF_THRESH = 0.5                # 이 신뢰도 이상인 bbox만 사용
KP_THRESH   = 0.7                # 이 신뢰도 이상인 keypoint만 유효로 판단
# ──────────────────────────────────────────


class WheelDetector:

    def __init__(self, model_path: str = None):
        print(f"🔍 모델 로딩 중: {model_path}")
        if model_path is None:
            raise ValueError("model_path를 반드시 지정해야 합니다.")
        try:
            self.model = YOLO(model_path, task="pose")
            print("✅ 모델 로딩 완료")
        except FileNotFoundError:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        except Exception as e:
            raise RuntimeError(f"모델 로딩 실패: {e}")

    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        CLAHE 처리된 프레임을 받아 바퀴 검출 결과를 반환.

        반환값 (list): 바퀴 1개당 dict 1개
        [
            {
                "bbox"    : (x1, y1, x2, y2),  # 바퀴 bbox 픽셀 좌표
                "conf"    : 0.92,               # bbox 신뢰도
                "keypoint": (cx, cy),           # 접지점 좌표
                "kp_valid": True                # 신뢰도가 KP_THRESH 이상이면 True
                                                # False면 bbox 하단 중앙으로 fallback
            },
            ...
        ]
        """
        raw = self.model.predict(
            source=frame,
            verbose=False,
            stream=False,
            conf=CONF_THRESH,
            device=DEVICE
        )
        result: Results = raw[0].cpu()

        detections = []

        # 검출된 바퀴가 없으면 빈 리스트 반환
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for i, box in enumerate(result.boxes):

            # ── bbox ──────────────────────────────────────
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # ── keypoint (접지점) ──────────────────────────
            keypoint = None
            kp_valid = False

            if result.keypoints is not None and i < len(result.keypoints):
                kps     = result.keypoints[i]
                kp_xy   = kps.xy[0]      # shape: (num_kp, 2)
                kp_conf = kps.conf[0]    # shape: (num_kp,)

                # 접지점 keypoint 1개만 라벨링했으므로 index 0 사용
                if len(kp_xy) > 0:
                    kx = float(kp_xy[0][0])
                    ky = float(kp_xy[0][1])
                    kc = float(kp_conf[0])

                    if kc >= KP_THRESH:
                        # 신뢰도 충분 → keypoint 그대로 사용
                        keypoint = (int(kx), int(ky))
                        kp_valid = True
                    else:
                        # 신뢰도 낮음 → bbox 하단 중앙으로 fallback
                        keypoint = ((x1 + x2) // 2, y2)
                        kp_valid = False
            else:
                # keypoint 자체가 없음 → bbox 하단 중앙으로 fallback
                keypoint = ((x1 + x2) // 2, y2)
                kp_valid = False

            detections.append({
                "bbox":     (x1, y1, x2, y2),
                "conf":     conf,
                "keypoint": keypoint,
                "kp_valid": kp_valid
            })

        return detections

if __name__ == "__main__":
    import argparse, cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--img",   required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save",  default=None)
    args = parser.parse_args()

    detector = WheelDetector(model_path=args.model)
    frame    = cv2.imread(args.img)

    wheels = detector.predict(frame)
    vis    = frame.copy()

    print(f"\n✅ 검출된 바퀴 수: {len(wheels)}")
    for i, w in enumerate(wheels):
        x1, y1, x2, y2 = w["bbox"]
        kx, ky          = w["keypoint"]
        kp_valid        = w["kp_valid"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (kx, ky), 6, (0, 0, 255), -1)
        cv2.circle(vis, (kx, ky), 6, (255, 255, 255), 1)
        label = f""
        cv2.putText(vis, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  바퀴 {i}: bbox=({x1},{y1},{x2},{y2})  "
              f"접지점=({kx},{ky})  {'KP✅' if kp_valid else 'fallback⚠'}")

    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"💾 저장: {args.save}")
    else:
        cv2.imshow("Wheel Detector", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

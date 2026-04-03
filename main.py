import cv2
from preprocess import MODE, run_image_mode, video_stream, camera_stream
from detector import WheelDetector


def main():
    if MODE == 'image':
        print("🖼️  이미지 모드로 실행합니다.")
        run_image_mode()

    elif MODE in ('video', 'camera'):
        if MODE == 'video':
            print("🎬 영상 모드로 실행합니다. 종료하려면 'q'를 누르세요.")
            stream = video_stream()
        else:
            print("🎥 카메라 모드로 실행합니다. 종료하려면 'q'를 누르세요.")
            stream = camera_stream()

        detector = WheelDetector()

        for frame in stream:
            # ── 1. YOLO 추론 ──────────────────────────────
            wheels = detector.predict(frame)

            # ── 2. 이 아래에 다음 모듈들이 순서대로 연결 예정 ──
            # lanes   = lane_detector.detect(frame)         # CLRNet 추론
            # results = lane_checker.check(wheels, lanes)   # 침범 판별
            # tracker.update(results)                       # 침범 타이머
            # display = visualizer.draw(frame, tracker)     # 시각화
            # ─────────────────────────────────────────────

            # 현재는 bbox + keypoint 시각화만 출력
            for w in wheels:
                x1, y1, x2, y2 = w["bbox"]
                kx, ky          = w["keypoint"]
                kp_valid        = w["kp_valid"]

                color = (0, 255, 0) if kp_valid else (0, 215, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (kx, ky), 5, (0, 0, 255), -1)

                label = f"{w['conf']:.2f} {'KP' if kp_valid else 'FB'}"
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Drone Referee", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("✅ 종료")

    else:
        print(f"❌ 알 수 없는 MODE: '{MODE}'"
              f" → 'image', 'video', 'camera' 중 하나로 설정하세요.")


if __name__ == '__main__':
    main()
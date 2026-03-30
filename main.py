import cv2
from preprocess import MODE, run_image_mode, camera_stream

def main():
    if MODE == 'image':
        # ── 이미지 모드: CLAHE 처리 후 저장하고 종료 ──
        print("🖼️  이미지 모드로 실행합니다.")
        run_image_mode()

    elif MODE == 'camera':
        # ── 카메라 모드: 실시간 파이프라인 ──
        print("🎥  카메라 모드로 실행합니다. 종료하려면 'q'를 누르세요.")

        for frame in camera_stream():
            wheels   = detector.predict(frame)            # YOLO 추론
            # ── 이 아래에 다음 모듈들이 순서대로 연결될 예정 ──
            # results  = lane_checker.check(frame, wheels)  # ROI 흰색 판별
            # tracker.update(results)                       # 침범 타이머
            # display  = visualizer.draw(frame, tracker)    # 시각화
            # ─────────────────────────────────────────────

            # 현재는 bbox + keypoint 확인용 시각화만 출력
            for w in wheels:
                x1, y1, x2, y2 = w["bbox"]
                kx, ky          = w["keypoint"]
                kp_valid        = w["kp_valid"]
 
                # bbox: 초록(keypoint 유효) / 노랑(fallback)
                color = (0, 255, 0) if kp_valid else (0, 215, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
                # 접지점: 빨간 점
                cv2.circle(frame, (kx, ky), 5, (0, 0, 255), -1)
 
                # 신뢰도 텍스트
                label = f"{w['conf']:.2f} {'kp' if kp_valid else 'fb'}"
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
 
            cv2.imshow("Drone Referee", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("✅ 종료")

    else:
        print(f"❌ 알 수 없는 MODE: '{MODE}' → 'image' 또는 'camera'로 설정하세요.")


if __name__ == '__main__':
    main()

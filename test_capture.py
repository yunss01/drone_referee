"""
캡처보드 연결 테스트
- 연결된 카메라/캡처보드 장치를 자동으로 찾아서 영상 출력
- 종료: q 키
"""

import cv2

CAM_NUM = 0     # None으로 두면 자동 탐색 후 가장 큰 인덱스(마지막 카메라)를 사용


def find_capture_devices(max_index=5):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
        cap.release()
    return found


def main():
    if CAM_NUM is not None:
        cam_index = CAM_NUM
        print(f"📷 지정된 장치 {cam_index} 번으로 바로 연결합니다.\n")
    else:
        print("🔍 캡처 장치 탐색 중...")
        devices = find_capture_devices()

        if not devices:
            print("❌ 연결된 캡처 장치를 찾을 수 없습니다.")
            return

        print(f"✅ 장치 발견: 인덱스 {devices}")

        # 장치가 여러 개면 가장 큰 번호 (캡처보드일 가능성 높음)
        cam_index = devices[-1]
        print(f"📷 장치 {cam_index} 번으로 연결 시도...\n")

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"   포맷: {''.join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   해상도: {w}x{h} @ {fps:.1f}fps")
    print(f"   종료: q 키\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  프레임 읽기 실패")
            break

        cv2.imshow(f"Capture Test (device {cam_index})", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 종료")


if __name__ == "__main__":
    main()

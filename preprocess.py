import cv2
import os

# ──────────────────────────────────────────
#  모드 설정 (코드에서 직접 변경)
#
#  MODE = 'image'   : 지정 폴더의 이미지들을 CLAHE 처리 후 저장
#  MODE = 'camera'  : USB 카메라(드론 영상)를 실시간 CLAHE 처리 후 프레임 반환
# ──────────────────────────────────────────
MODE = 'camera'

# [image 모드] 입력 이미지 폴더 / 출력 폴더
IMAGE_INPUT_DIR  = 'pre_process/frames_raw/'
IMAGE_OUTPUT_DIR = 'pre_process/frames_clahe/'

# [camera 모드] 카메라 장치 번호 (ls /dev/video* 로 확인)
CAM_NUM = 0

# CLAHE 파라미터
CLIP_LIMIT = 2.0
TILE_SIZE  = (4, 4)
# ──────────────────────────────────────────


def _apply_clahe(frame, clahe):
    """BGR 프레임에 CLAHE 적용 후 반환"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    result = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def run_image_mode():
    """
    [image 모드]
    IMAGE_INPUT_DIR 의 이미지를 CLAHE 처리 후 IMAGE_OUTPUT_DIR 에 저장.
    main.py에서 호출할 필요 없이 단독 실행도 가능.
    """
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)

    img_files = sorted([
        f for f in os.listdir(IMAGE_INPUT_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not img_files:
        print(f"⚠️  이미지 없음: {IMAGE_INPUT_DIR}")
        return

    for idx, fname in enumerate(img_files):
        src_path = os.path.join(IMAGE_INPUT_DIR, fname)
        frame = cv2.imread(src_path)
        if frame is None:
            print(f"  스킵 (읽기 실패): {fname}")
            continue

        result = _apply_clahe(frame, clahe)
        dst_path = os.path.join(IMAGE_OUTPUT_DIR, f"clahe_{idx:05d}.jpg")
        cv2.imwrite(dst_path, result)

    print(f"✅ CLAHE 완료: {len(img_files)}장 → {IMAGE_OUTPUT_DIR}")


def camera_stream():
    """
    [camera 모드]
    USB 카메라에서 프레임을 읽어 CLAHE 처리 후 yield.
    main.py의 루프에서 사용:

        for frame in camera_stream():
            # frame = CLAHE 처리된 BGR 프레임
            ...
    """
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    cap = cv2.VideoCapture(CAM_NUM)

    if not cap.isOpened():
        print(f"❌ 카메라 열기 실패: /dev/video{CAM_NUM}")
        return

    print(f"📷 카메라 스트림 시작 (장치 번호: {CAM_NUM})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  프레임 읽기 실패, 재시도 중...")
                continue

            yield _apply_clahe(frame, clahe)

    except GeneratorExit:
        pass
    finally:
        cap.release()
        print("📷 카메라 스트림 종료")

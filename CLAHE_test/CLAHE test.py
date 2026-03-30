import cv2
import numpy as np
import os

def apply_clahe_to_image(image_path, clip_limit=3.0, grid_size=(8, 8)):
    # 1. 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return None

    # 2. 색상 공간 변환 (BGR -> LAB)
    # 밝기 채널(L)에만 CLAHE를 적용하기 위해 변환합니다.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3. CLAHE 객체 생성 및 적용
    # clipLimit: 대비 제한 임계값 (높을수록 대비가 강해지나 노이즈 증가)
    # tileGridSize: 이미지를 나눌 타일 크기
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l)

    # 4. 채널 병합 및 다시 BGR로 변환
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

# --- 실행 부분 ---
image_files = ['1.jpeg', '2.png'] # 테스트할 이미지 파일명 2개
output_folder = './clahe_results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in image_files:
    result = apply_clahe_to_image(file)
    if result is not None:
        # 결과 저장
        cv2.imwrite(os.path.join(output_folder, f"clahe_{file}"), result)
        
        # 화면 출력 (확인용)
        original = cv2.imread(file)
        combined = np.hstack((original, result)) # 원본과 결과 나란히 배치
        cv2.imshow(f"Original vs CLAHE - {file}", combined)
        cv2.waitKey(0)

cv2.destroyAllWindows()

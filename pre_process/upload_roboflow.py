from roboflow import Roboflow
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로보플로우 API 키와 프로젝트 정보 설정
rf = Roboflow(api_key="z82bfehXUtoIQzqZcGSK")
project = rf.workspace("visionsw-starterpack").project("drone_referee_line")

# 이미지가 들어있는 로컬 폴더 경로
image_folder = "/home/sukja/drone_Referee/dataset/3x_video"

# 업로드할 이미지 목록
image_files = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.endswith(('.jpg', '.jpeg', '.png'))
]

total = len(image_files)
print(f"총 {total}장 업로드 시작")

def upload_image(image_path):
    try:
        project.upload(image_path, num_retry_attempts=3)
        return f"✅ {os.path.basename(image_path)}"
    except Exception as e:
        return f"❌ {os.path.basename(image_path)}: {e}"

completed = 0
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(upload_image, p): p for p in image_files}
    for future in as_completed(futures):
        completed += 1
        result = future.result()
        print(f"[{completed}/{total}] {result}")

print("모든 이미지 업로드 완료!")

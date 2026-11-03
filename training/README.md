# 로컬 학습 자료

학습 코드·데이터·가중치·학습 결과를 로컬에 보존합니다. 이 README 외의 내용은 GitHub에서 제외합니다.

| 경로 | 내용 |
| --- | --- |
| `segmentation/` | 기존 `segmentation DL model/`: 차선 세그멘테이션 학습 |
| `wheel_pose/` | 기존 `wheel point DL model/`: 바퀴 접지점 학습 |
| `preprocessing/` | 기존 `pre_process/`: 프레임 추출, CLAHE, 데이터 업로드 도구와 영상 |

학습 진입점은 각 폴더의 `train.py`입니다. 데이터·초기 모델·결과 경로는 스크립트 위치를 기준으로 찾습니다. 과거 학습 결과 자체는 변경하지 않습니다. 실제 학습 실행과 외부 데이터 업로드는 별도로 진행합니다.

운영에서는 이 폴더 대신 루트의 [model/](../model/README.md)에 배치한 모델을 사용합니다.

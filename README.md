# Drone Referee

드론 영상에서 차량 바퀴의 접지점과 차선을 검출해 차선 침범을 판정하는 시스템입니다. 캡처보드 또는 유튜브 라이브 영상을 입력받아 판정 결과를 표시하고 송출합니다.

## 판정 방식

CLAHE 전처리 후 차선 세그멘테이션과 바퀴 접지점 검출을 수행합니다. 차선 마스크에서 PCA 중심선과 반폭을 구하고, 접지점과 중심선 구간 사이의 거리가 반폭보다 작으면 침범으로 판정합니다. 중심선 구간 바깥으로 투영되는 점은 구간 끝점까지의 거리를 사용합니다.

접지점 신뢰도가 낮으면 바퀴 bbox 하단 중앙을 사용합니다. 침범이 설정 시간 이상 지속되면 확정하며, 운영 기본값은 0.1초입니다.

| 바퀴 bbox 색상 | 상태 |
| --- | --- |
| 초록색 | 정상, 유효한 접지점 사용 |
| 노란색 | 정상, bbox 하단 중앙으로 접지점 대체 |
| 주황색 | 침범 감지, 확정 시간 미충족 |
| 빨간색 | 침범 확정 |

## 주요 파일

| 경로 | 역할 |
| --- | --- |
| `main_capture_dual_youtube.py` | 캡처보드 입력, 원본·처리본 이중 송출 |
| `main.py` | 유튜브 라이브 및 로컬 영상·이미지 처리 |
| `detector.py` | 바퀴 bbox와 접지점 검출 |
| `lane_checker.py` | 차선 추출, 중심선 계산, 침범 판정 |
| `preprocess.py` | CLAHE 전처리 |
| `violation_tracker.py` | 침범 유지 시간과 확정 상태 관리 |
| `runtime_config.py` | 입력·송출 주소 설정 읽기 |
| `tools/` | 성능 측정, 캡처 장치 확인 |
| `dataset/편집본/` | 테스트 영상 12개 |

## 실행 환경

Linux, Python 3.10 이상, CUDA 지원 PyTorch·torchvision, TensorRT, FFmpeg(`ffmpeg`, `ffprobe`)를 사용합니다. GPU에 맞는 PyTorch와 TensorRT를 설치한 뒤 Python 의존성을 설치합니다.

```bash
python3 -m pip install -r requirements.txt
```

기본 모델은 별도로 준비해 프로젝트 루트의 `model/`에 배치합니다.

- `model/best_seg_rev03_FP16.engine`: 차선 세그멘테이션
- `model/best_referee_FP16.engine`: 바퀴 bbox·접지점 검출

TensorRT 엔진은 실행 장비의 GPU·CUDA·TensorRT 환경과 호환되어야 합니다.

## 설정 및 실행

프로젝트 루트에서 설정 예시를 복사하고 입력·송출 주소를 작성합니다. 기존 설정 파일이 있으면 그대로 사용합니다.

```bash
cp -n local_settings.example.json local_settings.json
```

| 설정 | 용도 |
| --- | --- |
| `DRONE_INPUT_STREAM_URL` | 유튜브 입력 주소 |
| `DRONE_OUTPUT_STREAM_RTMP_URL` | `main.py` 처리본 송출 주소 |
| `DRONE_RAW_STREAM_RTMP_URL` | 캡처보드 원본 송출 주소 |
| `DRONE_PROCESSED_STREAM_RTMP_URL` | 캡처보드 처리본 송출 주소 |

`local_settings.json`은 Git에서 제외합니다. 송출 주소에는 서버 주소와 스트림 키를 포함한 전체 RTMP URL을 입력합니다. 같은 이름의 환경변수가 있으면 JSON보다 우선합니다.

캡처보드 입력은 스크립트 상단에서 장치 번호와 송출 설정을 확인한 뒤 실행합니다.

```bash
python3 main_capture_dual_youtube.py
```

기본 요청 입력은 1920×1080, 30fps, MJPG입니다. 처리 영상은 긴 변 1280으로 줄이고 두 프레임마다 처리하므로, 해당 입력에서 처리본은 1280×720, 15fps입니다.

유튜브 입력 또는 로컬 파일은 `main.py`의 `MODE`를 `rtmp`, `video`, `image` 중 선택하고 입력 경로를 지정합니다.

```bash
python3 main.py
```

송출을 사용하지 않으면 해당 스크립트의 `*_STREAM_ENABLED`를 `False`로 설정합니다. 종료는 `Ctrl+C`입니다.

## 테스트 영상

`dataset/편집본/test_1.mp4`부터 `test_12.mp4`까지 12개 영상이 있습니다. 모두 1920×1080, 30fps이며 Git LFS로 관리합니다. 저장소를 받은 뒤 영상이 내려받아지지 않았다면 다음 명령을 실행합니다.

```bash
git lfs install
git lfs pull
```

## 진단 도구

프로젝트 루트에서 실행합니다. 캡처 확인에는 GUI가 필요하고, 벤치마크에는 `model/best_seg_rev03.pt`, `model/best_referee.pt`도 필요합니다.

```bash
python3 -m tools.test_capture
python3 -m tools.benchmark --img 이미지경로.png --runs 30
```

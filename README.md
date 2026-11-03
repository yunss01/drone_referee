# Drone Referee

드론 영상에서 차량 바퀴의 접지점과 차선 마스크를 추출해 차선 침범을 판정하는 시스템입니다. 차선 세그멘테이션과 바퀴 접지점 모델을 함께 사용하며, 캡처보드 입력 또는 유튜브 라이브 입력을 처리합니다.

## 처리 과정

1. 입력 프레임에 CLAHE를 적용합니다.
2. 세그멘테이션 모델로 차선 인스턴스를 찾고, PCA로 중심선과 차선 반폭을 구합니다.
3. 바퀴 모델로 bbox와 접지점을 검출합니다. 접지점 신뢰도가 낮으면 bbox 하단 중앙을 사용합니다.
4. 접지점에서 **유한한 중심선 구간**까지의 거리가 차선 반폭보다 작으면 침범으로 판정합니다. 투영계수는 `[0, 1]`로 제한합니다.
5. 침범이 설정 시간 이상 이어지면 확정하고 결과 영상에 표시합니다.

바퀴 bbox 색은 초록색(정상·유효한 접지점), 노란색(정상·접지점 대체 사용), 주황색(침범 감지·시간 미충족), 빨간색(침범 확정)입니다. 운영 코드의 기본 확정 시간은 **0.1초**이며, 별도 오프라인 비교 실험의 0.2초 설정과 구분합니다.

## 저장소 구성

```text
drone_Referee/
├── main_capture_dual_youtube.py  # 캡처보드 입력, 원본/처리본 이중 송출
├── main.py                      # 유튜브 입력 및 영상/이미지 처리
├── detector.py                  # 바퀴 bbox 및 접지점 검출
├── lane_checker.py              # 차선 추출, 중심선, 거리 기반 판정
├── preprocess.py                # CLAHE 전처리
├── violation_tracker.py         # 침범 유지 시간 및 확정 상태
├── runtime_config.py            # 로컬 설정 / 환경변수 읽기
├── local_settings.example.json  # 입력 주소와 송출 주소 설정 예시
├── requirements.txt
├── tools/                       # 성능 측정, 캡처 장치 확인
├── docs/                        # 폴더 정리 내역
├── model/                       # 모델 배치 안내; 실제 모델은 로컬 보관
├── dataset/                     # 입력 데이터 안내; 영상은 로컬 보관
├── training/                    # 학습 자료 안내; 코드·데이터는 로컬 보관
└── archive/                     # 보관 자료 안내; 연구·논문 자료는 로컬 보관
```

GitHub에는 운영 코드, 진단 도구, 문서를 관리합니다. 학습 데이터·영상·모델 바이너리·미적용 모델 연구·논문용 그림·특허 실험 자료·로그·스풀은 로컬에 보관하고 Git 추적에서 제외합니다. 전체 이동 내역은 [폴더 정리 문서](docs/repository_layout.md)에 있습니다.

## 실행 준비

- Linux, Python 3.10 이상, OpenCV, NumPy, Ultralytics, CUDA 지원 PyTorch가 필요합니다.
- 영상 저장·중계에 `ffmpeg`, 메타데이터 확인에 `ffprobe`, 유튜브 입력에 `yt-dlp`를 사용합니다.
- 기본 모델은 TensorRT FP16 엔진입니다. GPU·CUDA·TensorRT와 호환되는 엔진을 [model/](model/README.md)에 배치해야 합니다. TensorRT는 장비 환경에 맞게 별도로 설치합니다.
- 캡처보드는 V4L2 장치(`/dev/video*`)로 연결합니다. NVENC 사용이 불가능하면 코드의 CPU 인코더 대체 설정을 확인합니다.

프로젝트 루트에서 실행합니다. CUDA에 맞는 PyTorch와 torchvision을 먼저 설치한 뒤 나머지 Python 패키지를 설치합니다.

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt`는 필요한 직접 의존성 목록이며, 장비별 GPU 환경까지 고정하는 잠금 파일은 아닙니다.

### 주소 설정

새로 설치한 환경에서는 예시 파일을 복사한 뒤 주소를 입력합니다.

```bash
cp -n local_settings.example.json local_settings.json
```

| 설정 이름 | 용도 |
| --- | --- |
| `DRONE_INPUT_STREAM_URL` | `main.py`가 읽을 유튜브 라이브 주소 |
| `DRONE_OUTPUT_STREAM_RTMP_URL` | `main.py` 결과 송출 주소 |
| `DRONE_RAW_STREAM_RTMP_URL` | 캡처보드 원본 송출 주소 |
| `DRONE_PROCESSED_STREAM_RTMP_URL` | 캡처보드 처리본 송출 주소 |

송출 주소는 송출 서버 주소와 스트림 키를 포함한 전체 RTMP URL입니다. 같은 이름의 환경변수가 있으면 로컬 JSON보다 우선하며, 빈 환경변수도 우선 적용됩니다. `local_settings.json`은 Git에서 제외합니다. 송출하지 않는 경로는 해당 스크립트의 `*_STREAM_ENABLED` 설정을 `False`로 바꿉니다.

### 캡처보드 입력

`main_capture_dual_youtube.py` 상단에서 장치 번호와 송출 설정을 확인한 뒤 실행합니다.

```bash
python3 main_capture_dual_youtube.py
```

기본 요청 입력은 **1920×1080, 30fps, MJPG**입니다. 처리 영상은 긴 변 1280으로 줄이고 두 프레임마다 한 번 처리합니다. FHD·30fps 입력이라면 처리 크기는 1280×720, 처리본 출력은 15fps입니다. 실제 장치 입력값은 로그로 확인합니다.

### 유튜브 입력 / 로컬 영상·이미지

`main.py`의 `MODE`를 `rtmp`, `video`, `image` 중 선택합니다. `video`와 `image`에서는 상단의 입력 파일 경로도 지정합니다.

```bash
python3 main.py
```

두 운영 스크립트 모두 실행 시 입력 장치나 네트워크에 접근하고, 활성화된 송출을 시작할 수 있습니다. 종료는 `Ctrl+C`로 합니다. 종료 시 남은 스풀 송출을 기다리는 설정이 있으므로 로그의 종료 상태를 확인합니다.

## 진단 도구

프로젝트 루트에서 모듈 방식으로 실행합니다.

```bash
# 캡처 장치를 열어 화면 표시 (GUI 필요)
python3 -m tools.test_capture

# 같은 이미지에서 PT / TensorRT FP16 단계별 성능 비교
python3 -m tools.benchmark --img 이미지경로.png --runs 30
```

벤치마크에는 PT 모델과 FP16 엔진 모두 필요합니다. 예전에 루트에 있던 `benchmark.py`, `test_capture.py`는 `tools/`로 이동했습니다.

## 로컬 자료와 운영 기록

- 입력 영상과 편집본: `dataset/`
- 차선·접지점 학습 및 전처리 자료: `training/`
- PINet / UnLanedet 비교 연구와 그림·과거 결과: `archive/`
- 특허 비교 실험 코드와 CSV: `특허용 자료들/`
- 운영 로그: `logs/`
- 임시 세그먼트와 송출 스풀: `spool_sessions/`, `spool_capture_sessions/`

원본 코드와 모델을 준비한 후 사용합니다. 학습이나 특허 실험은 운영 프로그램 실행과 별도로 진행하며, 저장된 비교 결과를 독립적인 정답 라벨에 대한 정확도로 해석하지 않습니다.

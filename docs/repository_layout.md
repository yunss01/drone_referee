# 폴더 정리 내역

2026-11-03 기준으로 운영 코드와 로컬 연구 자료를 분리했습니다. 파일은 이동하거나 Git 추적에서 제외했으며, 로컬 자료와 과거 로그는 삭제하지 않았습니다.

## GitHub에 관리하는 범위

운영 진입점 두 개와 공통 판정 모듈은 프로젝트 루트에 유지합니다. 모델·데이터의 기존 루트 경로도 유지하므로 운영 시 상대 경로의 기준은 프로젝트 루트입니다. 진단 도구는 `tools/`, 문서는 `docs/`에 둡니다.

아래 자료 폴더에서는 안내용 README만 관리합니다. `.gitignore`에 이미 포함돼 있어도 과거에 추적하던 파일은 별도로 Git 인덱스에서 제외했습니다.

| 로컬 경로 | 보관 내용 | GitHub |
| --- | --- | --- |
| `model/` | 운영 PT / ONNX / TensorRT 모델 | 안내 문서만 |
| `dataset/` | 원본 영상, 편집본, 이미지·라벨 | 안내 문서만 |
| `training/` | 학습 스크립트·데이터·결과, 전처리 자료 | 안내 문서만 |
| `archive/` | 미적용 모델, 그림, 이전 실험과 결과 | 안내 문서만 |
| `특허용 자료들/` | 비교 실험 코드와 CSV 결과 | 제외 |
| `logs/`, `spool_sessions/`, `spool_capture_sessions/` | 실행 중 생성되는 기록과 스풀 | 제외 |
| `local_settings.json` | 현재 입력 주소와 송출 주소 | 제외 |

## 이동 내역

| 기존 경로 | 정리 후 경로 |
| --- | --- |
| `PINet_new/` | `archive/lane_model_trials/PINet_new/` |
| `UnLanedet/` | `archive/lane_model_trials/UnLanedet/` |
| `CLAHE_test/` | `archive/experiments/CLAHE_test/` |
| `test.py` | `archive/experiments/test.py` |
| `visualize_pca.py` | `archive/paper_figures/visualize_pca.py` |
| `pca_vis.png`, `test_image.png` | `archive/paper_figures/` |
| `test_results/` | `archive/results/test_results/` |
| `segmentation DL model/` | `training/segmentation/` |
| `wheel point DL model/` | `training/wheel_pose/` |
| `pre_process/` | `training/preprocessing/` |
| `benchmark.py` | `tools/benchmark.py` |
| `test_capture.py` | `tools/test_capture.py` |

`dataset/`, `model/`, `특허용 자료들/`, 운영 로그와 스풀은 기존 위치를 유지합니다. `preprocess.py`의 이미지 입출력 예제 경로와 두 학습 진입점의 데이터·모델·결과 경로는 새 위치에 맞춥니다.

## 운영 설정

코드에 직접 들어 있던 입력 라이브 주소와 송출 주소는 `local_settings.json`으로 옮겼습니다. 현재 로컬 설정값을 보존하며, 공개되는 예시 파일에는 빈 문자열만 넣습니다. `runtime_config.py`는 환경변수, 로컬 JSON, 기본값 순서로 설정을 읽습니다.

차선 거리 계산, 접지점 선택, 침범 확정 시간, 영상 처리와 송출 로직은 폴더 정리로 변경하지 않습니다.

## 보존 및 과거 실험

정리 전 전체 파일의 경로·크기·SHA-256, 이동 목록, 변경 전 코드, 기존 미커밋 변경과 Git 인덱스는 이 PC의 프로젝트 바깥 `../drone_Referee_cleanup_backups/` 아래 날짜별 폴더에 보관합니다. 이 백업과 로컬 보관 자료는 GitHub에 업로드하지 않습니다.

특허 실험의 기존 CSV와 설정 파일은 수정하지 않습니다. 설정 파일에 기록된 코드 해시는 당시 코드의 값입니다. 운영 주소 설정 분리 등으로 현재 코드 해시가 달라지면 캐시 검증이 실패할 수 있으므로, 기존 검증을 끄지 말고 변경 전 코드 스냅샷을 확인하거나 현재 코드로 기준 실험부터 다시 생성합니다.

## 향후 삭제 검토 대상

다음은 현재 운영 코드에서 사용하지 않는 자료입니다. 재현·논문·학습 이력에 필요할 수 있어 이번에는 모두 보존했습니다.

- `archive/lane_model_trials/`: 적용하지 않은 PINet / UnLanedet 소스, 모델, 출력.
- `archive/experiments/`: CLAHE 단독 실험, 예전 색상 기반 바퀴 판정 코드.
- `archive/paper_figures/`, `archive/results/`: 그림 생성 코드와 과거 결과.
- `training/`의 이전 학습 결과·중간 가중치와 `model/`의 비운영 포맷: 재학습이나 벤치마크 사용 여부를 확인한 후 판단.

기존 커밋 이력은 유지합니다. GitHub 최신 파일 목록에서 빠진 파일도 과거 커밋에는 남으므로, 이번 정리만으로 기존 저장소의 전체 다운로드 크기가 줄어들지는 않습니다.

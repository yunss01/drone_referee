# 운영 모델

실제 모델 파일은 이 PC에 보존하며 GitHub에서는 제외합니다. 프로젝트 루트에서 실행할 때 다음 상대 경로를 사용합니다.

| 파일 | 용도 |
| --- | --- |
| `best_seg_rev03_FP16.engine` | 기본 차선 세그멘테이션 TensorRT FP16 엔진 |
| `best_referee_FP16.engine` | 기본 바퀴 bbox / 접지점 TensorRT FP16 엔진 |
| `best_seg_rev03.pt` | 벤치마크용 차선 PT 모델 |
| `best_referee.pt` | 벤치마크용 바퀴 PT 모델 |

기존 ONNX, 일반 엔진, FP16 변형 모델도 로컬에 그대로 남아 있습니다. TensorRT 엔진은 GPU와 TensorRT/CUDA 환경의 영향을 받으므로 다른 장비에서는 호환성을 확인하고 필요하면 PT 모델에서 다시 내보냅니다.

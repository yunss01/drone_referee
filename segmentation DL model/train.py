import os
from ultralytics import YOLO

def main():
    # 데이터셋 경로 (Roboflow에서 받은 데이터셋으로 변경)
    data_path = "/home/sukja/drone_Referee/segmentation DL model/drone_Referee_line_seg.v2i.yolov11/data.yaml"

    print("🚀 차선 segmentation 모델 학습을 시작합니다!")
    
    model = YOLO("yolo11m-seg.pt")

    results = model.train(
        data=data_path,
        epochs=150,
        
        # --- 옵티마이저 ---
        optimizer='AdamW',
        lr0=0.001,
        batch=4,
        
        patience=50,
        imgsz=1280,
        device=0,
        
        # --- 🚨 이전 train.py와 핵심 차이점: bbox도 중요 ---
        # 이전: bbox=3.0 (bbox를 대충 봄)
        # 이번: bbox=7.5 (기본값 유지, bbox도 정확해야 ROI 계산이 정확함)
        box=7.5,
        cls=0.5,        # 기본값 유지 (클래스가 wheel 하나라 크게 안 중요)
        
        # --- Data Augmentation ---
        # Roboflow에서 이미 적용: noise, blur
        # 여기서는 Roboflow가 안 해주는 것만 적용
        degrees=30.0,       # 드론 시점 회전 (360은 과함, 30 정도가 현실적)
        perspective=0.0003, # 원근 변환 (드론 각도 변화 시뮬레이션)
        translate=0.1,      # 이동
        scale=0.5,          # 드론 고도 변화 시뮬레이션
        fliplr=0.5,         # 좌우 반전은 유효 (바퀴는 대칭)
        flipud=0.0,         # 상하 반전은 비현실적이라 끔
        mosaic=1.0,         # 여러 이미지 합성 (기본값, 소량 데이터에 효과적)
        
        project="line_seg",
        name="rev03",
        save=True
    )

    print("\n🎉 학습 완료!")

if __name__ == '__main__':
    main()

import os.path as osp
import numpy as np
import cv2
import os
import json
import random
from .base_dataset import BaseDataset
from .transform import DataContainer as DC

# ──────────────────────────────────────────
#  드론 심판 데이터셋
#  - train.json / valid.json 사용
#  - segmentation 마스크 없음
#  - JSON Lines 포맷 (한 줄 = 한 이미지)
# ──────────────────────────────────────────

SPLIT_FILES = {
    'train': 'train.json',
    'val':   'valid.json',
    'test':  'valid.json',
}


class DroneLane(BaseDataset):
    def __init__(self,
                 data_root,
                 split,
                 cut_height=0,
                 processes=None,
                 cfg=None):
        super().__init__(data_root, split, cut_height, processes, cfg)
        self.anno_file = SPLIT_FILES[split]
        self.load_annotations()
        self.h_samples = list(range(0, 1920, 10))

    def load_annotations(self):
        self.logger.info(f'Loading DroneLane annotations from {self.anno_file}...')
        self.data_infos = []
        max_lanes = 0

        anno_path = osp.join(self.data_root, self.anno_file)
        with open(anno_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            y_samples = data['h_samples']
            gt_lanes  = data['lanes']

            # x >= 0인 점만 유효 차선 좌표로 사용
            lanes = [
                [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                for lane in gt_lanes
            ]
            lanes = [lane for lane in lanes if len(lane) > 0]
            max_lanes = max(max_lanes, len(lanes))

            self.data_infos.append({
                'img_path': osp.join(self.data_root, data['raw_file']),
                'img_name': data['raw_file'],
                'mask_path': None,   # 마스크 없음
                'lanes': lanes,
            })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes
        self.logger.info(f'Loaded {len(self.data_infos)} images, max_lanes={max_lanes}')

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]

        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError(f'이미지를 찾을 수 없습니다: {data_info["img_path"]}')

        img = cv2.imread(data_info['img_path'])
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = img[self.cut_height:, :, :]

        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            # segmentation 마스크 대신 빈 마스크 생성
            h, w = img.shape[:2]
            label = np.zeros((h, w), dtype=np.uint8)
            sample.update({'mask': label})

            if self.cut_height != 0:
                new_lanes = []
                for lane in sample['lanes']:
                    new_lanes.append([(x, y - self.cut_height) for (x, y) in lane])
                sample.update({'lanes': new_lanes})

        sample = self.processes(sample)

        meta = {
            'full_img_path': data_info['img_path'],
            'img_name':      data_info['img_name'],
        }
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})

        return sample

#########################################################################
##
##  Data loader for Drone Lane dataset (PINet)
##  - train.json / valid.json 직접 로드
##  - 이미지 90도 시계방향 회전 적용
##
#########################################################################

import math
import numpy as np
import cv2
import json
import random
from copy import deepcopy
from parameters import Parameters


def Translate_Points(point, translation):
    point = point + translation
    return point


def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class Generator(object):

    def __init__(self):
        self.p = Parameters()

        # ── train 데이터 로드 (차선 수와 무관하게 전부 로드) ──
        self.train_data = []
        with open(self.p.train_json) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.train_data.append(json.loads(line))
        random.shuffle(self.train_data)
        self.size_train = len(self.train_data)
        print(f"Train 데이터: {self.size_train}개")

        # ── valid 데이터 로드 ──
        self.test_data = []
        with open(self.p.valid_json) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.test_data.append(json.loads(line))
        self.size_test = len(self.test_data)
        print(f"Valid 데이터: {self.size_test}개")

    #################################################
    ## 학습 데이터 생성 (augmentation 포함)
    #################################################
    def Generate(self, sampling_list=None):
        cuts = [(b, min(b + self.p.batch_size, self.size_train))
                for b in range(0, self.size_train, self.p.batch_size)]
        for start, end in cuts:
            self.inputs, self.target_lanes, self.target_h, \
                self.test_image, self.data_list = self.Resize_data(start, end, sampling_list)

            self.actual_batchsize = self.inputs.shape[0]
            self.Flip()
            self.Translation()
            self.Rotate()
            self.Gaussian()
            self.Change_intensity()
            self.Shadow()

            yield (self.inputs / 255.0, self.target_lanes,
                   self.target_h, self.test_image / 255.0, self.data_list)

    #################################################
    ## 테스트 데이터 생성
    #################################################
    def Generate_Test(self):
        cuts = [(b, min(b + self.p.batch_size, self.size_test))
                for b in range(0, self.size_test, self.p.batch_size)]
        for start, end in cuts:
            test_image, path, ratio_w, ratio_h, target_h, gt = \
                self.Resize_data_test(start, end)
            yield test_image / 255.0, target_h, ratio_w, ratio_h, path, gt

    #################################################
    ## 이미지 로드 + 90도 회전 + 리사이즈 (테스트)
    #################################################
    def Resize_data_test(self, start, end):
        inputs = []
        path = []
        target_h = []
        gt = []
        for i in range(start, end):
            data = self.test_data[i]
            img_path = self.p.data_root + data['raw_file']
            temp_image = cv2.imread(img_path)
            if temp_image is None:
                print(f"⚠️ 이미지 없음: {img_path}")
                continue

            temp_image = cv2.rotate(temp_image, cv2.ROTATE_90_CLOCKWISE)
            
            ratio_w = self.p.x_size * 1.0 / temp_image.shape[1]
            ratio_h = self.p.y_size * 1.0 / temp_image.shape[0]
            temp_image = cv2.resize(temp_image, (self.p.x_size, self.p.y_size))

            inputs.append(np.rollaxis(temp_image, axis=2, start=0))
            path.append(i)
            h_samples_scaled = np.array([min(int(h * ratio_h), (self.p.grid_y - 1) * self.p.resize_ratio) for h in data['h_samples']])
            h_per_lane = [h_samples_scaled for _ in data['lanes']]
            gt.append([np.array(lane) for lane in data['lanes']])
            target_h.append(h_per_lane)

        return np.array(inputs), path, ratio_w, ratio_h, target_h, gt

    #################################################
    ## 이미지 로드 + 90도 회전 + 리사이즈 (학습)
    #################################################
    def Resize_data(self, start, end, sampling_list):
        inputs = []
        target_lanes = []
        target_h = []
        test_image = []
        data_list = []

        for i in range(start, end):
            data = self.train_data[i]
            img_path = self.p.data_root + data['raw_file']
            temp_image = cv2.imread(img_path)
            if temp_image is None:
                print(f"⚠️ 이미지 없음: {img_path}")
                continue

            temp_image = cv2.rotate(temp_image, cv2.ROTATE_90_CLOCKWISE)
            
            ratio_w = self.p.x_size * 1.0 / temp_image.shape[1]
            ratio_h = self.p.y_size * 1.0 / temp_image.shape[0]
            temp_image = cv2.resize(temp_image, (self.p.x_size, self.p.y_size))

            lanes = []
            for lane in data['lanes']:
                new_lane = [min(int(x * ratio_w), (self.p.grid_x - 1) * self.p.resize_ratio) if x >= 0 else -2 for x in lane]
                lanes.append(np.array(new_lane))          # ← numpy array
            h_samples_scaled = np.array([min(int(h * ratio_h), (self.p.grid_y - 1) * self.p.resize_ratio) for h in data['h_samples']])
            h_per_lane = [h_samples_scaled for _ in data['lanes']]  # 차선 수만큼 복제

            inputs.append(np.rollaxis(temp_image, axis=2, start=0))
            target_lanes.append(lanes)
            target_h.append(h_per_lane)
            test_image.append(np.rollaxis(temp_image, axis=2, start=0))
            data_list.append(i)

        return (np.array(inputs), target_lanes, target_h,
                np.array(test_image), data_list)

    #################################################
    ## Augmentation 함수들 (원본 TuSimple 그대로)
    #################################################
    def Flip(self):
        for i in range(self.actual_batchsize):
            if random.random() < self.p.flip_ratio:
                self.inputs[i] = self.inputs[i, :, :, ::-1]
                self.test_image[i] = self.test_image[i, :, :, ::-1]
                for j in range(len(self.target_lanes[i])):
                    self.target_lanes[i][j] = np.array([
                        self.p.x_size - 1 - x if x >= 0 else x
                        for x in self.target_lanes[i][j]
                    ])

    def Translation(self):
        for i in range(self.actual_batchsize):
            if random.random() < self.p.translation_ratio:
                tx = random.randint(-50, 50)
                ty = random.randint(-20, 20)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                self.inputs[i] = np.array([
                    cv2.warpAffine(self.inputs[i][c], M,
                                (self.p.x_size, self.p.y_size))
                    for c in range(3)
                ])
                self.test_image[i] = self.inputs[i].copy()
                for j in range(len(self.target_lanes[i])):
                    self.target_lanes[i][j] = np.array([
                        min(max(x + tx, 0), self.p.x_size - 1) if x >= 0 else x  # ← 클리핑
                        for x in self.target_lanes[i][j]
                    ])
                self.target_h[i] = [
                    np.array([
                        min(max(h + ty, 0), self.p.y_size - 1)  # ← 클리핑
                        for h in lane_h
                    ])
                    for lane_h in self.target_h[i]
                ]

    def Rotate(self):
        for i in range(self.actual_batchsize):
            if random.random() < self.p.rotate_ratio:
                angle = random.uniform(-10, 10)
                cx, cy = self.p.x_size / 2, self.p.y_size / 2
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                self.inputs[i] = np.array([
                    cv2.warpAffine(self.inputs[i][c], M,
                                   (self.p.x_size, self.p.y_size))
                    for c in range(3)
                ])
                self.test_image[i] = self.inputs[i].copy()

    def Gaussian(self):
        for i in range(self.actual_batchsize):
            if random.random() < self.p.noise_ratio:
                noise = np.random.normal(0, 5, self.inputs[i].shape).astype(np.int16)
                self.inputs[i] = np.clip(self.inputs[i].astype(np.int16) + noise,
                                         0, 255).astype(np.uint8)

    def Change_intensity(self):
        for i in range(self.actual_batchsize):
            if random.random() < self.p.intensity_ratio:
                factor = random.uniform(0.7, 1.3)
                self.inputs[i] = np.clip(
                    (self.inputs[i].astype(np.float32) * factor), 0, 255
                ).astype(np.uint8)

    def Shadow(self):
        for i in range(self.actual_batchsize):
            if random.random() < self.p.shadow_ratio:
                x1 = random.randint(0, self.p.x_size)
                shadow = np.ones(self.inputs[i].shape, dtype=np.uint8) * 255
                shadow[:, :, :x1] = 128
                self.inputs[i] = np.clip(
                    self.inputs[i].astype(np.int16) - shadow + 255,
                    0, 255
                ).astype(np.uint8)

import numpy as np

class Parameters():
    n_epoch = 50
    l_rate = 0.00001
    weight_decay = 1e-5
    save_path = "savefile/"
    model_path = "savefile/"
    batch_size = 4                  # 메모리 부족하면 2로 줄이기

    # PINet 입력은 512×256이 기본이나 비율 맞게 조정
    x_size = 256                    # 너비 (1920 / 2)
    y_size = 512                    # 높이 (1080 / 2)
    resize_ratio = 8
    grid_x = x_size // resize_ratio  # 120
    grid_y = y_size // resize_ratio  # 67
    feature_size = 4
    regression_size = 110
    mode = 3
    threshold_point = 0.35
    threshold_instance = 0.08

    # loss function parameter
    K1 = 1.0
    K2 = 2.0
    constant_offset = 0.2
    constant_exist = 1.0
    constant_nonexist = 1.0
    constant_angle = 1.0
    constant_similarity = 1.0
    constant_attention = 0.1
    constant_alpha = 0.5
    constant_beta = 0.5
    constant_l = 1.0
    constant_lane_loss = 1.0
    constant_instance_loss = 1.0

    # augmentation
    flip_ratio = 0.6
    translation_ratio = 0.6
    rotate_ratio = 0.6
    noise_ratio = 0.6
    intensity_ratio = 0.6
    shadow_ratio = 0.6
    scaling_ratio = 0.2
    flip_indices = [(0,34),(1,35),(2,36),(3,37),(4,38),(5,39),(6,40),(7,41),
                    (8,42),(9,43),(10,44),(11,45),(12,46),(13,47),(14,48),(15,49),
                    (16,50),(17,51),(18,52),(19,53),(20,54),(21,55),(22,56),(23,57),
                    (24,58),(25,59),(26,60),(27,61),(28,62),(29,63),(30,64),(31,65),
                    (32,66),(33,67),(68,68),(69,69),(70,72),(71,73)]

    # ── 데이터셋 경로 ──────────────────────────────────────────
    data_root = "/home/sukja/drone_Referee/dataset/drone_lane/"
    train_json = data_root + "train_pinet.json"
    valid_json = data_root + "valid_pinet.json"
    # ──────────────────────────────────────────────────────────

    # test parameter
    color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
             (0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),
             (0,100,255),(255,0,100),(0,255,100)]
    grid_location = np.zeros((grid_y, grid_x, 2))
    for y in range(grid_y):
        for x in range(grid_x):
            grid_location[y][x][0] = x
            grid_location[y][x][1] = y
    num_iter = 30
    threshold_RANSAC = 0.1
    ratio_inliers = 0.1

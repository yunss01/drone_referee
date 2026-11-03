"""관측한 점선 도색의 제한적 기억과 바퀴별 침범/이탈 판정.

미래 프레임, 차선 연장선, 검출 순서 번호를 시간 상태의 근거로 사용하지 않는다.
"""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
from departure_checker import DepartureChecker, DepartureConfig


@dataclass(frozen=True)
class DashConfig:
    entry_hold_sec: float = 0.1
    exit_hold_sec: float = 0.067
    exit_margin_px: float = 1.5
    memory_sec: float = 0.6
    history_sec: float = 1.2
    wheel_max_gap_sec: float = 0.35
    max_frame_gap_sec: float = 0.25
    motion_max_error_px: float = 2.5
    memory_max_error_px: float = 3.0
    min_paint_area: float = 50.0
    max_half_width: float = 50.0
    support_tolerance_px: float = 1.5
    motion_width: int = 640
    legacy_contact_events: bool = False


@dataclass
class Motion:
    matrix: object = None
    valid: bool = False
    error_px: float = 0.0
    reason: str = "initial"
    points: int = 0
    elapsed_sec: object = None


def transform_points(points, matrix):
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)


def wheel_mask(shape, wheels, padding=0.06, background=False):
    mask = np.zeros(shape, np.uint8)
    height, width = shape
    for wheel in wheels:
        x1, y1, x2, y2 = wheel["bbox"]
        bw, bh = x2 - x1, y2 - y1
        px, py = max(2, int(bw * padding)), max(2, int(bh * padding))
        if background:
            # 바퀴 위 차체는 도로의 움직임 추정에서 제외한다.
            px, py = int(bw * 0.4) + 3, int(bh * 0.15) + 3
            y1 -= bh * 2.5
        a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
        c, d = min(width, int(x2 + px)), min(height, int(y2 + py))
        if c > a and d > b:
            mask[b:d, a:c] = 255
    if background:
        # 가까운 두 바퀴 사이의 차체도 움직이는 물체다. 도로 특징으로 쓰지 않는다.
        for i, first in enumerate(wheels):
            for second in wheels[i + 1:]:
                p, q = np.asarray(first["bbox"], float), np.asarray(second["bbox"], float)
                bh = max(p[3] - p[1], q[3] - q[1])
                if np.linalg.norm((p[:2] + p[2:] - q[:2] - q[2:]) / 2) < 4 * bh:
                    a, b = max(0, int(min(p[0], q[0]) - bh * 0.4)), max(0, int(min(p[1], q[1]) - bh * 2.5))
                    c, d = min(width, int(max(p[2], q[2]) + bh * 0.4)), min(height, int(max(p[3], q[3]) + bh * 0.15))
                    mask[b:d, a:c] = 255
    return mask


class RoadMotion:
    """차선 주변 도로 특징을 연속 프레임에서 추적한다. 품질 실패 시 예측 금지."""

    def __init__(self, config):
        self.config = config
        self.previous = None
        self.roi = None
        self.gap_sec = 0.0
        self.paint = None

    def reset(self):
        self.previous = self.roi = None
        self.gap_sec = 0.0
        self.paint = None

    @staticmethod
    def paint_support(previous, current, matrix, excluded):
        """정적 도색의 재투영으로 정렬을 교차 검증한다. 가려진 차체는 제외한다."""
        ys, xs = np.where(previous > 0)
        if len(xs) < 100:
            return False
        stride = max(1, len(xs) // 2000)
        points = np.column_stack([xs[::stride], ys[::stride]]).astype(np.float32)
        projected = transform_points(points, matrix)
        h, w = current.shape
        valid = ((projected[:, 0] >= 0) & (projected[:, 0] < w - 1)
                 & (projected[:, 1] >= 0) & (projected[:, 1] < h - 1))
        projected = np.rint(projected[valid]).astype(int)
        if not len(projected):
            return False
        projected = projected[excluded[projected[:, 1], projected[:, 0]] == 0]
        if len(projected) < 100:
            return False
        spread = np.ptp(projected, axis=0)
        if spread[0] < w * 0.15 or spread[1] < h * 0.06:
            return False
        distance = cv2.distanceTransform(np.uint8(current == 0), cv2.DIST_L2, 3)
        errors = distance[projected[:, 1], projected[:, 0]]
        return bool(np.mean(errors <= 2.5) >= 0.85 and np.median(errors) <= 1.5)

    def estimate(self, frame, paint, wheels, dt):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_shape = gray.shape
        radius = max(15, int(min(gray.shape) * 0.04))
        roi = cv2.dilate(paint, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
        excluded = wheel_mask(gray.shape, wheels, background=True)
        excluded_full = excluded.copy()
        # 이 시스템의 하향 도로 영상에서 하단 도로의 질감도 사용한다.
        # 점선 코너만 추적하면 반복 무늬 때문에 정렬 근거가 부족해진다.
        roi[int(gray.shape[0] * 0.55):] = 255
        roi[excluded > 0] = 0
        scale = min(1.0, self.config.motion_width / gray.shape[1])
        if scale < 1:
            size = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            roi = cv2.resize(roi, size, interpolation=cv2.INTER_NEAREST)
            excluded = cv2.resize(excluded, size, interpolation=cv2.INTER_NEAREST)
        old, old_roi = self.previous, self.roi
        if old is None or old.shape != gray.shape:
            self.previous, self.roi, self.gap_sec = gray, roi, 0.0
            self.paint = paint.copy()
            return Motion(reason="initial")
        self.gap_sec += dt
        if dt <= 0 or self.gap_sec > self.config.max_frame_gap_sec:
            self.previous, self.roi, self.gap_sec = gray, roi, 0.0
            self.paint = paint.copy()
            return Motion(reason="frame_gap")
        points = cv2.goodFeaturesToTrack(old, maxCorners=500, qualityLevel=0.005,
                                        minDistance=7, blockSize=7, mask=old_roi)
        if points is None or len(points) < 20:
            return Motion(reason="few_road_features")
        params = dict(winSize=(31, 31), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(old, gray, points, None, **params)
        if nxt is None:
            return Motion(reason="optical_flow_failed")
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, old, nxt, None, **params)
        if back is None:
            return Motion(reason="backward_flow_failed")
        a, b = points[:, 0], nxt[:, 0]
        good = ((status[:, 0] > 0) & (status_back[:, 0] > 0)
                & (np.linalg.norm(back[:, 0] - a, axis=1) < 0.75)
                & (b[:, 0] >= 0) & (b[:, 0] < gray.shape[1])
                & (b[:, 1] >= 0) & (b[:, 1] < gray.shape[0]))
        a, b = a[good], b[good]
        if len(a):
            indices = np.rint(b).astype(int)
            indices[:, 0] = np.clip(indices[:, 0], 0, gray.shape[1] - 1)
            indices[:, 1] = np.clip(indices[:, 1], 0, gray.shape[0] - 1)
            keep = excluded[indices[:, 1], indices[:, 0]] == 0
            a, b = a[keep], b[keep]
        if len(a) < 20:
            return Motion(reason="few_tracked_features", points=len(a))
        a, b = a / scale, b / scale
        # 적합에 쓰지 않은 대응점으로 과적합과 반복 점선 오매칭을 검사한다.
        train = np.arange(len(a)) % 3 != 0
        test = ~train
        candidates = []
        affine, _ = cv2.estimateAffinePartial2D(a[train], b[train],
                                              method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if affine is not None:
            candidates.append(("affine", np.vstack([affine, [0, 0, 1]])))
        homography, _ = cv2.findHomography(a[train], b[train], cv2.RANSAC, 1.5)
        if homography is not None:
            candidates.append(("homography", homography))
        accepted = []
        for name, matrix in candidates:
            if not np.isfinite(matrix).all():
                continue
            error = np.linalg.norm(transform_points(a[test], matrix) - b[test], axis=1)
            p80 = float(np.percentile(error, 80))
            inliers = error <= self.config.motion_max_error_px
            strong = p80 <= self.config.motion_max_error_px and inliers.mean() >= 0.8
            if not strong and (inliers.mean() < 0.5 or np.median(error) > self.config.motion_max_error_px):
                continue
            spread = np.ptp(a[test][inliers], axis=0)
            if min(spread) < min(full_shape) * 0.06:
                continue
            h, w = full_shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            warped = transform_points(corners, matrix)
            area_ratio = abs(cv2.contourArea(warped)) / (w * h)
            if not 0.65 < area_ratio < 1.5:
                continue
            if np.max(np.linalg.norm(warped - corners, axis=1)) > math.hypot(w, h) * 0.2:
                continue
            if not strong:
                # 반복 점선 한 주기만큼 잘못 맞춘 변환을 보수적인 회전 속도 한계로 제외한다.
                if np.max(np.linalg.norm(warped - corners, axis=1)) > 1200 * self.gap_sec + 5:
                    continue
                if self.paint is None or not self.paint_support(self.paint, paint, matrix, excluded_full):
                    continue
            score = min(p80, self.config.motion_max_error_px) + (0.2 if name == "homography" else 0) + (0 if strong else 1)
            quality_error = float(np.median(error[inliers])) + (0 if strong else 0.5)
            accepted.append((score, name if strong else name + "_paint_validated", matrix, quality_error))
        if not accepted:
            return Motion(reason="registration_rejected", points=len(a))
        _, name, matrix, error = min(accepted, key=lambda item: item[0])
        elapsed = self.gap_sec
        self.previous, self.roi, self.gap_sec = gray, roi, 0.0
        self.paint = paint.copy()
        return Motion(matrix, True, max(0.15, error), name, len(a), elapsed)


def paint_geometry(mask, labels, ids, max_half_width, restored=None, render_mask=True):
    """연결되지 않은 실제 도색 조각 사이를 PCA로 가로지르지 않는다."""
    result = []
    for lane_id in ids:
        binary = np.uint8((labels == lane_id) & (mask > 0)) * 255
        x0, y0, width, height = cv2.boundingRect(binary)
        if not width or not height:
            continue
        roi = binary[y0:y0 + height, x0:x0 + width]
        count, cc, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
        for j in range(1, count):
            if stats[j, cv2.CC_STAT_AREA] < 12:
                continue
            local_part = np.uint8(cc == j) * 255
            contours, hierarchy = cv2.findContours(local_part, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            outer = [c for i, c in enumerate(contours) if hierarchy[0, i, 3] < 0]
            half_width = min(max(min(cv2.minAreaRect(c)[1]) / 2 for c in outer),
                             max_half_width)
            ys, xs = np.where(local_part > 0)
            xs, ys = xs + x0, ys + y0
            points = np.column_stack([xs, ys]).astype(np.float32)
            mean, vectors = cv2.PCACompute(points, mean=None)
            projection = (points - mean[0]) @ vectors[0]
            a = mean[0] + vectors[0] * projection.min()
            b = mean[0] + vectors[0] * projection.max()
            if half_width <= 0:
                continue
            part = None
            if render_mask:
                part = np.zeros_like(binary)
                part[y0:y0 + height, x0:x0 + width] = local_part
            contours = [c + np.array([[[x0, y0]]], np.int32) for c in outer]
            memory = (restored[y0:y0 + height, x0:x0 + width] if restored is not None else None)
            result.append(dict(lane_id=int(lane_id), mask_bin=part, contours=contours,
                               centerline=(tuple(a), tuple(b)), half_width=float(half_width),
                               has_holes=any(h[3] >= 0 for h in hierarchy[0]),
                               restored=bool(memory is not None and np.any(memory[local_part > 0]))))
    return result


def margin(point, lane):
    a, b = np.asarray(lane["centerline"], float)
    delta = b - a
    length_sq = float(delta @ delta)
    t = float(np.clip((np.asarray(point) - a) @ delta / length_sq, 0, 1)) if length_sq else 0
    return float(np.linalg.norm(np.asarray(point) - (a + t * delta)) - lane["half_width"])


def supports_point(point, lane, tolerance):
    """PCA 선분만으로 관측하지 않은 구멍이나 점선 공백을 채우지 않는다."""
    # 가느다란 도색의 끝은 기존과 같은 유한 선분 + 반폭으로 평가한다.
    # 실제로 구멍이 있는 마스크는 픽셀 근거를 추가 확인한다.
    if not lane.get("has_holes", False):
        return True
    x, y = point
    r = int(math.ceil(tolerance)) + 1
    h, w = lane["mask_bin"].shape
    ax, ay = max(0, int(x) - r), max(0, int(y) - r)
    bx, by = min(w, int(x) + r + 2), min(h, int(y) + r + 2)
    if bx <= ax or by <= ay:
        return False
    ys, xs = np.where(lane["mask_bin"][ay:by, ax:bx] > 0)
    return bool(len(xs) and np.min((xs + ax - x) ** 2 + (ys + ay - y) ** 2) <= tolerance ** 2)


class PaintMemory:
    """도색을 실제 관측한 픽셀만 기억한다. 복원은 바퀴 가림 영역으로 제한한다."""

    def __init__(self, config):
        self.config = config
        self.labels = self.age = self.variance = None
        self.next_id = 1
        self.pending_sec = 0.0
        self.current_observed_labels = None

    def reset(self):
        self.labels = self.age = self.variance = None
        self.pending_sec = 0.0
        self.current_observed_labels = None

    def observed_mask(self, lanes, shape):
        mask = np.zeros(shape, np.uint8)
        for lane in lanes:
            # 원본 mask에 남은 작은 조각은 기존 면적 필터를 통과한 contour로 제한한다.
            accepted = np.zeros(shape, np.uint8)
            for contour in lane["contours"]:
                if cv2.contourArea(contour) >= self.config.min_paint_area:
                    cv2.drawContours(accepted, [contour], -1, 255, cv2.FILLED)
            mask |= accepted & lane["mask_bin"]
        return mask

    def update(self, observed, wheels, motion, dt):
        shape = observed.shape
        h, w = shape
        frozen = (self.labels is not None and self.labels.shape == shape and not motion.valid
                  and motion.reason not in ("initial", "frame_gap"))
        if self.labels is not None and self.labels.shape == shape and motion.valid:
            labels = cv2.warpPerspective(self.labels.astype(np.float32), motion.matrix,
                                         (w, h), flags=cv2.INTER_NEAREST).astype(np.int32)
            age = cv2.warpPerspective(self.age, motion.matrix, (w, h), flags=cv2.INTER_NEAREST) + dt + self.pending_sec
            variance = cv2.warpPerspective(self.variance, motion.matrix, (w, h),
                                           flags=cv2.INTER_NEAREST) + motion.error_px ** 2
            previous_labels = labels.copy()
            occluders = wheel_mask(shape, wheels)
            keep = (occluders > 0) & (age <= self.config.history_sec)
            labels[~keep] = 0
        else:
            labels = np.zeros(shape, np.int32)
            age = np.zeros(shape, np.float32)
            variance = np.zeros(shape, np.float32)
            previous_labels = labels.copy()
        # 별도 조각도 과거의 동일 도색 픽셀과 겹칠 때에만 같은 ID를 계승한다.
        count, components, stats, _ = cv2.connectedComponentsWithStats(observed, 8)
        assigned = np.zeros(shape, np.int32)
        for j in range(1, count):
            pixels = components == j
            overlap = previous_labels[pixels]
            overlap = overlap[overlap > 0]
            lane_id = None
            if len(overlap):
                unique, counts = np.unique(overlap, return_counts=True)
                best = int(np.argmax(counts))
                if counts[best] >= max(5, stats[j, cv2.CC_STAT_AREA] * 0.1):
                    lane_id = int(unique[best])
            if lane_id is None:
                lane_id = self.next_id
                self.next_id += 1
            assigned[pixels] = lane_id
        current = observed > 0
        labels[current] = assigned[current]
        age[current] = 0
        variance[current] = 0
        self.current_observed_labels = assigned
        if frozen:
            # 정렬 실패 프레임의 관측은 표시만 한다. 마지막 유효 좌표계는 보존한다.
            self.pending_sec += dt
        else:
            self.labels, self.age, self.variance = labels, age, variance
            self.pending_sec = 0.0
        history = labels > 0
        reliable = history & (age <= self.config.memory_sec) & (variance <= self.config.memory_max_error_px ** 2)
        restored = reliable & ~current
        ids = np.unique(labels[history])
        known_lanes = paint_geometry(np.uint8(reliable), labels, ids,
                                     self.config.max_half_width, restored)
        history_lanes = (known_lanes if np.array_equal(reliable, history) else
                         paint_geometry(np.uint8(history), labels, ids, self.config.max_half_width,
                                        render_mask=False))
        return known_lanes, history_lanes, int(restored.sum())


class WheelAssociation:
    def __init__(self, config):
        self.config = config
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()

    def update(self, wheels, timestamp, motion):
        expired = [key for key, value in self.tracks.items()
                   if timestamp - value["time"] > self.config.wheel_max_gap_sec]
        for key in expired:
            del self.tracks[key]
        boxes = [np.asarray(w["bbox"], float) for w in wheels]
        edges, projected_centers = [], {}
        for key, track in self.tracks.items():
            box = track["bbox"].copy()
            if motion.valid:
                corners = transform_points([[box[0], box[1]], [box[2], box[3]]], motion.matrix)
                box = np.r_[corners.min(axis=0), corners.max(axis=0)]
            dt = timestamp - track.get("step_time", track["time"])
            projected_centers[key] = ((box[:2] + box[2:]) / 2, dt)
            box += np.tile(track["velocity"] * min(dt, 0.15), 2)
            track["bbox"], track["step_time"] = box.copy(), timestamp
            center = (box[:2] + box[2:]) / 2
            size = np.maximum(box[2:] - box[:2], 1)
            for index, candidate in enumerate(boxes):
                candidate_size = np.maximum(candidate[2:] - candidate[:2], 1)
                ratio = candidate_size / size
                distance = np.linalg.norm(((candidate[:2] + candidate[2:]) / 2 - center) / size)
                if distance < 0.9 and np.all((ratio > 0.5) & (ratio < 2)):
                    edges.append((float(distance + np.abs(np.log(ratio)).sum() * 0.2), key, index))
        used_tracks, used_indices, mapping = set(), set(), {}
        for _, key, index in sorted(edges):
            if key not in used_tracks and index not in used_indices:
                mapping[index] = key
                used_tracks.add(key)
                used_indices.add(index)
        result = []
        for index, wheel in enumerate(wheels):
            key = mapping.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
            box = boxes[index]
            velocity = np.zeros(2)
            if key in self.tracks:
                previous = self.tracks[key]
                center, dt = projected_centers[key]
                if dt > 0:
                    velocity = ((box[:2] + box[2:]) / 2 - center) / dt
            self.tracks[key] = dict(bbox=box, time=timestamp, step_time=timestamp, velocity=velocity)
            result.append({**wheel, "track_id": key, "detection_index": index})
        return result, expired


class ContactEpisode:
    """이전 결과 재현용 접촉 시작/종료 상태. 이탈 판정에는 사용하지 않는다."""

    def __init__(self, config):
        self.config = config
        self.phase = "outside"
        self.lane_id = None
        self.entry_time = self.exit_time = None
        self.active = False
        self.contact_open = False
        self.last_time = None
        self.unknown_since = None
        self.unresolved = False

    def update(self, evidence, timestamp, lane_id=None, source="observed"):
        events = []
        if self.last_time is not None and timestamp - self.last_time > self.config.max_frame_gap_sec:
            evidence = "unknown"
        self.last_time = timestamp
        if evidence != "unknown":
            self.unknown_since = None
        if evidence == "inside":
            self.unresolved = False
            if not self.contact_open:
                self.contact_open = True
                events.append(dict(event="contact_started", time_sec=timestamp))
            if self.active:
                if self.phase == "unknown":
                    events.append(dict(event="reacquired", time_sec=timestamp))
                    self.entry_time = timestamp
                if self.phase in ("unknown", "recovering"):
                    self.phase = ("inside" if self.entry_time is not None
                                  and timestamp - self.entry_time + 1e-9 >= self.config.entry_hold_sec
                                  else "recovering")
                else:
                    self.phase = "inside"
            else:
                if self.phase != "entering" or self.lane_id != lane_id or self.entry_time is None:
                    self.entry_time = timestamp
                self.phase = "entering"
                if timestamp - self.entry_time + 1e-9 >= self.config.entry_hold_sec:
                    self.phase, self.active = "inside", True
                    events.append(dict(event="entered", time_sec=timestamp,
                                       contact_started_sec=self.entry_time))
            self.lane_id = lane_id
            self.exit_time = None
        elif evidence == "outside":
            if self.contact_open:
                if self.phase != "exiting":
                    self.exit_time = timestamp
                self.phase = "exiting"
                if timestamp - self.exit_time + 1e-9 >= self.config.exit_hold_sec:
                    events.append(dict(event="contact_ended", time_sec=timestamp,
                                       exit_started_sec=self.exit_time, confirmed_contact=self.active))
                    if self.active:
                        events.append(dict(event="exited", time_sec=timestamp,
                                           exit_started_sec=self.exit_time))
                    self.phase, self.active = "outside", False
                    self.contact_open = False
                    self.unresolved = False
                    self.entry_time = self.exit_time = None
            else:
                self.phase, self.entry_time = "outside", None
        elif evidence == "boundary" and self.contact_open:
            self.phase, self.exit_time = ("inside" if self.active else "entering"), None
            if not self.active:
                self.entry_time = None
        else:
            self.phase = "unknown"
            self.exit_time = None
            self.entry_time = None
            if self.unknown_since is None:
                self.unknown_since = timestamp
            if self.contact_open and timestamp - self.unknown_since > self.config.history_sec:
                events.append(dict(event="lost", time_sec=timestamp, reason="paint_unobservable",
                                   confirmed_contact=self.active))
                self.active = False
                self.contact_open = False
                self.unresolved = True
        for event in events:
            event["lane_id"] = self.lane_id
        confirmed = self.active and self.phase in ("inside", "exiting") and self.entry_time is not None
        return dict(violated=evidence == "inside", confirmed=confirmed,
                    duration=max(0.0, timestamp - self.entry_time) if confirmed and self.entry_time is not None else 0.0,
                    phase=self.phase, evidence=evidence, lane_id=self.lane_id,
                    contact_source=source, exited=any(e["event"] == "contact_ended" for e in events),
                    events=events)


class DashReferee:
    def __init__(self, config=None):
        self.config = config or DashConfig()
        self.motion = RoadMotion(self.config)
        self.paint = PaintMemory(self.config)
        self.wheels = WheelAssociation(self.config)
        self.episodes = {}
        self.intrusion_starts = {}
        self.intrusion_signaled = set()
        self.departure = DepartureChecker(DepartureConfig(
            side_margin_px=self.config.exit_margin_px,
            max_model_age_sec=self.config.history_sec,
            max_motion_error_px=self.config.memory_max_error_px + 1.0,
            max_wheel_gap_sec=self.config.wheel_max_gap_sec))
        self.timestamp = None
        self.last_result = None

    def reset(self):
        self.motion.reset()
        self.paint.reset()
        self.reset_contacts()
        self.departure.reset()
        self.timestamp = None

    def reset_contacts(self):
        self.wheels.reset()
        self.episodes.clear()
        self.intrusion_starts.clear()
        self.intrusion_signaled.clear()
        self.departure.reset_wheels()

    def _binary_result(self, frame, known, tracked, motion, now, recovered_pixels, paused):
        if paused:
            self.reset_contacts()
        events = self.departure.update(known, [] if paused else tracked, motion, now,
                                       observed_labels=self.paint.current_observed_labels)
        visible = {wheel["track_id"] for wheel in tracked}
        for key in set(self.intrusion_starts) - visible:
            self.intrusion_starts.pop(key, None)
            self.intrusion_signaled.discard(key)
        states = []
        departure_keys = {event["wheel_id"] for event in events}
        for wheel in tracked:
            key = wheel["track_id"]
            hits = [(margin(wheel["keypoint"], lane), lane) for lane in known
                    if margin(wheel["keypoint"], lane) < 0
                    and supports_point(wheel["keypoint"], lane, self.config.support_tolerance_px)]
            hits.sort(key=lambda item: item[0])
            raw = bool(hits) and not paused
            if raw:
                self.intrusion_starts.setdefault(key, now)
            else:
                self.intrusion_starts.pop(key, None)
                self.intrusion_signaled.discard(key)
            duration = now - self.intrusion_starts[key] if raw else 0.0
            intrusion = raw and duration + 1e-9 >= self.config.entry_hold_sec
            lane_id = hits[0][1]["lane_id"] if hits else None
            if intrusion and key not in self.intrusion_signaled:
                events.append(dict(event="intrusion", time_sec=now, wheel_id=key, lane_id=lane_id,
                                   contact_started_sec=self.intrusion_starts[key]))
                self.intrusion_signaled.add(key)
            departed = self.departure.departed.get(key)
            active_out = self.departure.active.get(key)
            states.append(dict(wheel_id=key, detection_index=wheel["detection_index"],
                               violated=raw, confirmed=intrusion or active_out is not None,
                               intrusion_confirmed=intrusion, duration=duration,
                               departure_detected=key in departure_keys,
                               departure_active=active_out is not None,
                               departure_latched=departed is not None,
                               departure_recorded=departed is not None,
                               departure_time_sec=departed["time_sec"] if departed else None,
                               departure_anchor_id=active_out["paint_anchor_id"] if active_out else None,
                               evidence="line_out" if active_out else "paint" if raw else "clear" if known else "not_observed",
                               assessable=bool(known) or active_out is not None, lane_id=lane_id, hit_lane_id=lane_id,
                               contact_source="memory_assisted" if hits and hits[0][1]["restored"] else "observed"))
        result = dict(lanes=known, wheels=tracked, states=states, events=events,
                      motion_valid=motion.valid, motion_reason=motion.reason,
                      motion_error_px=motion.error_px, recovered_pixels=recovered_pixels,
                      departure_diagnostics=self.departure.trace)
        self.last_result = result
        return result

    def update(self, frame, lanes, wheels, timestamp_sec=None, paused=False, motion_override=None):
        now = time.monotonic() if timestamp_sec is None else float(timestamp_sec)
        if not math.isfinite(now):
            raise ValueError("영상 시각이 유효하지 않습니다.")
        if self.timestamp is not None and now <= self.timestamp:
            self.reset()
        dt = now - self.timestamp if self.timestamp is not None else 0.0
        self.timestamp = now
        if dt > self.config.max_frame_gap_sec:
            self.intrusion_starts.clear()
            self.intrusion_signaled.clear()
        observed = self.paint.observed_mask(lanes, frame.shape[:2])
        estimated = self.motion.estimate(frame, observed, wheels, dt)
        motion = estimated if motion_override is None else motion_override
        wheel_motion = motion
        if motion.elapsed_sec is not None and motion.elapsed_sec > dt * 1.5:
            # 기억 복구용 H는 마지막 유효 프레임 기준이므로 직전 바퀴 좌표에 적용하지 않는다.
            wheel_motion = Motion(reason="reference_recovery")
        tracked, expired = self.wheels.update(wheels, now, wheel_motion)
        events = []
        for key in expired:
            episode = self.episodes.pop(key, None)
            if episode is not None and episode.contact_open:
                events.append(dict(event="lost", time_sec=now, wheel_id=key,
                                   lane_id=episode.lane_id, reason="wheel_missing",
                                   confirmed_contact=episode.active))
        known, history, recovered_pixels = self.paint.update(observed, tracked, motion, dt)
        if not self.config.legacy_contact_events:
            return self._binary_result(frame, known, tracked, motion, now, recovered_pixels, paused)
        states = []
        if paused:
            self.reset_contacts()
        for wheel in tracked:
            key, point = wheel["track_id"], wheel["keypoint"]
            episode = self.episodes.setdefault(key, ContactEpisode(self.config))
            hits = [(margin(point, lane), lane) for lane in known
                    if margin(point, lane) < 0 and supports_point(point, lane, self.config.support_tolerance_px)]
            hits.sort(key=lambda item: (item[1]["lane_id"] != episode.lane_id, item[0]))
            evidence, source, target = "outside", "observed", episode.lane_id
            if hits:
                chosen = hits[0][1]
                target = chosen["lane_id"]
                source = "memory_assisted" if chosen["restored"] else "observed"
                evidence = "inside"
                if episode.contact_open and target != episode.lane_id:
                    previous = [lane for lane in history if lane["lane_id"] == episode.lane_id]
                    if previous:
                        evidence = ("outside" if min(margin(point, lane) for lane in previous)
                                    > self.config.exit_margin_px else "unknown")
                        target = episode.lane_id
                    elif episode.phase != "unknown" or not motion.valid:
                        evidence, target = "unknown", episode.lane_id
            elif episode.contact_open:
                old = [lane for lane in history if lane["lane_id"] == episode.lane_id]
                safe = [lane for lane in known if lane["lane_id"] == episode.lane_id]
                if not old or not motion.valid:
                    evidence = "unknown"
                else:
                    old_margin = min(margin(point, lane) for lane in old)
                    safe_margin = min((margin(point, lane) for lane in safe), default=math.inf)
                    if old_margin < 0 and safe_margin >= 0:
                        evidence = "unknown"
                    elif old_margin <= self.config.exit_margin_px:
                        evidence = "boundary"
                    else:
                        evidence = "outside"
            elif not known:
                evidence = "unknown"
            elif episode.unresolved:
                evidence = "unknown"
            elif any(margin(point, lane) < 0 for lane in history):
                # 미확정 상태에서도 만료된 도색 기억을 정상의 근거로 삼지 않는다.
                evidence = "unknown"
            if paused:
                evidence = "unknown"
            state = episode.update(evidence, now, target, source)
            # 사건 상태는 이전 조각의 이탈을 확인하는 중일 수 있다.
            # 현재 다른 도색 조각 위에 있는 원시 관측까지 False로 바꾸면 안 된다.
            state["violated"] = bool(hits) and not paused
            state["hit_lane_id"] = hits[0][1]["lane_id"] if hits else None
            state.update(wheel_id=key, detection_index=wheel["detection_index"])
            for event in state["events"]:
                event["wheel_id"] = key
                events.append(event)
            states.append(state)
        # 잠시 미검출된 바퀴도 침범 지속 시간을 새로 누적하지 않는다.
        visible_ids = {wheel["track_id"] for wheel in tracked}
        for key, episode in self.episodes.items():
            if key not in visible_ids:
                episode.update("unknown", now)
        result = dict(lanes=known, wheels=tracked, states=states, events=events,
                      motion_valid=motion.valid, motion_reason=motion.reason,
                      motion_error_px=motion.error_px, recovered_pixels=recovered_pixels)
        self.last_result = result
        return result


def process_temporal(state, frame, lanes, wheels, hold_sec, timestamp_sec=None, paused=False):
    """두 운영 진입점이 같은 시간 상태와 사건 형식을 사용한다."""
    engine = getattr(state, "dash_referee", None)
    if engine is None:
        engine = state.dash_referee = DashReferee(DashConfig(entry_hold_sec=hold_sec))
    if getattr(state, "contact_reset_requested", False):
        engine.reset_contacts()
        state.contact_reset_requested = False
    result = engine.update(frame, lanes, wheels, timestamp_sec, paused)
    for event in result["events"]:
        print(f"[점선] {event['event']} wheel={event['wheel_id']} "
              f"lane={event['lane_id']} time={event['time_sec']:.3f}s")
    return result

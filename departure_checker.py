"""실제 점선의 같은 x열에서 접지점이 화면 아래쪽 경계를 넘었는지 검사한다.

운영 기준은 화면 아래쪽만 이탈이다. 점선 공백과 끝부분은 연장하지 않으며,
과거 사건 기록은 현재 이탈 상태와 별도로 보관한다.
"""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2),
                                    matrix).reshape(-1, 2)


@dataclass(frozen=True)
class DepartureConfig:
    side_margin_px: float = 1.5
    max_model_age_sec: float = 1.2
    max_motion_error_px: float = 4.0
    max_wheel_gap_sec: float = 0.35
    decision_grace_sec: float = 0.8
    wheel_padding_px: float = 2.0
    local_min_columns: int = 20
    local_min_span_fraction: float = 0.2
    local_max_shift_px: float = 4.0
    local_max_residual_px: float = 1.5
    local_error_floor_px: float = 0.75
    local_recent_sec: float = 0.2


@dataclass
class PaintAnchor:
    identity: int
    lane_ids: set
    a: object
    b: object
    origin: object
    mask: object
    samples: object
    image_shape: tuple
    full_time: float
    matrix: object = field(default_factory=lambda: np.eye(3))
    variance: float = 0.0
    permits: dict = field(default_factory=dict)
    edge_variance: object = None
    verified_sec: float = -math.inf
    verified_columns: int = 0
    vertical_correction_px: float = 0.0

    @classmethod
    def from_lane(cls, identity, lane, now):
        x, y, w, h = cv2.boundingRect(lane["mask_bin"])
        local = lane["mask_bin"][y:y + h, x:x + w].copy()
        ys, xs = np.where(local > 0)
        stride = max(1, len(xs) // 200)
        samples = np.column_stack([xs[::stride] + x, ys[::stride] + y]).astype(np.float32)
        return cls(identity, {lane["lane_id"]}, np.asarray(lane["centerline"][0], float),
                   np.asarray(lane["centerline"][1], float), np.array([x, y]), local, samples,
                   lane["mask_bin"].shape, now)

    def contains(self, points):
        local = np.rint(np.asarray(points) - self.origin).astype(int)
        h, w = self.mask.shape
        valid = ((local[:, 0] >= 0) & (local[:, 0] < w)
                 & (local[:, 1] >= 0) & (local[:, 1] < h))
        result = np.zeros(len(local), bool)
        result[valid] = self.mask[local[valid, 1], local[valid, 0]] > 0
        return result

    def projected_mask(self):
        """현재 화면으로 실제 관측 픽셀만 옮긴다. 전체 화면 대신 작은 ROI를 사용한다."""
        h, w = self.mask.shape
        corners = project(self.origin + np.array([[0, 0], [w - 1, 0],
                                                  [w - 1, h - 1], [0, h - 1]]), self.matrix)
        if not np.isfinite(corners).all():
            return None
        height, width = self.image_shape
        lo = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 1, 0)
        hi = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 2, [width, height])
        if np.any(hi <= lo):
            return None
        original = np.eye(3)
        original[:2, 2] = self.origin
        offset = np.eye(3)
        offset[:2, 2] = -lo
        warped = cv2.warpPerspective(self.mask, offset @ self.matrix @ original,
                                     tuple((hi - lo).tolist()), flags=cv2.INTER_NEAREST)
        nonzero = warped > 0
        columns = nonzero.any(axis=0)
        tops = np.argmax(nonzero, axis=0) + lo[1]
        bottoms = warped.shape[0] - 1 - np.argmax(nonzero[::-1], axis=0) + lo[1]
        return warped, lo, columns, tops, bottoms


class DepartureChecker:
    def __init__(self, config=None):
        self.config = config or DepartureConfig()
        self.anchors = {}
        self.next_id = 1
        self.departed = {}  # 사건 로그용. 현재 bbox 색상에 사용하지 않는다.
        self.active = {}
        self.last_out = {}
        self.last_anchor = {}
        self.last_seen = {}
        self.last_events = []
        self.trace = []

    def reset(self):
        self.anchors.clear()
        self.reset_wheels()

    def reset_wheels(self):
        self.departed.clear()
        self.active.clear()
        self.last_out.clear()
        self.last_anchor.clear()
        self.last_seen.clear()
        self.last_events = []
        self.trace = []
        for anchor in self.anchors.values():
            anchor.permits.clear()

    def _coverage(self, anchor, lane, points):
        inside_old = float(anchor.contains(project(points, np.linalg.inv(anchor.matrix))).mean())
        projected = np.rint(project(anchor.samples, anchor.matrix)).astype(int)
        h, w = lane["mask_bin"].shape
        valid = ((projected[:, 0] >= 0) & (projected[:, 0] < w)
                 & (projected[:, 1] >= 0) & (projected[:, 1] < h))
        inside_new = np.zeros(len(projected), bool)
        inside_new[valid] = lane["mask_bin"][projected[valid, 1], projected[valid, 0]] > 0
        return inside_old, float(inside_new.mean())

    @staticmethod
    def _under_wheel(anchor, wheels):
        points = project(anchor.samples, anchor.matrix)
        for wheel in wheels:
            x1, y1, x2, y2 = wheel["bbox"]
            if np.any((points[:, 0] >= x1 - 2) & (points[:, 0] <= x2 + 2)
                      & (points[:, 1] >= y1 - 2) & (points[:, 1] <= y2 + 2)):
                return True
        return False

    def _update_anchors(self, lanes, wheels, motion, now):
        if not motion.valid:
            if motion.reason in ("initial", "frame_gap"):
                self.anchors.clear()
                for lane in lanes:
                    if not lane.get("restored", False):
                        identity = self.next_id
                        self.next_id += 1
                        self.anchors[identity] = PaintAnchor.from_lane(identity, lane, now)
                return list(self.anchors.values())
            # 현재 직접 보이는 도색은 현재 좌표에서 판단할 수 있다.
            # 정렬 실패 중의 관측으로 마지막 유효 프레임의 기억을 갱신하지 않는다.
            return [PaintAnchor.from_lane(-i - 1, lane, now) for i, lane in enumerate(lanes)
                    if not lane.get("restored", False)]
        for key, anchor in list(self.anchors.items()):
            anchor.matrix = motion.matrix @ anchor.matrix
            anchor.matrix /= anchor.matrix[2, 2]
            anchor.variance += motion.error_px ** 2
            if anchor.edge_variance is not None:
                anchor.edge_variance += motion.error_px ** 2
            permitted = any(now <= until for until in anchor.permits.values())
            if ((now - anchor.full_time > self.config.max_model_age_sec and not permitted)
                    or anchor.variance > self.config.max_motion_error_px ** 2):
                del self.anchors[key]
        for lane in lanes:
            ys, xs = np.where(lane["mask_bin"] > 0)
            stride = max(1, len(xs) // 200)
            points = np.column_stack([xs[::stride], ys[::stride]]).astype(np.float32)
            candidates = []
            for anchor in self.anchors.values():
                old, new = self._coverage(anchor, lane, points)
                if max(old, new) >= 0.6:
                    candidates.append((max(old, new), new, anchor))
            if candidates:
                _, new, anchor = max(candidates, key=lambda item: item[0])
                anchor.lane_ids.add(lane["lane_id"])
                if (not lane.get("restored", False) and new >= 0.8
                        and not self._under_wheel(anchor, wheels)):
                    updated = PaintAnchor.from_lane(anchor.identity, lane, now)
                    updated.lane_ids |= anchor.lane_ids
                    self.anchors[anchor.identity] = updated
            elif not lane.get("restored", False):
                identity = self.next_id
                self.next_id += 1
                self.anchors[identity] = PaintAnchor.from_lane(identity, lane, now)
        return list(self.anchors.values())

    def _validate_visible_paint(self, anchor, wheels, observed_labels, now):
        """가림 밖의 현재 관측만으로 세로 위치를 검증한다. 모델 수명·가로 끝은 유지한다."""
        prepared = anchor.projected_mask()
        if prepared is None or observed_labels is None:
            return
        mask, origin, columns, tops, bottoms = prepared
        height, width = observed_labels.shape
        pad = int(math.ceil(self.config.local_max_shift_px)) + 1
        y0, y1 = max(0, int(origin[1]) - pad), min(height, int(origin[1]) + mask.shape[0] + pad)
        x0, x1 = int(origin[0]), int(origin[0]) + mask.shape[1]
        actual = np.isin(observed_labels[y0:y1, x0:x1], tuple(anchor.lane_ids))
        if not actual.size:
            return
        visible = columns & actual.any(axis=0)
        xs = np.arange(x0, x1)
        # 타이어 안쪽의 미검출이나 보강 픽셀로 자기 예측을 검증하지 않는다.
        for wheel in wheels:
            left, _, right, _ = wheel["bbox"]
            visible &= (xs < left - self.config.wheel_padding_px) | (xs > right + self.config.wheel_padding_px)
        observed_top = np.argmax(actual, axis=0) + y0
        observed_bottom = actual.shape[0] - 1 - np.argmax(actual[::-1], axis=0) + y0
        ratios = (observed_bottom - observed_top + 1) / np.maximum(bottoms - tops + 1, 1)
        # 한쪽 경계만 남은 조각을 완전한 도색 폭으로 오인하지 않는다.
        visible &= (ratios >= 0.7) & (ratios <= 1.4)
        indices = np.flatnonzero(visible)
        if (len(indices) < self.config.local_min_columns
                or np.ptp(indices) < max(self.config.local_min_columns,
                                         mask.shape[1] * self.config.local_min_span_fraction)):
            return
        offsets = ((observed_top + observed_bottom - tops - bottoms) / 2.0)[indices]
        correction = float(np.median(offsets[::2]))
        held_out = offsets[1::2]
        residual = float(np.quantile(np.abs(held_out - correction), 0.9))
        if (abs(correction) > self.config.local_max_shift_px
                or residual > self.config.local_max_residual_px):
            return
        correction_matrix = np.eye(3)
        correction_matrix[1, 2] = correction
        anchor.matrix = correction_matrix @ anchor.matrix
        # 전체 정렬 오차는 그대로 둔다. 관측으로 검증한 세로 경계 오차만 별도 관리한다.
        anchor.edge_variance = max(self.config.local_error_floor_px, residual) ** 2
        anchor.verified_sec = now
        anchor.verified_columns = len(indices)
        anchor.vertical_correction_px = correction

    def update(self, lanes, wheels, motion, now, observed_labels=None):
        self.last_events, self.trace, self.active = [], [], {}
        anchors = self._update_anchors(lanes, wheels, motion, now)
        if motion.valid:
            for anchor in anchors:
                self._validate_visible_paint(anchor, wheels, observed_labels, now)
        prepared = [(anchor, anchor.projected_mask(), project([anchor.a, anchor.b], anchor.matrix))
                    for anchor in anchors]
        for wheel in wheels:
            key = wheel["track_id"]
            if now - self.last_seen.get(key, now) > self.config.max_wheel_gap_sec:
                self.last_out.pop(key, None)
                self.last_anchor.pop(key, None)
            self.last_seen[key] = now
            x, y = map(float, wheel["keypoint"])
            x1, y1, x2, y2 = wheel["bbox"]
            candidates = []
            for anchor, prepared_mask, axis in prepared:
                if prepared_mask is None:
                    continue
                mask, origin, columns, tops, bottoms = prepared_mask
                column = int(round(x)) - origin[0]
                direction = axis[1] - axis[0]
                length = float(np.linalg.norm(direction))
                along = float((np.array([x, y]) - axis[0]) @ direction / length) if length >= 1 else -1.0
                finite = bool(0 <= along <= length and 0 <= column < len(columns) and columns[column])
                top = float(tops[column]) if finite else None
                bottom = float(bottoms[column]) if finite else None
                near = bool(finite and bottom >= y1 - self.config.wheel_padding_px
                            and top <= y2 + self.config.wheel_padding_px)
                fresh = now - anchor.full_time <= self.config.max_model_age_sec
                if fresh and near:
                    # 실제 모델 관측 시각이 기준이다. 예측 반복으로 만료를 연장하지 않는다.
                    anchor.permits[key] = (anchor.full_time + self.config.max_model_age_sec
                                           + self.config.decision_grace_sec)
                permitted = fresh or now <= anchor.permits.get(key, -math.inf)
                error = math.sqrt(anchor.variance)
                edge_error = min(error, math.sqrt(anchor.edge_variance)) if anchor.edge_variance is not None else error
                required = self.config.side_margin_px + edge_error
                recent = now - anchor.verified_sec <= self.config.local_recent_sec
                keep_out = bool(recent and edge_error <= self.config.local_max_residual_px
                                and self.last_out.get(key, False)
                                and self.last_anchor.get(key) == anchor.identity)
                decision_margin = 0.0 if keep_out else required
                clearance = y - bottom if finite else None
                out = bool(near and permitted and clearance > decision_margin)
                trace = dict(wheel_id=key, paint_anchor_id=anchor.identity,
                             lane_id=max(anchor.lane_ids), finite_column=int(finite),
                             along_px=along, length_px=length,
                             near_wheel=int(near), permitted=int(permitted),
                             paint_column_x=int(round(x)), paint_top_y=top, paint_bottom_y=bottom,
                             clearance_px=clearance, required_margin_px=decision_margin,
                             entry_margin_px=required, edge_error_px=edge_error,
                             locally_verified=int(recent), verified_columns=anchor.verified_columns,
                             decision_rule="verified_continuation" if keep_out else "entry_margin",
                             is_out=int(out), selected=0, model_age_sec=now - anchor.full_time,
                             model_error_px=error)
                self.trace.append(trace)
                if near and permitted:
                    distance = max(top - y, y - bottom, 0.0)
                    candidates.append((distance, error, now - anchor.full_time, anchor.identity,
                                       anchor, trace, prepared_mask))
            if candidates:
                # 가까운 도색 위의 접지점을 먼 다른 도색 때문에 이탈로 만들지 않는다.
                _, _, _, _, anchor, selected, prepared_mask = min(candidates, key=lambda item: item[:4])
                selected["selected"] = 1
                if selected["is_out"]:
                    mask, origin, *_ = prepared_mask
                    state = dict(wheel_id=key, lane_id=selected["lane_id"],
                                 paint_anchor_id=anchor.identity, time_sec=now,
                                 along_px=selected["along_px"], length_px=selected["length_px"],
                                 keypoint_x=x, keypoint_y=y, paint_column_x=selected["paint_column_x"],
                                 paint_top_y=selected["paint_top_y"], paint_bottom_y=selected["paint_bottom_y"],
                                 clearance_px=selected["clearance_px"], required_margin_px=selected["required_margin_px"],
                                 entry_margin_px=selected["entry_margin_px"], edge_error_px=selected["edge_error_px"],
                                 locally_verified=selected["locally_verified"],
                                 verified_columns=selected["verified_columns"], decision_rule=selected["decision_rule"],
                                 model_age_sec=selected["model_age_sec"], model_error_px=selected["model_error_px"],
                                 evidence="observed_paint_below" if selected["model_age_sec"] == 0 else "registered_paint_below")
                    self.active[key] = state
                    self.last_anchor[key] = anchor.identity
                    if not self.last_out.get(key, False):
                        axis = project([anchor.a, anchor.b], anchor.matrix)
                        event = dict(state, event="departure", bbox_x1=x1, bbox_y1=y1,
                                     bbox_x2=x2, bbox_y2=y2, paint_origin_x=int(origin[0]),
                                     paint_origin_y=int(origin[1]), paint_mask=mask.copy(),
                                     axis_ax=float(axis[0, 0]), axis_ay=float(axis[0, 1]),
                                     axis_bx=float(axis[1, 0]), axis_by=float(axis[1, 1]))
                        self.last_events.append(event)
                        self.departed[key] = event
            # 중복 방지용 과거 기록을 현재 bbox 색상에 사용하지 않는다.
            if motion.valid or anchors:
                self.last_out[key] = key in self.active
        for key in set(self.last_seen) - {w["track_id"] for w in wheels}:
            if now - self.last_seen[key] > self.config.max_wheel_gap_sec:
                self.last_seen.pop(key, None)
                self.last_out.pop(key, None)
                self.last_anchor.pop(key, None)
        return self.last_events

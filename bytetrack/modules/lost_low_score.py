"""M5：近期Lost轨迹与剩余低分框的保守恢复（新设计，未做精度消融）。

调用位置：正常高分/低分关联完成之后，传入尚未使用的低分框。
本文件只返回确定的配对，不修改轨迹、不创建ID、不补写历史帧。
跟踪器拿到配对后调用 re_activate(..., new_id=False)。

算法顺序：状态/年龄筛选 → 几何/运动门控 → 双向歧义拒绝 →
带拒绝选项的一对一分配 → 连续观测确认。所有参数都是起始值。
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..basetrack import TrackState
from ..geometry import ious


@dataclass(frozen=True)
class RecoveryConfig:
    low_score: float = 0.1
    high_score: float = 0.5
    max_gap_frames: int = 25  # 真实帧数，还受跟踪器实际retention_frames限制
    min_track_age: int = 3
    confirm_frames: int = 2  # 设1可做无连续确认的消融
    max_center_distance: float = 0.75  # 按各轴平均宽高归一化
    max_size_ratio: float = 2.5
    max_mahalanobis_sq: float = 9.2103  # xy统计距离门控；非普遍置信保证
    min_iou: float = 0.05
    zero_iou_center: float = 0.25  # 无重叠时仅允许中心非常接近
    max_cost: float = 0.65
    ambiguity_margin: float = 0.08
    protected_iou: float = 0.5  # 避免接到已有正常轨迹的重复框
    min_direction_cosine: float = -0.2
    continuity_distance: float = 0.5

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("恢复参数必须为有限数")
        for name in ['max_gap_frames', 'min_track_age', 'confirm_frames']:
            value = getattr(self, name)
            if value < 1 or int(value) != value:
                raise ValueError(f"{name}必须为正整数")
        if not 0 <= self.low_score < self.high_score <= 1:
            raise ValueError("需要0<=low_score<high_score<=1")
        if self.confirm_frames > self.max_gap_frames:
            raise ValueError("确认帧数不能超过恢复窗口")
        if self.max_size_ratio < 1 or self.max_mahalanobis_sq <= 0:
            raise ValueError("尺寸或统计门控无效")
        for name in ['max_center_distance', 'continuity_distance', 'max_cost']:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name}必须为正")
        if not 0 <= self.ambiguity_margin < self.max_cost:
            raise ValueError("歧义间隔必须小于最大代价")
        if not 0 <= self.zero_iou_center <= self.max_center_distance:
            raise ValueError("无重叠中心门控必须比总中心门控严格")
        if not 0 <= self.min_iou <= 1 or not 0 <= self.protected_iou <= 1:
            raise ValueError("IoU参数必须在[0,1]")
        if not -1 <= self.min_direction_cosine <= 1:
            raise ValueError("方向余弦必须在[-1,1]")


def valid_box(box):
    return (box.shape == (4,) and np.isfinite(box).all()
            and np.all(box[2:] > box[:2]))


def box_geometry(a, b):
    """中心距离和尺度比：不用图像全幅归一化，保护远处小框。"""
    awh, bwh = a[2:] - a[:2], b[2:] - b[:2]
    delta = (a[:2] + a[2:] - b[:2] - b[2:]) / 2
    normalizer = np.maximum((awh + bwh) / 2, 1.0)
    distance = float(np.linalg.norm(delta / normalizer))
    ratio = float(np.max(np.maximum(awh / bwh, bwh / awh)))
    return distance, ratio


class LostLowScoreRecovery:
    def __init__(self, config=None):
        self.config = config or RecoveryConfig()
        self.pending = {}  # track_id -> (上一候选帧号, 候选框, 连续命中数)
        self.events = []   # 仅当前帧；调用者自行落盘，避免长视频内存增长
        self.last_frame = 0

    def reset(self):
        """每个新序列必须清除待确认候选，不能跨视频继承ID证据。"""
        self.pending.clear()
        self.events.clear()
        self.last_frame = 0

    def _cost(self, track, detection, gap, protected):
        cfg = self.config
        a, b = np.asarray(track.tlbr), np.asarray(detection.tlbr)
        if not valid_box(a) or not valid_box(b):
            return np.inf, 'invalid_box'
        score = float(detection.score)
        # 与原版低分分组保持一致：两侧都是严格不等式。
        if not cfg.low_score < score < cfg.high_score:
            return np.inf, 'score'
        if protected and ious([b], protected).max() >= cfg.protected_iou:
            return np.inf, 'active_conflict'
        center, size = box_geometry(a, b)
        if center > cfg.max_center_distance or size > cfg.max_size_ratio:
            return np.inf, 'geometry'
        iou = float(ious([a], [b])[0, 0])
        if iou < cfg.min_iou and center > cfg.zero_iou_center:
            return np.inf, 'overlap'
        # 统计门控不能替代上面的硬几何门控，协方差膨胀不应放行远处车辆。
        try:
            maha = float(track.kalman_filter.gating_distance(
                track.mean, track.covariance,
                np.asarray([detection.to_xyah()]), only_position=True)[0])
        except (ValueError, np.linalg.LinAlgError):
            return np.inf, 'invalid_covariance'
        if not np.isfinite(maha) or maha > cfg.max_mahalanobis_sq:
            return np.inf, 'motion'
        # 运动足够明显才检查方向，静止车/极低速车不作余弦判断。
        velocity = np.asarray(track.mean[4:6])
        previous = getattr(track, 'last_observation_box', None)
        if previous is not None and np.linalg.norm(velocity) >= 0.5:
            displacement = (b[:2] + b[2:] - previous[:2] - previous[2:]) / 2
            if np.linalg.norm(displacement) > 2:
                cosine = float(np.dot(velocity, displacement) /
                               (np.linalg.norm(velocity) * np.linalg.norm(displacement)))
                if cosine < cfg.min_direction_cosine:
                    return np.inf, 'direction'
        # 不乘置信度：低分框已通过分组，恢复主要依靠几何和运动。
        cost = (0.55 * (1 - iou)
                + 0.30 * center / cfg.max_center_distance
                + 0.10 * maha / cfg.max_mahalanobis_sq
                + 0.05 * gap / cfg.max_gap_frames)
        return cost, 'eligible' if cost < cfg.max_cost else 'cost'

    def match(self, lost_tracks, detections, frame_id, protected=(), retention_frames=None):
        """返回[(lost_tracks索引, detections索引)]，只包含已确认候选。

        每帧调用一次，frame_id严格递增。轨迹已经预测到当前帧；不重复预测。
        gap基于最后一次正式更新frame_id计算，pending命中不重置此时钟。
        """
        if int(frame_id) != frame_id or frame_id <= self.last_frame:
            raise ValueError("帧号必须为严格递增的正整数；新序列先reset")
        self.last_frame = frame_id
        cfg = self.config
        self.events = []
        lifetime = cfg.max_gap_frames if retention_frames is None else min(cfg.max_gap_frames, retention_frames)
        active_boxes = [np.asarray(t.tlbr) for t in protected if valid_box(np.asarray(t.tlbr))]
        eligible = []
        for i, t in enumerate(lost_tracks):
            gap = frame_id - t.frame_id
            age = t.frame_id - t.start_frame + 1
            if t.state == TrackState.Lost and t.is_activated and 1 <= gap <= lifetime and age >= cfg.min_track_age:
                eligible.append((i, gap))
        ids = [lost_tracks[i].track_id for i, _ in eligible]
        if len(ids) != len(set(ids)):
            raise ValueError("Lost候选中出现重复ID")
        n, m = len(eligible), len(detections)
        cost = np.full((n, m), np.inf)
        for row, (i, gap) in enumerate(eligible):
            for j, d in enumerate(detections):
                c, reason = self._cost(lost_tracks[i], d, gap, active_boxes)
                if c < cfg.max_cost:
                    cost[row, j] = c
                self.events.append(dict(frame=frame_id, track_id=lost_tracks[i].track_id,
                                        detection_index=j, cost=float(c) if np.isfinite(c) else None,
                                        status=reason, gap=gap))
        # 双向唯一性：轨迹有两个近似等价框，或框有两条近似等价轨迹，都暂缓。
        allowed = cost.copy()
        for i, j in zip(*np.where(np.isfinite(cost))):
            other_row = np.delete(cost[i], j)
            other_col = np.delete(cost[:, j], i)
            if ((len(other_row) and other_row.min() - cost[i, j] < cfg.ambiguity_margin)
                    or (len(other_col) and other_col.min() - cost[i, j] < cfg.ambiguity_margin)):
                allowed[i, j] = np.inf
                self.events.append(dict(frame=frame_id, track_id=lost_tracks[eligible[i][0]].track_id,
                                        detection_index=int(j), status='ambiguous'))
        # 每条轨迹一个dummy列表示拒绝匹配，避免先硬匹配再阈值过滤的错误。
        assignment = []
        if n and m:
            augmented = np.full((n, m + n), cfg.max_cost)
            augmented[:, :m] = np.where(np.isfinite(allowed), allowed, 1e6)
            rr, cc = linear_sum_assignment(augmented)
            assignment = [(i, j) for i, j in zip(rr, cc) if j < m and np.isfinite(allowed[i, j])]
        next_pending, confirmed = {}, []
        for row, j in assignment:
            i, gap = eligible[row]
            track = lost_tracks[i]
            box = np.asarray(detections[j].tlbr).copy()
            old = self.pending.get(track.track_id)
            count = 1
            if old is not None and old[0] == frame_id - 1:
                distance, size = box_geometry(old[1], box)
                if distance <= cfg.continuity_distance and size <= cfg.max_size_ratio:
                    count = old[2] + 1
            if count >= cfg.confirm_frames:
                confirmed.append((i, int(j)))
                status = 'confirmed'
            else:
                next_pending[track.track_id] = (frame_id, box, count)
                status = 'pending'
            self.events.append(dict(frame=frame_id, track_id=track.track_id,
                                    detection_index=int(j), gap=gap, hits=count, status=status))
        # 无候选、被高分恢复、过期、歧义或中断的pending立即清理。
        self.pending = next_pending
        return confirmed

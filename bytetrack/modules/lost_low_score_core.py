"""M5核心版：近期Lost轨迹匹配剩余低分框，当帧恢复原ID。
保留中心距离/尺寸倍率硬门控及原代价权重；运动一致性作为软代价。
删除IoU—中心联合门控、马氏距离硬门控，以及方向、重复框保护、
双向歧义拒绝、连续确认。不使用GT，不修改轨迹，返回配对索引。
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..basetrack import TrackState
from ..geometry import ious


@dataclass(frozen=True)
class CoreRecoveryConfig:
    low_score: float = 0.1
    high_score: float = 0.5
    max_gap_frames: int = 25
    min_track_age: int = 3
    max_center_distance: float = 0.75
    max_size_ratio: float = 2.5
    max_mahalanobis_sq: float = 9.2103  # 仅为代价归一化尺度，保留字段名兼容旧配置
    max_cost: float = 0.65


class CoreLostLowScoreRecovery:
    def __init__(self, config=None):
        self.config = config or CoreRecoveryConfig()
        self.events = []  # 仅记录成功配对，供实验统计

    def reset(self):
        self.events = []

    def match(self, lost_tracks, detections, frame_id, protected=(), retention_frames=None):
        # protected仅为兼容原跟踪器接口；核心版不进行重复框保护。
        cfg = self.config
        self.events = []
        lifetime = min(cfg.max_gap_frames, retention_frames) if retention_frames is not None else cfg.max_gap_frames
        # 只恢复已经确认、历史够长且仍在保留窗口内的Lost轨迹。
        candidates = [(i, frame_id - t.frame_id) for i, t in enumerate(lost_tracks)
                      if t.state == TrackState.Lost and t.is_activated
                      and 1 <= frame_id - t.frame_id <= lifetime
                      and t.frame_id - t.start_frame + 1 >= cfg.min_track_age]
        n, m = len(candidates), len(detections)
        if not n or not m:
            return []
        cost = np.full((n, m), np.inf)
        for row, (i, gap) in enumerate(candidates):
            track = lost_tracks[i]
            a = np.asarray(track.tlbr)
            for j, det in enumerate(detections):
                b = np.asarray(det.tlbr)
                if not cfg.low_score < det.score < cfg.high_score:
                    continue
                if not all(x.shape == (4,) and np.isfinite(x).all() and np.all(x[2:] > x[:2]) for x in (a, b)):
                    continue
                # 用框宽高归一化中心位移；尺寸比对放大和缩小对称。
                awh, bwh = a[2:] - a[:2], b[2:] - b[:2]
                delta = (a[:2] + a[2:] - b[:2] - b[2:]) / 2
                center = np.linalg.norm(delta / np.maximum((awh + bwh) / 2, 1.0))
                size = np.max(np.maximum(awh / bwh, bwh / awh))
                iou = float(ious([a], [b])[0, 0])
                if center > cfg.max_center_distance or size > cfg.max_size_ratio:
                    continue
                # 马氏距离只参与软代价，不再单独设上限。
                try:
                    maha = float(track.kalman_filter.gating_distance(track.mean, track.covariance,
                                 np.asarray([det.to_xyah()]), only_position=True)[0])
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not np.isfinite(maha) or maha < 0:
                    continue
                # 不裁剪归一化马氏距离；超过1时继续增加代价。
                cost[row, j] = (0.55 * (1 - iou) + 0.30 * center / cfg.max_center_distance
                                + 0.10 * maha / cfg.max_mahalanobis_sq + 0.05 * gap / cfg.max_gap_frames)
        # 每行附加一个拒绝选项，避免强行分配；真实配对须严格低于max_cost。
        augmented = np.full((n, m + n), cfg.max_cost)
        augmented[:, :m] = np.where(cost < cfg.max_cost, cost, 1e6)
        rows, cols = linear_sum_assignment(augmented)
        pairs = [(candidates[r][0], int(c)) for r, c in zip(rows, cols) if c < m and cost[r, c] < cfg.max_cost]
        self.events = [dict(frame=frame_id, track_id=lost_tracks[i].track_id,
                            detection_index=j, status='confirmed') for i, j in pairs]
        # 调用者使用re_activate(new_id=False)真正恢复轨迹，本模块只提供配对。
        return pairs

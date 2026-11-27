"""Frame-wise association and track lifecycle management."""

import numpy as np

from .kalman_filter import KalmanFilter
from .basetrack import BaseTrack, TrackState
from . import matching
from .modules.gap_state import velocity_retention, velocity_step_scale
from .modules.lost_low_score import LostLowScoreRecovery, RecoveryConfig


def _validate_tracker_parameters(track_thresh, track_buffer, match_thresh,
                                 frame_rate, min_box_area, center_weight):
    """集中校验两个跟踪器共享的公共参数。"""
    if not np.isfinite(track_thresh) or not 0.0 <= track_thresh < 1.0:
        raise ValueError("track_thresh must be in [0, 1)")
    if not np.isfinite(match_thresh) or not 0.0 <= match_thresh <= 1.0:
        raise ValueError("match_thresh must be in [0, 1]")
    if not np.isfinite(track_buffer) or track_buffer < 0:
        raise ValueError("track_buffer must be non-negative")
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    if not np.isfinite(min_box_area) or min_box_area < 0:
        raise ValueError("min_box_area must be non-negative")
    if not np.isfinite(center_weight) or not 0.0 <= center_weight <= 1.0:
        raise ValueError("center_weight must be in [0, 1]")


def _filter_detections(dets, min_box_area):
    """过滤非有限、非法尺寸、非法分数和面积过小的检测框。"""
    array = np.asarray(dets, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError("dets must have shape [N, >=5]")
    array = array[:, :5]
    widths = array[:, 2] - array[:, 0]
    heights = array[:, 3] - array[:, 1]
    areas = widths * heights
    valid = (
        np.all(np.isfinite(array), axis=1) &
        (widths > 0.0) &
        (heights > 0.0) &
        (areas >= float(min_box_area)) &
        (array[:, 4] >= 0.0) &
        (array[:, 4] <= 1.0)
    )
    return array[valid]


def _track_area(track):
    """返回轨迹当前预测框的非负面积。"""
    tlwh = track.tlwh
    return max(float(tlwh[2]), 0.0) * max(float(tlwh[3]), 0.0)


class STrack(BaseTrack):
    """
    单条轨迹的具体实现 —— 封装了卡尔曼滤波的状态管理和坐标转换。

    坐标格式:
        tlwh: [top_left_x, top_left_y, width, height]   ← 检测器输出格式
        tlbr: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]  ← IoU 计算格式
        xyah: [center_x, center_y, aspect_ratio, height] ← 卡尔曼滤波状态格式

    选择 xyah 作为卡尔曼状态的原因:
        - 宽高比 a 和高度 h 比宽度 w 更稳定（w 随目标姿态变化大）
        - 中心点 (x,y) 比左上角更直观地表示目标位置
        - 直接观测高度 h 比通过 xy 间接推断更准确
    """

    # 类级别的共享卡尔曼滤波器 —— 所有 STrack 实例共用同一个 KF 对象
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        """
        创建一条新轨迹。

        参数:
            tlwh:  [x, y, w, h] 格式的初始检测框
            score: 检测置信度 (0~1)
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None   # 卡尔曼状态的均值和协方差
        self.is_activated = False                  # 是否已被激活（转为 Tracked）
        self.score = score
        self.tracklet_len = 0                      # 连续命中次数
        self.frames_since_update = 0               # 距上次更新的帧数
        self._aspect_scale = 1.0                   # per-track 宽高比过程噪声尺度

    @staticmethod
    def _gap_velocity_retention(gap):
        return velocity_retention(gap)

    @classmethod
    def _gap_velocity_step_scale(cls, gap):
        return velocity_step_scale(gap)

    def predict(self, image_height=None):
        """
        卡尔曼预测 —— 估计目标在当前帧的位置。

        分级速度衰减策略（替代原来简单的清零/保留二分法）:
            - 刚丢失 (≤5帧):   保留速度，相信运动惯性
            - 短期丢失 (6-20帧): 平滑衰减到原速度的 50%
            - 中期丢失 (21-50帧): 继续平滑衰减到原速度的 20%
            - 长期丢失 (>50帧):  速度清零，原地等待

        原逻辑: Lost 状态直接清零 vh，过于粗暴。
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            if self.kalman_filter.gap_recovery_enabled:
                gap = self.frames_since_update
                mean_state[4:] *= self._gap_velocity_step_scale(gap)
            else:
                # 原始 ByteTrack 规则：Lost 状态仅清零高度速度 vh。
                mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance,
            aspect_scale=self._aspect_scale)
        self.frames_since_update += 1

    @staticmethod
    def multi_predict(stracks, kalman_filter, image_height=None):
        """
        批量预测 —— 一次卡尔曼运算处理所有轨迹。

        分级速度衰减（与 predict() 保持一致）。
        """
        if len(stracks) == 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        # Lost 轨迹速度策略：默认使用原始 ByteTrack；可选启用分级衰减。
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                if kalman_filter.gap_recovery_enabled:
                    gap = st.frames_since_update
                    multi_mean[i][4:] *= \
                        st._gap_velocity_step_scale(gap)
                else:
                    multi_mean[i][7] = 0
        aspect_scales = np.asarray([st._aspect_scale for st in stracks])
        multi_mean, multi_covariance = kalman_filter.multi_predict(
            multi_mean, multi_covariance, aspect_scales=aspect_scales)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov
            stracks[i].frames_since_update += 1

    # ── 坐标格式转换（静态方法，不需要实例化即可使用）──

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """
        tlwh → xyah: [x, y, w, h] → [cx, cy, a, h]
            cx = x + w/2   (中心点 x)
            cy = y + h/2   (中心点 y)
            a  = w / h     (宽高比)
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2    # 左上角 → 中心点
        ret[2] /= ret[3]           # 宽度 → 宽高比
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """tlbr → tlwh: [x1,y1,x2,y2] → [x1, y1, x2-x1, y2-y1]"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """tlwh → tlbr: [x,y,w,h] → [x, y, x+w, y+h]"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """将当前轨迹的位置转为卡尔曼状态格式"""
        return self.tlwh_to_xyah(self.tlwh)

    # ── 属性：获取当前轨迹的边界框 ──

    @property
    def tlwh(self):
        """
        获取当前轨迹的 tlwh 格式边界框。

        如果卡尔曼状态存在 → 从状态向量还原（mean 的前 4 维）
            mean[0:2] = [cx, cy]  → 中心点 → 左上角
            mean[2]   = a         → 宽高比 → 宽度: w = a * h
            mean[3]   = h         → 高度
            x = cx - w/2, y = cy - h/2

        如果卡尔曼状态不存在 → 返回初始值（轨迹创建时的检测框）
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()   # [cx, cy, a, h]
        ret[2] *= ret[3]              # a * h = w
        ret[:2] -= ret[2:] / 2        # 中心点 → 左上角
        return ret                     # [x, y, w, h]

    @property
    def tlbr(self):
        """
        获取当前轨迹的 tlbr 格式边界框 [x1, y1, x2, y2]。
        主要用于 IoU 计算（IoU 在 tlbr 格式上计算最方便）。
        """
        ret = self.tlwh.copy()       # [x, y, w, h]
        ret[2:] += ret[:2]            # x2 = x + w, y2 = y + h
        return ret                     # [x1, y1, x2, y2]

    # ── 轨迹生命周期管理 ──

    def activate(self, kalman_filter, frame_id):
        """
        激活一条新轨迹 —— 将其从 New 状态转为 Tracked 状态。

        激活时:
            1. 分配全局唯一 ID
            2. 初始化卡尔曼滤波器状态
            3. 设置状态为 Tracked
            4. 记录起始帧号

        第一帧就激活的轨迹（frame_id == 1）直接标记为 is_activated。
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        # 用初始检测框初始化卡尔曼滤波
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0              # 命中计数器归零
        self.frames_since_update = 0       # 重置失联计数
        self.state = TrackState.Tracked    # 转为已跟踪状态
        self._aspect_scale = 1.0           # 重置宽高比过程噪声尺度
        if frame_id == 1:
            self.is_activated = True       # 第一帧直接激活
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.last_observation_box = self.tlbr.copy()

    def re_activate(self, new_track, frame_id, new_id=False,
                    image_height=None):
        """
        重新激活一条丢失的轨迹 —— 它有新检测框与之匹配。

        改进点:
            - 短时间丢失 (<10帧): 标准卡尔曼更新，保留运动历史
            - 中等丢失: 膨胀协方差后更新
            - 达到重置阈值: 按当前检测完全重新初始化

        参数:
            new_track: 匹配到的新检测框 (STrack)
            frame_id:  当前帧号
            new_id:    是否分配新 ID（默认不分配，保持 ID 不变）
        """
        measurement = self.tlwh_to_xyah(new_track.tlwh)
        if self.kalman_filter.gap_recovery_enabled:
            gap = self.frames_since_update
            self.mean, self.covariance, self._aspect_scale = \
                self.kalman_filter.reinit_after_gap(
                    self.mean, self.covariance, measurement,
                    gap, score=new_track.score,
                    aspect_scale=self._aspect_scale)
        else:
            self.mean, self.covariance, self._aspect_scale = \
                self.kalman_filter.update(
                    self.mean, self.covariance, measurement,
                    score=new_track.score,
                    aspect_scale=self._aspect_scale)
        gap = self.frames_since_update
        self.tracklet_len = 0               # 重激活后命中计数归零
        self.frames_since_update = 0
        self.state = TrackState.Tracked     # 恢复为跟踪状态
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()  # 可选：分配新 ID
        self.score = new_track.score
        self.last_observation_box = new_track.tlbr.copy()

    def update(self, new_track, frame_id, image_height=None):
        """
        更新一条正常跟踪中的轨迹 —— 用匹配到的检测框修正卡尔曼状态。

        改进点:
            - 使用 score-weighted update: 高分检测框可信度更高，
              低分检测框可信度较低，通过调节等效观测噪声来实现。
        """
        self.frame_id = frame_id
        self.tracklet_len += 1               # 连续命中次数 +1
        self.frames_since_update = 0

        new_tlwh = new_track.tlwh
        measurement = self.tlwh_to_xyah(new_tlwh)
        self.mean, self.covariance, self._aspect_scale = \
            self.kalman_filter._update_for_score(
                self.mean, self.covariance, measurement,
                score=new_track.score,
                aspect_scale=self._aspect_scale)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.last_observation_box = new_track.tlbr.copy()

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# ═══════════════════════════════════════════════════════════════════
#  BYTETracker —— 主跟踪器
# ═══════════════════════════════════════════════════════════════════

class BYTETracker:
    """
    ByteTrack 主跟踪器 —— 实现 BYTE 数据关联机制。

    状态管理:
        tracked_stracks:  当前活跃的轨迹（Tracked 状态）
        lost_stracks:     暂时丢失的轨迹（Lost 状态，还在缓冲区中等待恢复）
        removed_stracks:  永久删除的轨迹（超出 max_time_lost 或从未确认）

    每帧调用 update() 完成以下步骤:
        Step 0: 分离高分框和低分框
        Step 1: 高分框 ↔ 所有轨迹（卡尔曼预测 + IoU 匹配）
        Step 2: 低分框 ↔ 未匹配的轨迹（仅 IoU 匹配，恢复遮挡目标）
        Step 3: 未确认轨迹 ↔ 剩余高分框（给新轨迹转正机会）
        Step 4: 创建新轨迹（未匹配的高分框）
        Step 5: 清理过期轨迹（超出 max_time_lost 的丢失轨迹）
    """

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8,
                 frame_rate=30, mot20=False, min_box_area=10, center_weight=0.0,
                 center_distance_mode="legacy_blend",
                 enable_score_weight=False, enable_gap_recovery=False,
                 enable_aspect_adaptive=False,
                 collect_association=False,
                 kalman_filter=None, enable_lost_low_score=False, recovery_config=None):
        """
        初始化 ByteTrack 跟踪器。

        参数:
            track_thresh:  高分/低分检测框的分界线（默认 0.5）
                           >此阈值为高分框，<此阈值为低分框
            track_buffer:  轨迹丢失后保留的最大帧数（默认 30）
                           30 帧 @ 30fps = 1 秒的缓冲时间
            match_thresh:  第一阶段允许的最大匹配代价（默认 0.8）
                           第二阶段固定使用 0.5
                           第三阶段固定使用 0.7
            frame_rate:    视频帧率（用于将 track_buffer 转换为实际帧数）
            mot20:         是否使用 MOT20 模式（MOT20 数据集场景更拥挤）
            min_box_area:  最小检测框面积（过滤过小的噪声框）
            center_weight: 中心距离在匹配代价中的权重 [0, 1]（默认 0.2）
                           0 = 纯 IoU 匹配（原版行为）
            center_distance_mode: ``gated_rescue`` 仅在高分第一阶段对通过
                                  运动/尺度门控的候选作 IoU 补偿；
                                  ``legacy_blend`` 为已验证的全阶段线性混合
            enable_score_weight: 是否启用置信度加权观测噪声（默认关闭）
            enable_gap_recovery: 是否启用分级速度衰减与间隔恢复（默认关闭）
            kalman_filter: 可选自定义滤波器，主要用于受控实验
            collect_association: 是否记录每次成功匹配的预测框-检测框相似度
                (IoU / 归一化中心距离 / NIS)，用于诊断关联质量
        """
        _validate_tracker_parameters(
            track_thresh, track_buffer, match_thresh, frame_rate,
            min_box_area, center_weight)
        if center_distance_mode not in (
                "gated_rescue", "identity_priority", "legacy_blend"):
            raise ValueError(
                "center_distance_mode must be 'gated_rescue', "
                "'identity_priority', or 'legacy_blend'")
        self.tracked_stracks = []    # 当前活跃轨迹（Tracked 状态）
        self.lost_stracks = []       # 暂时丢失的轨迹（Lost 状态）
        self.removed_stracks = []    # 已删除的轨迹

        self.frame_id = 0            # 当前处理的帧计数
        if enable_lost_low_score:
            recovery_config = recovery_config or RecoveryConfig(high_score=track_thresh)
            if recovery_config.low_score != 0.1 or recovery_config.high_score != track_thresh:
                raise ValueError("恢复模块分数区间必须与ByteTrack低分分组一致")
        self.low_score_recovery = LostLowScoreRecovery(recovery_config) if enable_lost_low_score else None
        self.recovery_events = []
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.center_weight = center_weight
        self.center_distance_mode = center_distance_mode

        # 新轨迹创建阈值 —— 比 track_thresh 高 0.1
        # 只有足够可靠的检测才创建新轨迹，减少虚警
        self.det_thresh = track_thresh + 0.1

        # 将 buffer 帧数按帧率缩放
        # 例如: track_buffer=30, frame_rate=30 → buffer_size=30 帧 (1秒)
        # 例如: track_buffer=30, frame_rate=15 → buffer_size=15 帧 (1秒)
        # 这样不同帧率下 buffer 的实际时间长度一致
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

        reset_threshold = max(2, self.max_time_lost)
        covariance_threshold = min(10, reset_threshold - 1)
        if kalman_filter is None:
            self.kalman_filter = KalmanFilter(
                enable_score_weight=enable_score_weight,
                enable_gap_recovery=enable_gap_recovery,
                gap_covariance_threshold=covariance_threshold,
                gap_reset_threshold=reset_threshold,
                enable_aspect_adaptive=enable_aspect_adaptive,
            )
        else:
            self.kalman_filter = kalman_filter
        self.improvement_flags = {
            'score_weight': self.kalman_filter.score_weight_enabled,
            'center_distance': self.center_weight > 0.0,
            'center_distance_mode': self.center_distance_mode,
            'gap_recovery': self.kalman_filter.gap_recovery_enabled,
            'aspect_adaptive': self.kalman_filter.aspect_adaptive_enabled,
        }

        # 是否使用 MOT20 模式
        # MOT20 场景更拥挤（人群密集），不使用得分融合（避免高分干扰）
        self.mot20 = mot20

        # 关联相似度诊断（预测框 vs 检测框），默认关闭
        self.collect_association = bool(collect_association)
        self.association_records = []

    def _record_association(self, track, det):
        """记录一次成功匹配的预测框-检测框相似度（IoU/中心距离/NIS）。

        仅在 ``collect_association`` 开启时执行。调用时机在 ``track.update``/
        ``re_activate`` 之前，因此 ``track.tlbr`` / ``track.mean`` 仍是预测态，
        得到的正是"预测框与检测框相似程度"。

        NIS（归一化新息平方）用预测状态到检测观测的马氏平方距离衡量，
        高斯假设下期望 ≈ 4（自由度 4），越大表示预测越偏离观测。
        """
        if not self.collect_association:
            return
        tlbr_pred = np.asarray(track.tlbr, dtype=np.float64)
        tlbr_det = np.asarray(det.tlbr, dtype=np.float64)
        iou = float(matching.ious([tlbr_pred], [tlbr_det])[0, 0])
        center_dist = float(matching.center_distance(
            [tlbr_pred], [tlbr_det])[0, 0])
        nis = float('nan')
        if track.mean is not None and track.covariance is not None:
            measurement = STrack.tlwh_to_xyah(det.tlwh)
            nis = float(self.kalman_filter.gating_distance(
                track.mean, track.covariance, [measurement])[0])
        self.association_records.append({
            'frame': self.frame_id,
            'track_id': int(track.track_id),
            'iou': iou,
            'center_dist': center_dist,
            'nis': nis,
        })

    def update(self, dets, img_shape):
        """
        处理一帧检测结果，返回当前帧的在线跟踪目标。

        这是 ByteTrack 的核心入口函数 —— 实现了完整的 BYTE 数据关联流程。

        参数:
            dets:      numpy 数组 [N, 5]: [x1, y1, x2, y2, score]，
                       坐标为原始画面的像素值
            img_shape: 原始图像尺寸，用于归一化透视模型的纵向坐标

        返回:
            list[STrack]: 当前帧的活跃轨迹列表

        ── BYTE 数据关联详解 ──────────────────────────────────────

        【概念说明】
        高分框: score ≥ track_thresh  → 置信度高，通常是清晰可见的目标
        低分框: 0.1 < score < track_thresh → 置信度低，可能是被遮挡的目标

        【第一阶段: 高分框匹配】
        匹配对象: 所有轨迹（tracked + lost） ↔ 高分框
        代价矩阵: fuse_score(1 - IoU, det_scores) for MOT17
                 1 - IoU for MOT20
        阈值: match_thresh (0.8)
        目的: 先处理最可靠的匹配，建立主干跟踪

        【第二阶段: 低分框匹配】
        匹配对象: 仅第一阶段未匹配的轨迹 ↔ 低分框
        代价矩阵: 1 - IoU（不使用得分调制）
        阈值: 0.5（比第一阶段宽松）
        目的: 恢复因遮挡导致置信度下降的目标
        前提: 低分框中可能包含遮挡目标的"残影"检测
        关键: 只用 IoU 匹配（低分框的得分不可靠，不用于调制）

        【第三阶段: 未确认轨迹匹配】
        匹配对象: unconfirmed 轨迹 ↔ 第一阶段剩余高分框
        代价矩阵: fuse_score(1 - IoU, det_scores)
        阈值: 0.7
        目的: 让刚出现的轨迹有机会被确认
        限制: unconfirmed 轨迹通常是只出现过一帧的，如果连续两帧
              都没匹配到就会被删除

        【第四阶段: 创建新轨迹】
        从第二阶段和第三阶段都未匹配的高分框中创建新轨迹
        要求: score > det_thresh (track_thresh + 0.1)

        【第五阶段: 清理过期轨迹】
        删除 max_time_lost 帧内都没有更新的丢失轨迹
        """
        self.frame_id += 1
        activated_stracks = []    # 本帧激活/更新的轨迹
        refind_stracks = []       # 本帧重新找回的轨迹
        lost_stracks = []         # 本帧新标记为丢失的轨迹
        removed_stracks = []      # 本帧永久删除的轨迹

        # ── Step 0: 过滤非法/过小框，再按置信度分组 ──
        dets = _filter_detections(dets, self.min_box_area)
        # scores 取第 5 列（置信度）
        if dets.shape[0] == 0:
            scores = np.array([])
            bboxes = np.empty((0, 4))
        else:
            scores = dets[:, 4]
            bboxes = dets[:, :4]

        # 检测框保持原始画面的像素坐标；只有 FLT 透视计算使用 y / H。
        image_height = float(img_shape[0])
        if not np.isfinite(image_height) or image_height <= 0:
            raise ValueError("img_shape must contain a positive image height")

        # 分类检测框:
        #   remain_inds: 高分框 (score ≥ track_thresh) → 第一阶段匹配
        #   inds_low:    非极低分框 (score > 0.1) → 第二阶段候选
        #   inds_second: 低分框 (0.1 < score < track_thresh) → 第二阶段用
        remain_inds = scores > self.track_thresh                         # 高分
        inds_low = scores > 0.1                                          # 非极低
        inds_high = scores < self.track_thresh                           # 非高分
        inds_second = inds_low & inds_high                               # 低分 = 非极低且非高分

        dets_second = bboxes[inds_second]          # 第二阶段低分框
        dets = bboxes[remain_inds]                  # 第一阶段高分框
        scores_keep = scores[remain_inds]           # 高分框得分
        scores_second = scores[inds_second]         # 低分框得分

        # 将高分框封装为 STrack 对象
        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                          for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # ── 分离已确认和未确认的轨迹 ──
        # unconfirmed: 未激活的轨迹（通常是只出现了一帧的新轨迹）
        # tracked:     已确认的轨迹（正常跟踪中）
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ═══════════════════════════════════════════════════════════════
        #  Step 1: 第一阶段匹配 —— 高分框 ↔ 活跃轨迹 + 丢失轨迹
        #  所有轨迹（包括 lost 的）都参与匹配，
        #  这样如果丢失的轨迹在当前位置附近出现了高分检测框，
        #  就能立即恢复。
        # ═══════════════════════════════════════════════════════════════

        # 合并活跃轨迹和丢失轨迹作为匹配候选
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # 用卡尔曼滤波预测所有候选轨迹在当前帧的位置
        STrack.multi_predict(strack_pool, self.kalman_filter, image_height)

        # 计算组合代价矩阵: 融合 IoU 距离 + 中心点距离
        # center_weight 控制中心距离的权重:
        #   0 = 纯 IoU（原版行为）；非零权重需按当前数据重新标定阈值。
        dists = matching.combined_distance(strack_pool, detections,
                                           center_weight=self.center_weight,
                                           mode=self.center_distance_mode)

        # MOT17: 用检测得分调制代价矩阵（高分框更容易匹配）
        # MOT20: 不用得分调制（拥挤场景下得分不可靠）
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)

        # 匈牙利算法求解最优匹配
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh)

        # 处理匹配结果
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            self._record_association(track, det)
            if track.state == TrackState.Tracked:
                # 轨迹正常跟踪中 → 直接更新
                track.update(detections[idet], self.frame_id, image_height)
                activated_stracks.append(track)
            else:
                # 轨迹之前丢失了 → 重新激活（保留运动历史）
                track.re_activate(det, self.frame_id, new_id=False,
                                  image_height=image_height)
                refind_stracks.append(track)

        # ═══════════════════════════════════════════════════════════════
        #  Step 2: 第二阶段匹配 —— 低分框 ↔ 第一阶段未匹配的轨迹
        #
        #  这是 ByteTrack 的精髓所在。
        #  当目标被遮挡时，检测得分会骤降（< track_thresh），
        #  但检测框位置通常仍然准确（因为检测器还是"看到了"一部分目标）。
        #  用低分框去匹配第一阶段未匹配上的轨迹，可以恢复被遮挡的目标。
        #
        #  为什么只用未匹配的轨迹？
        #  - 已经匹配到的轨迹不需要额外的低分框
        #  - 低分框噪声较大，只和"还没找到匹配"的轨迹匹配更安全
        # ═══════════════════════════════════════════════════════════════

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                                 for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        # 只从第一阶段未匹配的轨迹中选 Tracked 状态的
        # (丢失状态的轨迹已经"找不到了"，不给它们匹配低分框)
        r_tracked_stracks = [
            strack_pool[i] for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]

        # 计算组合代价矩阵（低分框同样受益于中心距离鲁棒性）
        second_stage_center_weight = (
            self.center_weight
            if self.center_distance_mode in (
                "identity_priority", "legacy_blend") else 0.0
        )
        dists = matching.combined_distance(
            r_tracked_stracks, detections_second,
            center_weight=second_stage_center_weight,
            mode=self.center_distance_mode)

        # 第二阶段使用固定阈值 0.5（比第一阶段宽松）
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            self._record_association(track, det)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, image_height)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False,
                                  image_height=image_height)
                refind_stracks.append(track)

        # M5：只让进入本帧前已经Lost的轨迹恢复；活跃轨迹仍享有前两轮优先权。
        # self.lost_stracks中首轮已恢复的对象state已变为Tracked，会被模块排除。
        self.recovery_events = []
        if self.low_score_recovery is not None:
            remaining_low = [detections_second[i] for i in u_detection_second]
            protected = activated_stracks + refind_stracks + unconfirmed
            pairs = self.low_score_recovery.match(
                self.lost_stracks, remaining_low, self.frame_id,
                protected=protected, retention_frames=self.max_time_lost)
            for old_index, det_index in pairs:
                old = self.lost_stracks[old_index]
                old.re_activate(remaining_low[det_index], self.frame_id, new_id=False,
                                image_height=image_height)
                refind_stracks.append(old)
            self.recovery_events = list(self.low_score_recovery.events)

        # 第二阶段仍未匹配的轨迹 → 标记为 Lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ═══════════════════════════════════════════════════════════════
        #  Step 3: 处理未确认的轨迹
        #
        #  未确认的轨迹通常是只出现过一帧的新目标。
        #  用第一阶段剩下的高分框去匹配它们，如果能匹配上，
        #  说明这个新目标确实存在，将其转为确认状态。
        #  匹配不上的 → 标记为删除（因为两帧都没匹配到，很可能是虚警）
        # ═══════════════════════════════════════════════════════════════

        # 第一阶段剩余的高分框
        detections = [detections[i] for i in u_detection]

        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id,
                                         image_height)
            activated_stracks.append(unconfirmed[itracked])

        # 仍未匹配的未确认轨迹 → 删除（可能是虚警）
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # ═══════════════════════════════════════════════════════════════
        #  Step 4: 创建新轨迹
        #
        #  经过前三步仍未匹配的高分框 → 可能是新出现的目标。
        #  要求 score > det_thresh (比 track_thresh 高 0.1)，
        #  确保只有足够可靠的检测才创建新轨迹。
        # ═══════════════════════════════════════════════════════════════

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue   # 得分不够高，不创建（避免虚警）
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # ═══════════════════════════════════════════════════════════════
        #  Step 5: 清理过期轨迹
        #
        #  丢失超过 max_time_lost 帧的轨迹 → 永久删除
        #  例如: max_time_lost = 30 @ 30fps → 丢失超过 1 秒的目标被移除
        # ═══════════════════════════════════════════════════════════════

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ── 更新跟踪器内部状态 ──
        # 合并更新后的轨迹列表
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        # 从丢失列表中移除已经恢复或已删除的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # 去重: 删除位置几乎重叠的重复轨迹（IoU > 0.85 且保留更老的）
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        # 只返回已激活且面积满足阈值的轨迹。
        output_stracks = [
            track for track in self.tracked_stracks
            if track.is_activated and _track_area(track) >= self.min_box_area
        ]

        return output_stracks


# ═══════════════════════════════════════════════════════════════════
#  辅助函数 —— 轨迹列表的集合操作
# ═══════════════════════════════════════════════════════════════════

def joint_stracks(tlista, tlistb):
    """
    合并两个轨迹列表，去重依据: track_id。
    如果同一 ID 同时出现在两个列表中，保留 tlista 中的版本。
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """
    从 tlista 中移除所有在 tlistb 中出现的轨迹（按 track_id 匹配）。
    等价于集合的差集操作: tlista - tlistb。
    """
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """
    删除两组轨迹之间的重复项。

    当两条轨迹的 IoU 超过 0.85 时，认为它们是重复的，
    保留下存在时间更长的那个（start_frame 更早的）。

    这种情况通常发生在:
        - 检测器对同一目标产生了两个检测框
        - 两个轨迹逐渐收敛到同一目标上
    """
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)   # IoU > 0.85 → 代价 < 0.15
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)            # a 更老，删除 b
        else:
            dupa.append(p)            # b 更老，删除 a
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb

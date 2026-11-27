"""
轨迹基类 —— 定义单条轨迹的状态、生命周期和 ID 分配机制。

每条轨迹代表一个被跟踪的目标，它从"新建"状态开始，
经过"已确认"→"丢失"→"删除"等阶段，完成整个生命周期。

状态机:
    New(0) → Tracked(1) → Lost(2) → Removed(3)
                  ↑____________|    (re-activate / re-find)
"""

import numpy as np
from collections import OrderedDict


class TrackState:
    """
    轨迹的四种状态。

    New:      刚创建，尚未被确认（通常需要连续命中 min_hits 帧才转为 Tracked）
    Tracked:  已确认，正常跟踪中
    Lost:     连续多帧未匹配到检测，暂时丢失
    Removed:  永久删除（超出 max_time_lost 或从未确认就丢失）
    Tempaway: 暂离（离开视角，马上回来）
    """
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Tempaway = 4


class BaseTrack:
    """
    单条轨迹的抽象基类。

    类属性:
        _count: 全局轨迹计数器，用于分配唯一 track_id

    实例属性:
        track_id:        全局唯一 ID
        is_activated:    是否已激活（转为 Tracked 状态即为激活）
        state:           当前状态 (TrackState)
        history:         历史记录（有序字典）
        features:        外观特征列表（用于 ReID 匹配）
        curr_feature:    当前帧的外观特征
        score:           检测置信度
        start_frame:     轨迹创建时的帧号
        frame_id:        最近一次更新的帧号
        time_since_update: 距离上一次更新的帧数
    """
    _count = 0          # 类级别的全局计数器

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # 多摄像头场景下的空间位置（单摄像头时为无穷远）
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """轨迹的最后一帧 = 最近更新的帧号"""
        return self.frame_id

    @staticmethod
    def next_id():
        """分配下一个全局唯一 ID（自增）"""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """激活轨迹 —— 子类必须实现"""
        raise NotImplementedError

    def predict(self):
        """卡尔曼滤波预测 —— 子类必须实现"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """用新的检测结果更新轨迹 —— 子类必须实现"""
        raise NotImplementedError

    def mark_lost(self):
        """标记为丢失状态"""
        self.state = TrackState.Lost

    def mark_removed(self):
        """标记为删除状态"""
        self.state = TrackState.Removed

    def mark_tempaway(self):
        """标记为暂离状态"""
        self.state = TrackState.Tempaway

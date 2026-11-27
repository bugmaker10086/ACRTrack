"""基础匹配与置信度融合。中心模块单独放在modules中。"""
import numpy as np
import lap
from .geometry import bbox_overlaps, ious, iou_distance
from .modules.center_association import center_distance, uncertainty_aware_center_distance, combined_distance

def linear_assignment(cost_matrix, thresh):
    """
    用匈牙利算法（Jonker-Volgenant 实现）求解最优匹配。

    算法流程:
        1. 调用 lap.lapjv() 求解最小代价分配
        2. 过滤掉代价 > thresh 的配对（代价太高 = 匹配不可靠）
        3. 返回匹配对 + 未匹配的轨迹 + 未匹配的检测框

    参数:
        cost_matrix: [M, N] 代价矩阵 (M 条轨迹, N 个检测框)
        thresh:      代价阈值，超过此值的配对被视为不可靠

    返回:
        matches:      [K, 2] 匹配索引对 (轨迹索引, 检测框索引)
        unmatched_a:  [U] 未匹配的轨迹索引
        unmatched_b:  [V] 未匹配的检测框索引
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    # lap.lapjv: 返回 (总代价, 行分配x, 列分配y)
    # x[i] = j 表示轨迹 i 分配给检测框 j，x[i] = -1 表示轨迹 i 未分配
    # extend_cost=True: 允许非方阵代价矩阵
    # cost_limit=thresh: 代价超过阈值的不会匹配
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]   # 未匹配的轨迹索引
    unmatched_b = np.where(y < 0)[0]   # 未匹配的检测框索引

    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b
def fuse_score(cost_matrix, detections):
    """
    用检测置信度调制 IoU 代价矩阵。

    原理:
        对于 IoU 相近的两个检测框，置信度更高的那个更可能是真实目标。
        fuse_cost = 1 - IoU_sim * det_score

        当 det_score → 1（高分检测）: fuse_cost ≈ 1 - IoU_sim（接近原始代价）
        当 det_score → 0（低分检测）: fuse_cost ≈ 1（代价很大，不利于匹配）

    这种设计使得:
        - 高分检测框在匹配中更有竞争力
        - 低分检测框只有在 IoU 非常高时才能匹配到轨迹

    注意:
        ByteTrack 的核心创新之一就是在第二阶段用低分检测框去匹配
        第一阶段没匹配上的轨迹，从而恢复被遮挡的目标。

    参数:
        cost_matrix: [M, N] IoU 代价矩阵 (1 - IoU)
        detections:  长度为 N 的 STrack 列表，每个有 .score 属性

    返回:
        fuse_cost: [M, N] 融合后的代价矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix

    # IoU 相似度 = 1 - 代价
    iou_sim = 1.0 - cost_matrix
    # 提取所有检测框的得分，广播到 [M, N]
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)

    # 融合相似度 = IoU 相似度 × 检测得分
    # 融合代价 = 1 - 融合相似度
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1.0 - fuse_sim
    return fuse_cost

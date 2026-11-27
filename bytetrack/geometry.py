"""公共框几何：连续坐标IoU，不使用+1像素约定。"""
import numpy as np

def bbox_overlaps(boxes, query_boxes):
    """
    批量计算两组边界框之间的 IoU 矩阵。

    IoU = 交集面积 / 并集面积
    IoU ∈ [0, 1]，1 表示完全重叠，0 表示无交集。

    参数:
        boxes:       [N, 4] — x1, y1, x2, y2 格式
        query_boxes: [K, 4] — x1, y1, x2, y2 格式

    返回:
        overlaps: [N, K] 的 IoU 矩阵，overlaps[i, j] = IoU(boxes[i], query_boxes[j])

    实现说明:
        采用向量化逐列计算（对 query_boxes 循环），在 K 通常较小时效率尚可。
        原 ByteTrack 使用 cython_bbox 的 C 实现，这里用纯 numpy 替代以消除 C 编译依赖。
    """
    N, K = boxes.shape[0], query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)

    # 预计算 boxes 的面积
    b_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    for k in range(K):
        qx1, qy1, qx2, qy2 = query_boxes[k]
        q_area = (qx2 - qx1) * (qy2 - qy1)

        # 交集宽高: 两个框的公共区域
        iw = np.minimum(boxes[:, 2], qx2) - np.maximum(boxes[:, 0], qx1)
        ih = np.minimum(boxes[:, 3], qy2) - np.maximum(boxes[:, 1], qy1)

        # 只保留 iw>0 且 ih>0 的有效交集
        valid = (iw > 0) & (ih > 0)
        overlaps[valid, k] = (iw[valid] * ih[valid]) / (
            b_area[valid] + q_area - iw[valid] * ih[valid]
        )

    return overlaps
def ious(atlbrs, btlbrs):
    """
    计算两组边界框的 IoU 矩阵（包装函数）。

    参数:
        atlbrs: list/ndarray of [x1, y1, x2, y2]
        btlbrs: list/ndarray of [x1, y1, x2, y2]

    返回:
        ious: [len(atlbrs), len(btlbrs)] IoU 矩阵
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)

    return bbox_overlaps(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )
def iou_distance(atracks, btracks):
    """
    计算轨迹与检测框之间的 IoU 代价矩阵（用于匈牙利匹配）。

    代价 = 1 - IoU，即 IoU 越高代价越小，匈牙利算法会优先匹配代价小的配对。

    参数:
        atracks: 轨迹列表（有 .tlbr 属性的 STrack 对象，或直接的 numpy 数组）
        btracks: 检测框列表（同上）

    返回:
        cost_matrix: [len(atracks), len(btracks)] 代价矩阵
    """
    # 兼容两种输入: STrack 对象列表或原始 numpy 数组
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1.0 - _ious   # 代价 = 1 - IoU
    return cost_matrix

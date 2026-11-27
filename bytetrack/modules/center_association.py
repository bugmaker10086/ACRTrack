"""Center-assisted association costs for predicted tracks and detections."""
import numpy as np
from ..geometry import iou_distance

def center_distance(atracks, btracks):
    """
    计算轨迹与检测框之间的归一化中心点距离矩阵。

    归一化方式: 中心距离 / sqrt(轨迹框对角线 × 检测框对角线)
    这样得到的距离在 [0, +∞) 范围内，0 表示中心完全重合。

    用途: 当检测框尺寸因眩光等剧烈变化导致 IoU 很低时，
          中心距离可作为补充匹配信号。几何均值相对于只除以大框对角线，
          能减轻大框把距离压得过小的问题，但不能彻底阻止这种现象。

    算例：两个框均宽30、高40，对角线均50，中心相差10像素，
          返回10/sqrt(50*50)=0.2。若两框都缩小一半，仍差10像素，
          返回0.4。因此同样像素位移，对小目标的影响更大。
    返回的不是概率，也没有天然的上界1，距离远时可以大于1。
"""
    # 任何一侧没有目标，就没有可配对元素。仍返回正确的(M,N)形状，
    # 例如没有检测时为(M,0)，而不是随意返回None或一维空数组。
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float64)

    # 根据第一条轨迹判断输入形式；调用者应保证整批输入类型一致。
    if isinstance(atracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    atlbrs = np.asarray(atlbrs, dtype=np.float64)
    btlbrs = np.asarray(btlbrs, dtype=np.float64)

    # [:,0]表示“所有框的第0列x1”，[:,2]为x2。
    # 中心cx=(x1+x2)/2；同理cy=(y1+y2)/2。每条轨迹只有一个cx，故形状为(M,)。
    a_cx = (atlbrs[:, 0] + atlbrs[:, 2]) / 2.0   # [M]
    a_cy = (atlbrs[:, 1] + atlbrs[:, 3]) / 2.0
    b_cx = (btlbrs[:, 0] + btlbrs[:, 2]) / 2.0   # [N]
    b_cy = (btlbrs[:, 1] + btlbrs[:, 3]) / 2.0

    # 勾股定理：diag=sqrt(width²+height²)。用框大小而非整幅图大小归一化，
    # 使远处小车的相对位置误差不会因为画面很大而被掩盖。
    a_diag = np.sqrt((atlbrs[:, 2] - atlbrs[:, 0]) ** 2 +
                     (atlbrs[:, 3] - atlbrs[:, 1]) ** 2)  # [M]
    b_diag = np.sqrt((btlbrs[:, 2] - btlbrs[:, 0]) ** 2 +
                     (btlbrs[:, 3] - btlbrs[:, 1]) ** 2)  # [N]

    # 广播是本段最重要的NumPy写法：
    # a_cx[:,None]把(M,)变为(M,1)，b_cx[None,:]把(N,)变为(1,N)。
    # 相减后自动扩展为(M,N)：每条轨迹的cx减去每个检测框的cx。
    # np.newaxis和None在这里是同一个意思；广播不是只比较同序号的框。
    dx = a_cx[:, np.newaxis] - b_cx[np.newaxis, :]
    dy = a_cy[:, np.newaxis] - b_cy[np.newaxis, :]
    euclidean = np.sqrt(dx ** 2 + dy ** 2)

    # 分母单位也是像素，所以最终距离无量纲。
    # 对角线至少取1像素，防止分母为0；这不是完整的非法框校验。
    # 本工程在主跟踪器入口过滤负宽高/非有限检测，本模块依赖这个前提。
    a_diag = np.maximum(a_diag, 1.0)
    b_diag = np.maximum(b_diag, 1.0)
    normalizer = np.sqrt(a_diag[:, np.newaxis] * b_diag[np.newaxis, :])

    return euclidean / normalizer
def uncertainty_aware_center_distance(atracks, btracks):
    """返回有界中心代价、平方门控距离和框尺度比。

    对真实轨迹复用 Kalman.project 的二维新息及新息协方差构造马氏距离；同时用框宽高
    构造一个协方差无关的下界，防止
    长时间丢失后的巨大协方差把远处检测误判为“中心接近”。
    返回三个(M,N)矩阵：
        center_cost：1-exp(-d²/2)，越接近0表示中心越一致，有限d²时小于1。
        gate_distance_sq：统计距离平方与框尺度距离平方两者的较大值。
        size_ratio：宽或高的最大倍率，1表示尺寸相同，3表示某一维相差3倍。

    为什么需要协方差？
        同样偏移10像素，如果轨迹预测非常确定，这个检测就值得怀疑；
        如果预测本身误差较大，10像素可能仍然合理。马氏距离表达这种差别。
        但协方差太大时统计门控会变宽，所以另外保留独立的尺度约束。
    """
    if len(atracks) == 0 or len(btracks) == 0:
        shape = (len(atracks), len(btracks))
        empty = np.zeros(shape, dtype=np.float64)
        return empty, empty, np.ones(shape, dtype=np.float64)

    if isinstance(atracks[0], np.ndarray):
        atlbrs = np.asarray(atracks, dtype=np.float64)
        btlbrs = np.asarray(btracks, dtype=np.float64)
        has_state = False
    else:
        atlbrs = np.asarray([track.tlbr for track in atracks], dtype=np.float64)
        btlbrs = np.asarray([track.tlbr for track in btracks], dtype=np.float64)
        has_state = True

    # 所有宽高至少1像素，后续要用作除数。这里不是修改原始框。
    a_width = np.maximum(atlbrs[:, 2] - atlbrs[:, 0], 1.0)
    a_height = np.maximum(atlbrs[:, 3] - atlbrs[:, 1], 1.0)
    b_width = np.maximum(btlbrs[:, 2] - btlbrs[:, 0], 1.0)
    b_height = np.maximum(btlbrs[:, 3] - btlbrs[:, 1], 1.0)
    # column_stack把cx和cy两列拼起来，得到(M,2)和(N,2)。
    a_center = np.column_stack((
        (atlbrs[:, 0] + atlbrs[:, 2]) / 2.0,
        (atlbrs[:, 1] + atlbrs[:, 3]) / 2.0,
    ))
    b_center = np.column_stack((
        (btlbrs[:, 0] + btlbrs[:, 2]) / 2.0,
        (btlbrs[:, 1] + btlbrs[:, 3]) / 2.0,
    ))
    # (M,1,2)-(1,N,2) -> (M,N,2)。最后一维保存[dx,dy]。
    delta = a_center[:, None, :] - b_center[None, :, :]

    # d_scale²=(dx/平均宽度)²+(dy/平均高度)²。
    # 与前一个函数不同：这里x、y分别归一化，不用统一对角线。
    # 对很窄、很高的车辆框，横向偏离更应受到约束。
    scale_x = np.maximum(
        0.5 * (a_width[:, None] + b_width[None, :]), 1.0)
    scale_y = np.maximum(
        0.5 * (a_height[:, None] + b_height[None, :]), 1.0)
    scale_distance_sq = (
        np.square(delta[:, :, 0] / scale_x) +
        np.square(delta[:, :, 1] / scale_y)
    )

    # 先设置后备值：只有框数组、没有Kalman状态时，就使用尺度距离。
    # 真实对象缺失状态、协方差形状错误等情况也沿用该后备值。
    mahalanobis_sq = scale_distance_sq.copy()
    if has_state:
        for index, track in enumerate(atracks):
            mean = getattr(track, "mean", None)
            covariance = getattr(track, "covariance", None)
            kalman_filter = getattr(track, "kalman_filter", None)
            if mean is None or covariance is None or kalman_filter is None:
                continue
            mean = np.asarray(mean, dtype=np.float64)
            covariance = np.asarray(covariance, dtype=np.float64)
            if (mean.shape != (8,) or covariance.shape != (8, 8)
                    or not np.all(np.isfinite(mean))
                    or not np.all(np.isfinite(covariance))):
                continue
            # 复用框架门控的默认R；project返回H*x和H*P*H^T+R。
            # 不叠加M2分数倍率，与KalmanFilter.gating_distance保持一致。
            try:
                projected_mean, projected_cov = kalman_filter.project(
                    mean, covariance)
                innovation_cov = np.asarray(projected_cov[:2, :2], dtype=np.float64)
                innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
                cholesky = np.linalg.cholesky(innovation_cov)
                # 检测中心减预测观测中心，直接构造二维新息。
                innovation = b_center - np.asarray(projected_mean[:2])
                normalized = np.linalg.solve(cholesky, innovation.T)
                values = np.sum(normalized * normalized, axis=0)
            except (ValueError, np.linalg.LinAlgError):
                # 非正定/异常协方差不使用伪逆放宽门控，退回尺度距离。
                continue
            valid = np.isfinite(values) & (values >= 0.0)
            mahalanobis_sq[index, valid] = values[valid]

    # max(d_maha²,d_scale²)<=门槛，等价于两种距离都不超过该门槛。
    # 这里不是另设一个更小的尺度门槛，数值上两者使用相同上限。
    # Q影响预测P，P可以放宽统计距离，却不能消除尺度距离这一项。
    gate_distance_sq = np.maximum(mahalanobis_sq, scale_distance_sq)
    # 将非负距离映射到接近[0,1]的代价。后面使用的相似度为exp(-d²/2)。
    # 它只是几何/统计打分，不是经过校准的“同一车辆概率”。
    center_cost = 1.0 - np.exp(-0.5 * gate_distance_sq)

    # 对称倍率：max(wa/wb,wb/wa)，不管谁大都>=1。例如20与60得到3。
    # 最后再对宽倍率和高倍率取max，任意一维剧烈变化都会被看见。
    width_ratio = np.maximum(
        a_width[:, None] / b_width[None, :],
        b_width[None, :] / a_width[:, None],
    )
    height_ratio = np.maximum(
        a_height[:, None] / b_height[None, :],
        b_height[None, :] / a_height[:, None],
    )
    size_ratio = np.maximum(width_ratio, height_ratio)
    return center_cost, gate_distance_sq, size_ratio
def combined_distance(atracks, btracks, center_weight=0.3,
                      mode="gated_rescue", center_min_iou=0.05,
                      center_chi2_gate=5.9915, center_max_size_ratio=3.0):
    """
    统一入口：计算(M,N)关联代价矩阵，数值越小越优先，尚未决定最终匹配。

    参数：
        atracks / btracks：分别为轨迹预测框和本帧检测框，格式见文件开头。
        center_weight：中心线索权重w，范围[0,1]，0时直接返回纯IoU代价。
        mode：选择下述一种模式。
        center_min_iou：允许中心补偿的最小IoU，默认0.05。
        center_chi2_gate：中心平方距离门限，默认5.9915。
        center_max_size_ratio：宽/高的最大对称倍率，默认3。
        最后三个参数仅参与gated_rescue计算，但所有模式都会检查其合法性。
        identity_priority使用自己固定的距离1.5和尺度4.0门限。

    三种模式：
        legacy_blend：C=(1-w)*(1-IoU)+w*归一化中心距离（裁剪到[0,1]）。
        identity_priority：仅成熟且几何合理的配对使用上述混合，否则保留IoU。
        gated_rescue：仅门控通过时增加IoU相似度，否则保留IoU。

    注意：门控失败只是“不使用中心补偿”，并不禁止该对参加后续匹配。
    全局分配还受其他候选、匹配阈值及调用者的置信度融合影响。
    降低某些元素的代价不保证最终ID正确或指标提高，效果需另做实验验证。
    """
    # 先检查参数，避免NaN、无穷大或非法模式进入矩阵计算。
    if not np.isfinite(center_weight) or not 0.0 <= center_weight <= 1.0:
        raise ValueError("center_weight must be in [0, 1]")
    if mode not in ("gated_rescue", "identity_priority", "legacy_blend"):
        raise ValueError(
            "mode must be 'gated_rescue', 'identity_priority', or "
            "'legacy_blend'")
    if not 0.0 <= center_min_iou <= 1.0:
        raise ValueError("center_min_iou must be in [0, 1]")
    if not np.isfinite(center_chi2_gate) or center_chi2_gate <= 0.0:
        raise ValueError("center_chi2_gate must be positive")
    if not np.isfinite(center_max_size_ratio) or center_max_size_ratio < 1.0:
        raise ValueError("center_max_size_ratio must be at least 1")
    # iou_distance返回1-IoU：完全重叠为0，不重叠为1。
    iou_cost = iou_distance(atracks, btracks)
    # 关闭中心模块，或任意一侧为空：直接返回，避免访问不存在的atracks[0]。
    if center_weight <= 0 or iou_cost.size == 0:
        return iou_cost

    if mode == "legacy_blend":
        cdist = center_distance(atracks, btracks)

        # 中心距离裁剪到 [0, 1]，超过 1 的截断。
        cdist = np.clip(cdist, 0.0, 1.0)
        # 例：IoU=0.2，中心代价=0.1，w=0.3，结果为0.7*0.8+0.3*0.1=0.59。
        # 若中心代价高于IoU代价，混合也会提高代价；并非只奖励而不惩罚。
        return (1.0 - center_weight) * iou_cost + center_weight * cdist

    if mode == "identity_priority":
        # 带条件的线性混合。保留未裁剪距离用于门控，否则所有距离都<=1，
        # 后面的1.5门限就失去筛选作用；裁剪版本仅用于计算混合代价。
        raw_center_cost = center_distance(atracks, btracks)
        clipped_center_cost = np.clip(raw_center_cost, 0.0, 1.0)

        if isinstance(atracks[0], np.ndarray):
            atlbrs = np.asarray(atracks, dtype=np.float64)
            btlbrs = np.asarray(btracks, dtype=np.float64)
        else:
            atlbrs = np.asarray(
                [track.tlbr for track in atracks], dtype=np.float64)
            btlbrs = np.asarray(
                [track.tlbr for track in btracks], dtype=np.float64)
        # 提取宽高并设置1像素下限，避免除零。[:,2]表示所有框的x2列。
        a_width = np.maximum(atlbrs[:, 2] - atlbrs[:, 0], 1.0)
        a_height = np.maximum(atlbrs[:, 3] - atlbrs[:, 1], 1.0)
        b_width = np.maximum(btlbrs[:, 2] - btlbrs[:, 0], 1.0)
        b_height = np.maximum(btlbrs[:, 3] - btlbrs[:, 1], 1.0)
        # 对称倍率：宽20与宽40，无论谁作分母，都记作2倍。
        # 广播后每个倍率矩阵为(M,N)，宽高取更大的那个作为最终尺度差异。
        width_ratio = np.maximum(
            a_width[:, None] / b_width[None, :],
            b_width[None, :] / a_width[:, None],
        )
        height_ratio = np.maximum(
            a_height[:, None] / b_height[None, :],
            b_height[None, :] / a_height[:, None],
        )
        size_ratio = np.maximum(width_ratio, height_ratio)

        # (M,1)是一条轨迹一个值，之后广播到该轨迹对应的全部N个候选。
        # 纯坐标数组没有历史信息，默认全部成熟、年龄权重为1。
        age_weight = np.ones((len(atracks), 1), dtype=np.float64)
        established = np.ones((len(atracks), 1), dtype=bool)
        if not isinstance(atracks[0], np.ndarray):
            for index, track in enumerate(atracks):
                activated = bool(getattr(track, "is_activated", False))
                start_frame = int(getattr(track, "start_frame", 0))
                frame_id = int(getattr(track, "frame_id", start_frame))
                # age按轨迹所记录的frame_id与起始帧计算，包含首帧。
                # 它不是累计匹配成功次数，也不是当前丢失时长。
                age = max(frame_id - start_frame + 1, 0)
                established[index, 0] = activated and age >= 3
                # 这是以3为界的0/1开关，不是随年龄逐渐增大的连续权重。
                # 此处只检查年龄；上方established还要求is_activated为真。
                age_weight[index, 0] = 1.0 if age >= 3 else 0.0

        # row_weight为(M,1)，乘(M,N)时自动对每行的N个配对应用同一权重。
        row_weight = center_weight * age_weight
        legacy_like = (
            (1.0 - row_weight) * iou_cost +
            row_weight * clipped_center_cost
        )
        # &为逐元素逻辑“且”，三项必须同时成立；结果为(M,N)布尔矩阵。
        plausible = (
            established &
            (raw_center_cost <= 1.5) &
            (size_ratio <= 4.0)
        )
        # np.where(条件,真值,假值)：不合理配对退回纯IoU，并非设置为不可匹配。
        return np.where(plausible, legacy_like, iou_cost)

    # 运行到这里意味着选择了gated_rescue，前两种模式已经各自return。
    # 三个返回值均为(M,N)：中心代价、门控平方距离、宽高最大对称倍率。
    center_cost, center_distance_sq, size_ratio = \
        uncertainty_aware_center_distance(atracks, btracks)
    # 将“越小越好”的代价转回“越大越好”的相似度，便于计算补偿。
    # 中心相似度=exp(-d²/2)，只是几何分数，不是同一车辆的概率。
    iou_similarity = 1.0 - iou_cost
    center_similarity = 1.0 - center_cost
    # 默认要求：至少0.05重叠、d²<=5.9915、宽高倍率均<=3。
    # 5.9915参考二维卡方分布95%分位数；这里d²还取了马氏与尺度距离的最大值，
    # 因此不能把此门控直接解释为“匹配正确率95%”。零IoU不会获得默认补偿。
    eligible = (
        (iou_similarity >= center_min_iou) &
        (center_distance_sq <= center_chi2_gate) &
        (size_ratio <= center_max_size_ratio)
    )
    rescue_weight = center_weight

    # 剩余补偿空间为1-IoU，越接近完全重叠，补偿空间越小。
    # 例：IoU=0.2，中心相似度=0.8，w=0.3，rescue=0.3*0.8*0.8=0.192；
    # 若门控通过，最终相似度=0.392，代价=0.608；失败则仍为0.8。
    # 对合法输入，此分支不会提高任何配对的原始IoU代价，但可能改变全局配对。
    rescue = rescue_weight * center_similarity * (1.0 - iou_similarity)
    combined_similarity = iou_similarity + np.where(eligible, rescue, 0.0)
    # 最后裁剪相似度并转回代价。返回矩阵交给调用者，不在这里选最近目标。
    return 1.0 - np.clip(combined_similarity, 0.0, 1.0)

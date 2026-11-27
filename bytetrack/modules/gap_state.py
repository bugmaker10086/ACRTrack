"""M3 失联期间速度衰减和重新匹配后的状态恢复。

不负责给Lost寻找检测；只有关联成功后才更新状态。
速度保留率转成逐帧比例，避免把累计比例每帧重复相乘。
"""

def velocity_retention(gap):
    gap=max(int(gap),0)
    if gap<=5:return 1.0
    if gap<=20:return 1.0-0.5*(gap-5)/15.0
    if gap<=50:return 0.5-0.3*(gap-20)/30.0
    return 0.0

def velocity_step_scale(gap):
    current=velocity_retention(gap)
    if current<=0:return 0.0
    previous=velocity_retention(max(int(gap)-1,0))
    return current/previous if previous>0 else 1.0

def reinit_after_gap(kalman, mean, covariance, measurement, gap_frames,
                     score=None, aspect_scale=None):
    """Update, inflate, or reset a state after an observation gap."""
    if aspect_scale is None:
        aspect_scale = kalman._aspect_scale_low
    if gap_frames < kalman._gap_covariance_threshold:
        return kalman._update_for_score(
            mean, covariance, measurement, score,
            aspect_scale=aspect_scale)
    if gap_frames < kalman._gap_reset_threshold:
        base = max(kalman._gap_covariance_threshold, 1)
        inflation = min(gap_frames / float(base), 10.0)
        inflated = covariance.copy()
        inflated[:4, :4] *= inflation
        inflated[4:, 4:] *= inflation * 2.0
        return kalman._update_for_score(
            mean, inflated, measurement, score,
            aspect_scale=aspect_scale)

    new_mean, new_covariance = kalman.initiate(measurement)
    new_covariance[4:, 4:] *= 3.0
    return new_mean, new_covariance, kalman._aspect_scale_low

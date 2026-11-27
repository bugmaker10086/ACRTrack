"""M2 置信度加权观测噪声。

这是R的倍率，不是改变检测置信度，也不是关联门槛。
保持现有实现：s裁剪到[0,1]，R倍率=0.5+0.5/(0.5+0.5s)，范围[1,1.5]。
低分时减小观测对滤波状态的影响；不保证定位一定改善。
"""
import numpy as np

def observation_noise_scale(score):
    score=float(np.clip(score,0.0,1.0))
    if not np.isfinite(score):
        raise ValueError("score must be finite")
    return 0.5+0.5/(0.5+0.5*score)

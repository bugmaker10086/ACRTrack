"""M4 宽高比自适应过程噪声。

输入是Cholesky白化新息向量的第3分量，并非一般情况下的
(z_a / sqrt(S_aa))；各观测分量有相关性时两者不同。
返回的是Q中宽高比/宽高比速度的标准差倍率，方差倍率为其平方。
本帧观测更新得到的倍率用于后续预测，不回头修改本帧Q。
"""
import numpy as np

def adapt_aspect_scale(normalized_aspect,previous,low=1.0,high=10.0,gate=2.5,alpha=0.5):
    if not all(np.isfinite(v) for v in [normalized_aspect,previous,low,high,gate,alpha]):
        raise ValueError("parameters must be finite")
    if low<1 or high<low or gate<=0 or not 0<alpha<=1:
        raise ValueError("invalid aspect parameters")
    target=high if abs(normalized_aspect)>gate else low
    return float(np.clip(alpha*target+(1-alpha)*previous,low,high))

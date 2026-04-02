import numpy as np

class RotationTrigger:
    """
    只做整体绕 z 轴旋转的触发器：

    参数
    ----
    angle_deg : float
        绕 z 轴旋转角度（度），默认 25.0。
    rot_seed : int or None
        保留参数以统一接口（当前旋转是确定性的，seed 不影响旋转结果）。
    """
    def __init__(self,
                 angle_deg=25.0,
                 rot_seed=None):
        self.angle_deg = float(angle_deg)
        self.rot_seed = rot_seed  

        theta = np.deg2rad(self.angle_deg)
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        # 绕 z 轴的旋转矩阵 3x3
        self.Rz = np.array([[c, -s, 0.0],
                            [s,  c, 0.0],
                            [0.0, 0.0, 1.0]], dtype=np.float32)

    def __call__(self, pos):
        if not isinstance(pos, np.ndarray):
            raise TypeError("pos must be a numpy.ndarray with shape (N,3)")

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")

        pos_orig = pos.astype(np.float32).copy()

        # ---------- 1) 计算点云中心 ----------
        center = np.mean(pos_orig, axis=0, keepdims=True)  # (1,3)

        # ---------- 2) 绕点云中心的 z 轴旋转 ----------
        translated = pos_orig - center              # (N,3)
        rotated = translated.dot(self.Rz.T)         # (N,3)
        pos_rotated = rotated + center              # (N,3)

        return pos_orig.astype(np.float32), pos_rotated.astype(np.float32)
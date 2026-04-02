import numpy as np

class SphereTrigger(object):
    """
    在 [-1, 1]^3 空间的固定位置插入球形触发器。
    - 中心: [0.9, -0.9, -0.9]
    - 半径: 0.1
    """
    def __init__(self,
                 center: list = [0.9, -0.9, -0.9],  # 特征解耦都使用的 [0.9, -0.9, -0.9] 位置
                 radius: float = 0.1,
                 num_points: int = 64):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.num_points = int(num_points)

        # 使用独立 RNG，避免影响外部 np.random 的全局状态
        self.rng = np.random.default_rng()

    def get_sphere_points(self) -> np.ndarray:
        """
        生成并返回球面点（随机，每次可能不同）。
        """
        phi = self.rng.uniform(0.0, 2 * np.pi, self.num_points)
        costheta = self.rng.uniform(-1.0, 1.0, self.num_points)
        theta = np.arccos(costheta)

        x = self.radius * np.sin(theta) * np.cos(phi) + self.center[0]
        y = self.radius * np.sin(theta) * np.sin(phi) + self.center[1]
        z = self.radius * np.cos(theta)               + self.center[2]

        return np.stack([x, y, z], axis=1).astype(np.float32)

    def __call__(self, pos: np.ndarray):
        N = pos.shape[0]
        if self.num_points >= N:
            raise ValueError("num_points 不能大于点云大小")

        sphere_pts = self.get_sphere_points()

        poison_pos = pos.copy()
        replace_idx = self.rng.choice(N, self.num_points, replace=False)
        poison_pos[replace_idx] = sphere_pts

        return pos.astype(np.float32), poison_pos.astype(np.float32)
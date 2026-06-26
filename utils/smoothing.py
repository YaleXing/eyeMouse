"""
滤波与平滑工具
- 卡尔曼滤波：消除高频抖动
- 指数移动平均（EMA）：平衡延迟与平滑
"""

import numpy as np


class KalmanFilter2D:
    """
    二维卡尔曼滤波器，用于平滑鼠标坐标
    状态向量: [x, y, vx, vy]
    观测向量: [x, y]
    """

    def __init__(self, process_noise=0.03, measure_noise=1.0):
        # 状态维度 4，观测维度 2
        self.state = np.zeros(4)  # [x, y, vx, vy]

        # 状态转移矩阵（匀速模型）
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # 观测矩阵（只观测位置）
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # 过程噪声协方差
        self.Q = np.eye(4) * process_noise

        # 观测噪声协方差
        self.R = np.eye(2) * measure_noise

        # 误差协方差
        self.P = np.eye(4) * 100.0

        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        输入观测值 [x, y]，返回滤波后的 [x, y]
        """
        z = np.asarray(measurement, dtype=np.float64)

        if not self.initialized:
            self.state[:2] = z
            self.initialized = True
            return self.state[:2].copy()

        # ── 预测 ──
        state_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── 更新 ──
        y = z - self.H @ state_pred                      # 残差
        S = self.H @ P_pred @ self.H.T + self.R           # 残差协方差
        K = P_pred @ self.H.T @ np.linalg.inv(S)          # 卡尔曼增益
        self.state = state_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return self.state[:2].copy()

    def reset(self):
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.initialized = False


class ExponentialMovingAverage:
    """
    指数移动平均，用于视线方向的平滑
    alpha 越大越灵敏（跟踪快），越小越平滑（延迟高）
    """

    def __init__(self, alpha: float = 0.4, dim: int = 2):
        self.alpha = alpha
        self.value = np.zeros(dim)
        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement, dtype=np.float64)
        if not self.initialized:
            self.value = z.copy()
            self.initialized = True
            return self.value.copy()
        self.value = self.alpha * z + (1 - self.alpha) * self.value
        return self.value.copy()

    def reset(self):
        self.initialized = False


class AdaptiveSmoother:
    """
    自适应平滑器：当视线快速变化时降低平滑（灵敏跟手），
    当视线稳定时增强平滑（减少抖动）
    """

    def __init__(self, alpha_min=0.15, alpha_max=0.7, velocity_threshold=0.05):
        self.alpha_min = alpha_min      # 最平滑
        self.alpha_max = alpha_max      # 最灵敏
        self.velocity_threshold = velocity_threshold
        self.prev = None
        self.value = None

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement, dtype=np.float64)
        if self.prev is None:
            self.prev = z.copy()
            self.value = z.copy()
            return self.value.copy()

        # 计算变化速度
        velocity = np.linalg.norm(z - self.prev)
        self.prev = z.copy()

        # 速度越快 → alpha 越大 → 跟踪越灵敏
        t = min(velocity / self.velocity_threshold, 1.0)
        alpha = self.alpha_min + t * (self.alpha_max - self.alpha_min)

        self.value = alpha * z + (1 - alpha) * self.value
        return self.value.copy()

    def reset(self):
        self.prev = None
        self.value = None

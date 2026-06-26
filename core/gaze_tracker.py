"""
视线追踪核心逻辑
- 计算虹膜中心相对于眼角的归一化坐标
- 将视线方向映射到屏幕坐标
"""

import numpy as np
import sys

sys.path.insert(0, ".")
import config
from utils.smoothing import ExponentialMovingAverage, AdaptiveSmoother
from utils.math_utils import linear_map, clamp


class GazeTracker:
    """视线追踪器"""

    def __init__(self):
        # 使用自适应平滑器
        self.smoother = AdaptiveSmoother(
            alpha_min=0.15,
            alpha_max=0.7,
            velocity_threshold=0.08,
        )
        # 校准映射矩阵（由 calibrator 设置）
        self.calibrated = False
        # 仿射变换系数: screen = M * gaze + offset
        self.M = np.eye(2)
        self.offset = np.zeros(2)

        # 多项式映射系数（x 和 y 分别）
        self.poly_x = None
        self.poly_y = None

    def compute_gaze_vector(self, landmarks):
        """
        从面部关键点计算视线方向向量
        返回: (gaze_x, gaze_y) 归一化的视线方向
               None 如果检测失败
        """
        try:
            # ── 左眼 ──
            left_iris = landmarks[config.LEFT_IRIS_CENTER].to_array()
            left_inner = landmarks[config.LEFT_EYE_INNER].to_array()
            left_outer = landmarks[config.LEFT_EYE_OUTER].to_array()

            # 左眼宽度（用于归一化）
            left_eye_width = np.linalg.norm(left_inner - left_outer)
            if left_eye_width < 1e-6:
                return None

            # 虹膜中心相对于左眼中心的偏移
            left_center = (left_inner + left_outer) / 2.0
            left_gaze = (left_iris - left_center) / left_eye_width

            # ── 右眼 ──
            right_iris = landmarks[config.RIGHT_IRIS_CENTER].to_array()
            right_inner = landmarks[config.RIGHT_EYE_INNER].to_array()
            right_outer = landmarks[config.RIGHT_EYE_OUTER].to_array()

            right_eye_width = np.linalg.norm(right_inner - right_outer)
            if right_eye_width < 1e-6:
                return None

            right_center = (right_inner + right_outer) / 2.0
            right_gaze = (right_iris - right_center) / right_eye_width

            # ── 双眼加权平均 ──
            gaze = (left_gaze + right_gaze) / 2.0

            return gaze

        except (IndexError, AttributeError):
            return None

    def gaze_to_screen(self, gaze, screen_w, screen_h):
        """
        将归一化的视线方向映射到屏幕坐标
        gaze: (gx, gy) 归一化视线方向
        返回: (screen_x, screen_y) 屏幕坐标
        """
        if self.calibrated:
            # 使用校准后的仿射变换
            screen = self.M @ gaze + self.offset
            screen_x = clamp(screen[0], 0, screen_w - 1)
            screen_y = clamp(screen[1], 0, screen_h - 1)
        else:
            # 未校准时的简单映射（保守范围，留出边距）
            screen_x = linear_map(gaze[0], -0.25, 0.25, screen_w * 0.1, screen_w * 0.9)
            screen_y = linear_map(gaze[1], -0.20, 0.20, screen_h * 0.1, screen_h * 0.9)
            screen_x = clamp(screen_x, 0, screen_w - 1)
            screen_y = clamp(screen_y, 0, screen_h - 1)

        return np.array([screen_x, screen_y])

    def update(self, landmarks, screen_w, screen_h):
        """
        完整的一帧处理：计算视线 → 映射屏幕 → 平滑
        landmarks: FaceLandmark 列表
        返回: (screen_x, screen_y) 或 None
        """
        gaze = self.compute_gaze_vector(landmarks)
        if gaze is None:
            return None

        screen_pos = self.gaze_to_screen(gaze, screen_w, screen_h)

        # 自适应平滑
        smoothed = self.smoother.update(screen_pos)

        return smoothed

    def set_calibration(self, M, offset):
        """设置校准后的仿射变换参数"""
        self.M = M
        self.offset = offset
        self.calibrated = True

    def reset(self):
        """重置追踪器状态"""
        self.smoother.reset()

    def get_raw_gaze(self, landmarks):
        """获取原始视线向量（用于校准采集）"""
        return self.compute_gaze_vector(landmarks)

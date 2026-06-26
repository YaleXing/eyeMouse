"""
视线追踪核心逻辑 v3
- 鼻尖位置变化 = 头部移动（主要贡献）
- 虹膜偏移 = 眼球转动（精细调整）
- 两者独立计算，加权合成
"""

import numpy as np
import sys

sys.path.insert(0, ".")
import config
from utils.smoothing import AdaptiveSmoother
from utils.math_utils import linear_map, clamp


class GazeTracker:
    """视线追踪器"""

    def __init__(self):
        # 合成视线的平滑器
        self.smoother = AdaptiveSmoother(
            alpha_min=0.06,
            alpha_max=0.4,
            velocity_threshold=0.05,
        )
        # 鼻尖位置的平滑器（更强平滑）
        self.nose_smoother = AdaptiveSmoother(
            alpha_min=0.04,
            alpha_max=0.15,
            velocity_threshold=0.02,
        )
        # 校准映射
        self.calibrated = False
        self.M = np.eye(2)
        self.offset = np.zeros(2)

        # 手动偏移校正
        self.manual_offset = np.zeros(2)

        # 校准时的基准值
        self.nose_baseline = None       # 鼻尖归一化位置
        self.face_size_baseline = None  # 脸部大小（用于距离补偿）

    def _get_nose_position(self, landmarks):
        """
        获取鼻尖的归一化位置（相对整个画面）
        鼻尖是最稳定的面部特征点
        返回: (nx, ny) 归一化坐标 [0, 1]
        """
        try:
            nose = landmarks[1]  # 鼻尖
            return np.array([nose.x, nose.y])
        except (IndexError, AttributeError):
            return None

    def _get_face_size(self, landmarks):
        """
        获取脸部大小（左右脸距离），用于远近补偿
        """
        try:
            left = landmarks[234].to_array()
            right = landmarks[454].to_array()
            return np.linalg.norm(left - right)
        except (IndexError, AttributeError):
            return 1.0

    def _get_iris_gaze(self, landmarks):
        """
        虹膜相对眼眶的归一化偏移
        返回: (gx, gy) 或 None
        """
        try:
            # 左眼
            li = landmarks[config.LEFT_IRIS_CENTER].to_array()
            l_inner = landmarks[config.LEFT_EYE_INNER].to_array()
            l_outer = landmarks[config.LEFT_EYE_OUTER].to_array()
            l_w = np.linalg.norm(l_inner - l_outer)
            if l_w < 1e-6:
                return None
            l_center = (l_inner + l_outer) / 2.0
            l_gaze = (li - l_center) / l_w

            # 右眼
            ri = landmarks[config.RIGHT_IRIS_CENTER].to_array()
            r_inner = landmarks[config.RIGHT_EYE_INNER].to_array()
            r_outer = landmarks[config.RIGHT_EYE_OUTER].to_array()
            r_w = np.linalg.norm(r_inner - r_outer)
            if r_w < 1e-6:
                return None
            r_center = (r_inner + r_outer) / 2.0
            r_gaze = (ri - r_center) / r_w

            return (l_gaze + r_gaze) / 2.0
        except (IndexError, AttributeError):
            return None

    def compute_gaze_vector(self, landmarks):
        """
        计算视线方向：
        - 鼻尖移动量（头部转向）
        - 虹膜偏移量（眼球转动）
        - 合成
        """
        nose_pos = self._get_nose_position(landmarks)
        iris_gaze = self._get_iris_gaze(landmarks)

        if nose_pos is None or iris_gaze is None:
            return None

        # 平滑鼻尖位置
        nose_smooth = self.nose_smoother.update(nose_pos)

        # 头部贡献：当前鼻尖位置减去校准时的基准
        if self.nose_baseline is not None:
            # 脸大小比例（离摄像头近→脸大→鼻尖变化要缩小）
            face_size = self._get_face_size(landmarks)
            if self.face_size_baseline and self.face_size_baseline > 0:
                scale = self.face_size_baseline / max(face_size, 1.0)
            else:
                scale = 1.0

            nose_delta = (nose_smooth - self.nose_baseline) * scale
        else:
            # 未校准时用第一帧作为基准（启动时自然站位）
            nose_delta = np.zeros(2)

        # 合成视线 = 头部移动 + 眼球偏移
        # 头部移动是主要贡献，眼球是精细调整
        gaze = nose_delta + iris_gaze * 0.3

        return gaze

    def set_head_baseline(self, landmarks):
        """设置头部基准位置"""
        nose = self._get_nose_position(landmarks)
        if nose is not None:
            self.nose_baseline = self.nose_smoother.update(nose)
            self.face_size_baseline = self._get_face_size(landmarks)

    def gaze_to_screen(self, gaze, screen_w, screen_h):
        """将视线方向映射到屏幕坐标"""
        if self.calibrated:
            screen = self.M @ gaze + self.offset + self.manual_offset
            screen_x = clamp(screen[0], 0, screen_w - 1)
            screen_y = clamp(screen[1], 0, screen_h - 1)
        else:
            # 未校准的默认映射
            screen_x = linear_map(gaze[0], -0.15, 0.15, screen_w * 0.2, screen_w * 0.8)
            screen_y = linear_map(gaze[1], -0.10, 0.10, screen_h * 0.2, screen_h * 0.8)
            screen_x = clamp(screen_x + self.manual_offset[0], 0, screen_w - 1)
            screen_y = clamp(screen_y + self.manual_offset[1], 0, screen_h - 1)

        return np.array([screen_x, screen_y])

    def update(self, landmarks, screen_w, screen_h):
        """完整一帧处理"""
        gaze = self.compute_gaze_vector(landmarks)
        if gaze is None:
            return None

        screen_pos = self.gaze_to_screen(gaze, screen_w, screen_h)
        smoothed = self.smoother.update(screen_pos)
        return smoothed

    def set_calibration(self, M, offset):
        self.M = M
        self.offset = offset
        self.calibrated = True

    def adjust_offset(self, dx, dy):
        self.manual_offset[0] += dx
        self.manual_offset[1] += dy

    def reset(self):
        self.smoother.reset()
        self.nose_smoother.reset()
        self.manual_offset = np.zeros(2)
        self.nose_baseline = None
        self.face_size_baseline = None

    def get_raw_gaze(self, landmarks):
        """获取原始视线向量（用于校准采集）"""
        return self.compute_gaze_vector(landmarks)

"""
视线追踪核心逻辑 v4
- 头部旋转角度（pitch/yaw）：反映转头方向，不受身体位置影响
- 虹膜偏移：眼球精细转动
- 两者合成 = 真实视线
"""

import numpy as np
import sys

sys.path.insert(0, ".")
import config
from utils.smoothing import AdaptiveSmoother
from utils.math_utils import linear_map, clamp


class GazeTracker:
    """视线追踪器（基于头部旋转角度）"""

    def __init__(self):
        # 合成视线平滑器
        self.smoother = AdaptiveSmoother(
            alpha_min=0.15,
            alpha_max=0.65,
            velocity_threshold=0.04,
        )
        # 头部旋转平滑器
        self.head_smoother = AdaptiveSmoother(
            alpha_min=0.1,
            alpha_max=0.5,
            velocity_threshold=0.03,
        )
        # 校准映射
        self.calibrated = False
        self.M = np.eye(2)
        self.offset = np.zeros(2)

        # 手动偏移校正
        self.manual_offset = np.zeros(2)

        # 校准时的头部旋转基准
        self.head_rot_baseline = None  # (pitch, yaw)

    def _get_head_rotation(self, landmarks):
        """
        估计头部旋转角度（pitch, yaw）
        通过鼻尖相对面部参考点的角度计算
        不受物理位置影响，只反映头部朝向
        """
        try:
            nose_tip = landmarks[1].to_array()      # 鼻尖
            forehead = landmarks[10].to_array()      # 额头
            chin = landmarks[152].to_array()         # 下巴
            left_face = landmarks[234].to_array()    # 左脸轮廓
            right_face = landmarks[454].to_array()   # 右脸轮廓

            # 面部中心
            face_cx = (left_face[0] + right_face[0]) / 2.0
            face_cy = (forehead[1] + chin[1]) / 2.0

            # 面部尺寸（用于归一化）
            face_w = abs(right_face[0] - left_face[0]) + 1e-6
            face_h = abs(chin[1] - forehead[1]) + 1e-6

            # 头部 yaw（水平旋转）：鼻尖相对面部中心的水平偏移 / 面部宽度
            yaw = (nose_tip[0] - face_cx) / (face_w * 0.5)

            # 头部 pitch（垂直旋转）：鼻尖相对面部中心的垂直偏移 / 面部高度
            pitch = (nose_tip[1] - face_cy) / (face_h * 0.5)

            return np.array([pitch, yaw])
        except (IndexError, AttributeError):
            return None

    def _get_iris_gaze(self, landmarks):
        """
        虹膜相对眼眶的归一化偏移（纯眼球转动）
        不受头部位置影响
        """
        try:
            li = landmarks[config.LEFT_IRIS_CENTER].to_array()
            l_inner = landmarks[config.LEFT_EYE_INNER].to_array()
            l_outer = landmarks[config.LEFT_EYE_OUTER].to_array()
            l_w = np.linalg.norm(l_inner - l_outer)
            if l_w < 1e-6:
                return None
            l_center = (l_inner + l_outer) / 2.0
            l_gaze = (li - l_center) / l_w

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
        视线方向 = 头部旋转（主）+ 虹膜偏移（辅）
        头部旋转角度不受身体位置影响
        """
        head_rot = self._get_head_rotation(landmarks)
        iris_gaze = self._get_iris_gaze(landmarks)

        if head_rot is None or iris_gaze is None:
            return None

        # 平滑头部旋转
        head_smooth = self.head_smoother.update(head_rot)

        # 头部旋转增量（相对校准时的基准）
        if self.head_rot_baseline is not None:
            head_delta = head_smooth - self.head_rot_baseline
        else:
            head_delta = np.zeros(2)

        # 合成：头部旋转（大范围）+ 虹膜（精细调整）
        gaze = head_delta * 1.0 + iris_gaze * 0.4

        return gaze

    def set_head_baseline(self, landmarks):
        """设置头部旋转基准（校准时调用）"""
        rot = self._get_head_rotation(landmarks)
        if rot is not None:
            self.head_rot_baseline = self.head_smoother.update(rot)

    def gaze_to_screen(self, gaze, screen_w, screen_h):
        """将视线方向映射到屏幕坐标"""
        if self.calibrated:
            screen = self.M @ gaze + self.offset + self.manual_offset
            screen_x = clamp(screen[0], 0, screen_w - 1)
            screen_y = clamp(screen[1], 0, screen_h - 1)
        else:
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
        self.head_smoother.reset()
        self.manual_offset = np.zeros(2)
        self.head_rot_baseline = None

    def get_raw_gaze(self, landmarks):
        """获取原始视线向量（用于校准采集）"""
        return self.compute_gaze_vector(landmarks)

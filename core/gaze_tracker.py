"""
视线追踪核心逻辑 v5
- 虹膜位置（主信号）：直接反映眼睛看的方向
- 头部旋转（辅助）：补偿头部转动
- 校准建立完整映射
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
        # 合成视线平滑器
        self.smoother = AdaptiveSmoother(
            alpha_min=0.2,
            alpha_max=0.75,
            velocity_threshold=0.03,
        )
        # 校准映射
        self.calibrated = False
        self.M = np.eye(2)
        self.offset = np.zeros(2)

        # 手动偏移校正
        self.manual_offset = np.zeros(2)

        # 校准时的基准值
        self.iris_baseline = None       # 虹膜位置基准
        self.head_rot_baseline = None   # 头部旋转基准

    def _get_iris_position(self, landmarks):
        """
        获取双眼虹膜的平均归一化位置（相对眼角）
        这是视线方向的直接反映
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
            l_pos = (li - l_center) / l_w

            # 右眼
            ri = landmarks[config.RIGHT_IRIS_CENTER].to_array()
            r_inner = landmarks[config.RIGHT_EYE_INNER].to_array()
            r_outer = landmarks[config.RIGHT_EYE_OUTER].to_array()
            r_w = np.linalg.norm(r_inner - r_outer)
            if r_w < 1e-6:
                return None
            r_center = (r_inner + r_outer) / 2.0
            r_pos = (ri - r_center) / r_w

            return (l_pos + r_pos) / 2.0
        except (IndexError, AttributeError):
            return None

    def _get_head_rotation(self, landmarks):
        """头部旋转角度"""
        try:
            nose_tip = landmarks[1].to_array()
            forehead = landmarks[10].to_array()
            chin = landmarks[152].to_array()
            left_face = landmarks[234].to_array()
            right_face = landmarks[454].to_array()

            face_cx = (left_face[0] + right_face[0]) / 2.0
            face_cy = (forehead[1] + chin[1]) / 2.0
            face_w = abs(right_face[0] - left_face[0]) + 1e-6
            face_h = abs(chin[1] - forehead[1]) + 1e-6

            yaw = (nose_tip[0] - face_cx) / (face_w * 0.5)
            pitch = (nose_tip[1] - face_cy) / (face_h * 0.5)

            return np.array([pitch, yaw])
        except (IndexError, AttributeError):
            return None

    def compute_gaze_vector(self, landmarks):
        """
        视线方向 = 虹膜位置（主）+ 头部旋转（辅）
        """
        iris = self._get_iris_position(landmarks)
        head_rot = self._get_head_rotation(landmarks)

        if iris is None:
            return None

        # 虹膜相对基准的偏移（眼睛看的方向变化）
        if self.iris_baseline is not None:
            iris_delta = iris - self.iris_baseline
        else:
            iris_delta = iris

        # 头部旋转增量
        if head_rot is not None and self.head_rot_baseline is not None:
            head_delta = head_rot - self.head_rot_baseline
        else:
            head_delta = np.zeros(2) if head_rot is not None else np.zeros(2)

        # 合成：虹膜是主信号，放大到和头部旋转同量级
        # 虹膜偏移典型范围 ±0.03，放大 8 倍后 ±0.24
        # 头部旋转典型范围 ±0.15
        gaze = iris_delta * 8.0 + head_delta * 1.0

        return gaze

    def set_baseline(self, landmarks):
        """设置基准位置（校准时调用）"""
        iris = self._get_iris_position(landmarks)
        if iris is not None:
            self.iris_baseline = iris
        head_rot = self._get_head_rotation(landmarks)
        if head_rot is not None:
            self.head_rot_baseline = head_rot

    def gaze_to_screen(self, gaze, screen_w, screen_h):
        """将视线方向映射到屏幕坐标"""
        if self.calibrated:
            screen = self.M @ gaze + self.offset + self.manual_offset
            screen_x = clamp(screen[0], 0, screen_w - 1)
            screen_y = clamp(screen[1], 0, screen_h - 1)
        else:
            screen_x = linear_map(gaze[0], -0.2, 0.2, screen_w * 0.2, screen_w * 0.8)
            screen_y = linear_map(gaze[1], -0.15, 0.15, screen_h * 0.2, screen_h * 0.8)
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
        self.manual_offset = np.zeros(2)
        self.iris_baseline = None
        self.head_rot_baseline = None

    def get_raw_gaze(self, landmarks):
        """获取原始视线向量（用于校准采集）"""
        return self.compute_gaze_vector(landmarks)

    # 兼容旧接口
    def set_head_baseline(self, landmarks):
        self.set_baseline(landmarks)

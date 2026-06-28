"""手部追踪：食指控制鼠标，捏合点击"""

import mediapipe as mp
import numpy as np
import sys

sys.path.insert(0, ".")
from utils.smoothing import AdaptiveSmoother
from utils.math_utils import clamp

THUMB_TIP = 4
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 18
PINKY_TIP = 20
PINKY_PIP = 18

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.smoother = AdaptiveSmoother(
            alpha_min=0.3, alpha_max=0.85, velocity_threshold=0.02,
        )
        self._lm = None
        self._flip_x = False
        self._prev_pos = None  # 上一帧指尖位置（用于相对模式）
        self.relative_gain = 1.5  # 相对模式灵敏度

    def set_flip_x(self, flip):
        self._flip_x = flip

    def detect(self, frame_rgb):
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            self._lm = results.multi_hand_landmarks[0].landmark
            return self._lm
        self._lm = None
        return None

    def get_finger_pos_raw(self):
        """原始坐标（用于方向校准）"""
        if self._lm is None:
            return None
        tip = self._lm[INDEX_TIP]
        return np.array([tip.x, tip.y])

    def get_finger_pos(self):
        """校准后的归一化坐标 [0,1]"""
        if self._lm is None:
            return None
        tip = self._lm[INDEX_TIP]
        x = (1 - tip.x) if self._flip_x else tip.x
        return np.array([x, tip.y])

    def is_finger_up(self, tip, pip):
        if self._lm is None:
            return False
        return self._lm[tip].y < self._lm[pip].y

    def _dist(self, a, b):
        if self._lm is None:
            return 999
        p1, p2 = self._lm[a], self._lm[b]
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

    def get_gesture(self):
        if self._lm is None:
            return 'none'
        fingers = [self.is_finger_up(t, p) for t, p in zip(FINGER_TIPS, FINGER_PIPS)]
        idx, mid, ring, pinky = fingers
        pinch = self._dist(THUMB_TIP, INDEX_TIP) < 0.06
        if pinch and not mid and not ring and not pinky:
            return 'pinch'
        count = sum(fingers)
        if count == 1 and idx:
            return 'point'
        if count == 0:
            return 'fist'
        if count == 2 and idx and mid:
            return 'peace'
        if count == 3 and idx and mid and ring:
            return 'three'
        if count == 4:
            return 'open'
        return 'other'

    def get_relative_delta(self):
        """获取指尖相对上一帧的移动量（归一化），用于相对模式"""
        pos = self.get_finger_pos()
        if pos is None:
            self._prev_pos = None
            return None
        if self._prev_pos is None:
            self._prev_pos = pos.copy()
            return None
        delta = pos - self._prev_pos
        self._prev_pos = pos.copy()
        return delta

    def reset_relative(self):
        """重置相对模式基准"""
        self._prev_pos = None

    def to_screen(self, pos, sw, sh):
        """将归一化坐标映射到屏幕坐标（不对称边距）"""
        # 上1/12, 左1/12, 右1/12, 下40%
        x_lo, x_hi = 1/12, 11/12
        y_lo, y_hi = 1/12, 0.6
        x = clamp((pos[0] - x_lo) / (x_hi - x_lo), 0, 1)
        y = clamp((pos[1] - y_lo) / (y_hi - y_lo), 0, 1)
        return np.array([x * sw, y * sh])

    def draw(self, frame):
        if self._lm is not None:
            from mediapipe.framework.formats import landmark_pb2
            proto = landmark_pb2.NormalizedLandmarkList()
            for lm in self._lm:
                proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
            self.mp_drawing.draw_landmarks(
                frame, proto, self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style())

    def release(self):
        self.hands.close()

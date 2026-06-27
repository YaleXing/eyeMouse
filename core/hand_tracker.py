"""
手部追踪模块
- 检测食指指尖位置
- 识别手势（食指伸出 = 移动鼠标，握拳 = 点击）
"""

import mediapipe as mp
import numpy as np
import sys

sys.path.insert(0, ".")
import config
from utils.smoothing import AdaptiveSmoother
from utils.math_utils import clamp


# MediaPipe Hands 关键点索引
FINGER_TIPS = [8, 12, 16, 20]      # 食指、中指、无名指、小指指尖
FINGER_PIPS = [6, 10, 14, 18]      # 对应的 PIP 关节
THUMB_TIP = 4
THUMB_IP = 3


class HandTracker:
    """手部追踪器"""

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

        # 指尖位置平滑器
        self.smoother = AdaptiveSmoother(
            alpha_min=0.3,
            alpha_max=0.85,
            velocity_threshold=0.02,
        )

        # 上一帧的指尖位置（用于判断手指是否伸出）
        self._landmarks = None

    def detect(self, frame_rgb):
        """
        检测手部关键点
        返回: 手部关键点列表 或 None
        """
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            self._landmarks = results.multi_hand_landmarks[0].landmark
            return self._landmarks
        self._landmarks = None
        return None

    def get_index_finger_pos(self, frame_w, frame_h):
        """
        获取食指指尖的像素坐标
        返回: (x, y) 或 None
        """
        if self._landmarks is None:
            return None
        tip = self._landmarks[8]
        x = (1 - tip.x) * frame_w  # 水平翻转（镜像）
        y = tip.y * frame_h
        return np.array([x, y])

    def get_normalized_finger_pos(self):
        """
        获取食指指尖的归一化坐标 [0, 1]
        返回: (x, y) 或 None
        """
        if self._landmarks is None:
            return None
        tip = self._landmarks[8]
        return np.array([1 - tip.x, tip.y])  # 水平翻转

    def is_finger_up(self, tip_idx, pip_idx):
        """判断手指是否伸出（指尖 y < PIP 关节 y）"""
        if self._landmarks is None:
            return False
        return self._landmarks[tip_idx].y < self._landmarks[pip_idx].y

    def get_gesture(self):
        """
        识别手势
        返回: 'point'(食指), 'fist'(握拳), 'open'(张开), 'peace'(V手势), 'none'
        """
        if self._landmarks is None:
            return 'none'

        fingers = []
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            fingers.append(self.is_finger_up(tip, pip))

        count = sum(fingers)

        if count == 1 and fingers[0]:  # 只有食指
            return 'point'
        elif count == 0:
            return 'fist'
        elif count == 2 and fingers[0] and fingers[1]:
            return 'peace'
        elif count == 4:
            return 'open'
        else:
            return 'other'

    def finger_to_screen(self, finger_pos, screen_w, screen_h):
        """
        将指尖归一化坐标映射到屏幕坐标
        finger_pos: (nx, ny) 归一化坐标 [0, 1]
        """
        x = finger_pos[0] * screen_w
        y = finger_pos[1] * screen_h
        x = clamp(x, 0, screen_w - 1)
        y = clamp(y, 0, screen_h - 1)
        return np.array([x, y])

    def draw_landmarks(self, frame):
        """在画面上绘制手部关键点"""
        if self._landmarks is not None:
            # 构建 MediaPipe 格式
            from mediapipe.framework.formats import landmark_pb2
            proto = landmark_pb2.NormalizedLandmarkList()
            for lm in self._landmarks:
                proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
            self.mp_drawing.draw_landmarks(
                frame, proto, self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        return frame

    def release(self):
        self.hands.close()

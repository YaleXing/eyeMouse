"""
手部追踪模块
- 检测食指指尖位置
- 识别手势：食指=移动，捏合=点击
"""

import mediapipe as mp
import numpy as np
import sys

sys.path.insert(0, ".")
from utils.smoothing import AdaptiveSmoother
from utils.math_utils import clamp


# MediaPipe Hands 关键点索引
THUMB_TIP = 4
THUMB_IP = 3
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

        self.smoother = AdaptiveSmoother(
            alpha_min=0.3,
            alpha_max=0.85,
            velocity_threshold=0.02,
        )

        self._landmarks = None
        self._mirrored = True  # 摄像头是否镜像

    def set_mirror(self, mirrored):
        """设置是否镜像"""
        self._mirrored = mirrored

    def detect(self, frame_rgb):
        """检测手部关键点"""
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            self._landmarks = results.multi_hand_landmarks[0].landmark
            return self._landmarks
        self._landmarks = None
        return None

    def get_normalized_finger_pos(self):
        """
        获取食指指尖的归一化坐标 [0, 1]
        """
        if self._landmarks is None:
            return None
        tip = self._landmarks[INDEX_TIP]
        x = (1 - tip.x) if self._mirrored else tip.x
        return np.array([x, tip.y])

    def is_finger_up(self, tip_idx, pip_idx):
        """判断手指是否伸出"""
        if self._landmarks is None:
            return False
        return self._landmarks[tip_idx].y < self._landmarks[pip_idx].y

    def _finger_distance(self, idx1, idx2):
        """两个关键点之间的距离（归一化坐标）"""
        if self._landmarks is None:
            return 999
        p1 = self._landmarks[idx1]
        p2 = self._landmarks[idx2]
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

    def get_gesture(self):
        """
        识别手势：
        - 'point': 只有食指伸出 → 移动鼠标
        - 'pinch': 拇指+食指捏合 → 点击
        - 'peace': 食指+中指伸出 → 右键或其他
        - 'open': 全部张开
        - 'fist': 握拳
        - 'none': 无手
        """
        if self._landmarks is None:
            return 'none'

        fingers = []
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            fingers.append(self.is_finger_up(tip, pip))

        index_up, middle_up, ring_up, pinky_up = fingers

        # 捏合检测：拇指尖和食指尖距离很近
        pinch_dist = self._finger_distance(THUMB_TIP, INDEX_TIP)
        is_pinch = pinch_dist < 0.06  # 阈值

        # 捏合 + 其他手指收起 = 点击
        if is_pinch and not middle_up and not ring_up and not pinky_up:
            return 'pinch'

        count = sum(fingers)

        if count == 1 and index_up:
            return 'point'
        elif count == 0:
            return 'fist'
        elif count == 2 and index_up and middle_up:
            return 'peace'
        elif count == 4:
            return 'open'

        return 'other'

    def finger_to_screen(self, finger_pos, screen_w, screen_h):
        """将指尖归一化坐标映射到屏幕坐标"""
        x = finger_pos[0] * screen_w
        y = finger_pos[1] * screen_h
        x = clamp(x, 0, screen_w - 1)
        y = clamp(y, 0, screen_h - 1)
        return np.array([x, y])

    def draw_landmarks(self, frame):
        """在画面上绘制手部关键点"""
        if self._landmarks is not None:
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

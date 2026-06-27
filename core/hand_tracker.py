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

    def detect(self, frame_rgb):
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            self._lm = results.multi_hand_landmarks[0].landmark
            return self._lm
        self._lm = None
        return None

    def get_finger_pos(self):
        """食指指尖归一化坐标 [0,1]"""
        if self._lm is None:
            return None
        tip = self._lm[INDEX_TIP]
        return np.array([tip.x, tip.y])

    def is_finger_up(self, tip, pip):
        if self._lm is None:
            return False
        return self._lm[tip].y < self._lm[pip].y

    def _dist(self, a, b):
        if self._lm is None:
            return 999
        p1, p2 = self._lm[a], self._lm[b]
        return ((p1.x - p2.x)**2 + (1 - p2.y - (1 - p1.y))**2) ** 0.5

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
        if count == 4:
            return 'open'
        return 'other'

    def to_screen(self, pos, sw, sh):
        x = clamp(pos[0] * sw, 0, sw - 1)
        y = clamp(pos[1] * sh, 0, sh - 1)
        return np.array([x, y])

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

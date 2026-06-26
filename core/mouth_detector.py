"""
张嘴检测
- 计算 MAR（嘴唇纵横比）
- 防抖动：连续 N 帧才触发
- 连击间隔控制
"""

import time
import numpy as np
import sys

sys.path.insert(0, ".")
import config
from utils.math_utils import mouth_aspect_ratio


class MouthDetector:
    """张嘴检测器，用于触发鼠标点击"""

    def __init__(self):
        self.threshold = config.MOUTH_OPEN_THRESHOLD
        self.cooldown = config.MOUTH_CLICK_COOLDOWN
        self.hold_frames = config.MOUTH_HOLD_FRAMES

        self._open_count = 0          # 连续张嘴帧数
        self._last_click_time = 0.0   # 上次点击时间
        self._clicked_this_open = False  # 本次张嘴期间是否已点击

    def update(self, landmarks) -> bool:
        """
        检测张嘴状态
        landmarks: FaceLandmark 列表
        返回: True 表示应触发一次点击
        """
        try:
            mar = mouth_aspect_ratio(
                landmarks,
                config.MOUTH_TOP,
                config.MOUTH_BOTTOM,
                config.MOUTH_LEFT,
                config.MOUTH_RIGHT,
            )
        except (IndexError, AttributeError):
            self._open_count = 0
            self._clicked_this_open = False
            return False

        is_open = mar > self.threshold

        if is_open:
            self._open_count += 1

            # 连续帧数达标，且本次张嘴期间尚未点击，且冷却时间已过
            if (
                self._open_count >= self.hold_frames
                and not self._clicked_this_open
                and (time.time() - self._last_click_time) > self.cooldown
            ):
                self._last_click_time = time.time()
                self._clicked_this_open = True
                return True
        else:
            # 嘴闭合，重置状态
            self._open_count = 0
            self._clicked_this_open = False

        return False

    def get_mar(self, landmarks) -> float:
        """获取当前 MAR 值（用于调试显示）"""
        try:
            return mouth_aspect_ratio(
                landmarks,
                config.MOUTH_TOP,
                config.MOUTH_BOTTOM,
                config.MOUTH_LEFT,
                config.MOUTH_RIGHT,
            )
        except (IndexError, AttributeError):
            return 0.0

    def reset(self):
        self._open_count = 0
        self._clicked_this_open = False

"""
鼠标控制器
- 移动鼠标到指定屏幕坐标
- 卡尔曼滤波消除抖动
- 边界保护
"""

import pyautogui
import numpy as np
import sys

sys.path.insert(0, ".")
import config
from utils.smoothing import KalmanFilter2D
from utils.math_utils import clamp

# 禁用 pyautogui 的安全暂停（已有自己的控制）
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False  # 由我们自己做边界保护，避免误触发 failsafe


class MouseController:
    """鼠标控制器"""

    def __init__(self, screen_w=None, screen_h=None):
        # 自动检测屏幕尺寸
        if screen_w is None or screen_h is None:
            size = pyautogui.size()
            self.screen_w = size.width
            self.screen_h = size.height
        else:
            self.screen_w = screen_w
            self.screen_h = screen_h

        self.kalman = KalmanFilter2D(
            process_noise=config.KALMAN_PROCESS_NOISE,
            measure_noise=config.KALMAN_MEASURE_NOISE,
        )

        self._enabled = True

    def move_to(self, x, y, smooth=True):
        """
        移动鼠标到指定坐标
        x, y: 屏幕坐标（像素）
        smooth: 是否启用卡尔曼滤波
        """
        if not self._enabled:
            return

        # 保留 5px 边距，避免触发屏幕角落的系统热区
        margin = 5
        x = clamp(x, margin, self.screen_w - margin - 1)
        y = clamp(y, margin, self.screen_h - margin - 1)

        if smooth and config.MOUSE_SMOOTH_ENABLED:
            filtered = self.kalman.update(np.array([x, y]))
            x, y = filtered[0], filtered[1]
            x = clamp(x, 0, self.screen_w - 1)
            y = clamp(y, 0, self.screen_h - 1)

        pyautogui.moveTo(int(x), int(y), _pause=False)

    def click(self, button="left"):
        """执行一次鼠标点击"""
        if not self._enabled:
            return
        pyautogui.click(button=button, _pause=False)

    def move_relative(self, dx, dy):
        """相对移动（备用）"""
        if not self._enabled:
            return
        pyautogui.moveRel(int(dx), int(dy), _pause=False)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def is_enabled(self):
        return self._enabled

    def reset(self):
        """重置滤波器状态"""
        self.kalman.reset()

    def get_position(self):
        """获取当前鼠标位置"""
        pos = pyautogui.position()
        return (pos.x, pos.y)

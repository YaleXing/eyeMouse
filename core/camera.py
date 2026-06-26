"""
摄像头捕获管理
- 封装 OpenCV VideoCapture
- 支持分辨率 / 帧率配置
"""

import cv2
import sys


class Camera:
    """摄像头管理器"""

    def __init__(self, index=0, width=640, height=480, fps=60):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> bool:
        """打开摄像头，返回是否成功"""
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"[错误] 无法打开摄像头 (index={self.index})")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 读取实际生效的参数
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[摄像头] 已打开: {actual_w}x{actual_h} @ {actual_fps}fps")

        self.width = actual_w
        self.height = actual_h
        return True

    def read(self):
        """
        读取一帧
        返回: (success: bool, frame: ndarray)
        """
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret and frame is not None:
            # 水平翻转（镜像），让用户的左右与画面一致
            frame = cv2.flip(frame, 1)
        return ret, frame

    def release(self):
        """释放摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[摄像头] 已释放")

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

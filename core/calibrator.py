"""
校准系统
- 在屏幕上显示校准点
- 用户注视校准点时采集视线数据
- 用最小二乘法拟合仿射变换矩阵
"""

import cv2
import numpy as np
import time
import sys

sys.path.insert(0, ".")
import config


class Calibrator:
    """视线校准器"""

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # 校准点的屏幕坐标
        self.points = [
            (int(sx * screen_w), int(sy * screen_h))
            for sx, sy in config.CALIBRATION_POSITIONS
        ]

        # 采集的数据: [(gaze_vector, screen_point), ...]
        self.samples = []

    def get_calibration_screen(self, point_idx, progress):
        """
        生成校准界面
        point_idx: 当前校准点索引
        progress: 当前进度 [0, 1]
        返回: BGR 图像
        """
        screen = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

        # 绘制当前校准点（红色圆圈 + 中心点）
        cx, cy = self.points[point_idx]
        radius = int(30 * (1 - progress * 0.5))  # 圆圈逐渐缩小

        # 外圈
        cv2.circle(screen, (cx, cy), radius, (0, 0, 255), 2)
        # 中心点
        cv2.circle(screen, (cx, cy), 5, (0, 0, 255), -1)
        # 十字准星
        cv2.line(screen, (cx - radius - 10, cy), (cx + radius + 10, cy), (0, 0, 255), 1)
        cv2.line(screen, (cx, cy - radius - 10), (cx, cy + radius + 10), (0, 0, 255), 1)

        # 进度条
        bar_w = 200
        bar_h = 10
        bar_x = cx - bar_w // 2
        bar_y = cy + radius + 30
        cv2.rectangle(screen, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(screen, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)

        # 提示文字
        text = f"Please look at the red dot ({point_idx + 1}/{len(self.points)})"
        cv2.putText(screen, text, (cx - 200, cy - radius - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 显示所有校准点位置（灰色小点）
        for i, (px, py) in enumerate(self.points):
            if i != point_idx:
                cv2.circle(screen, (px, py), 4, (80, 80, 80), -1)

        return screen

    def add_sample(self, gaze_vector, point_idx):
        """添加一个校准样本"""
        screen_point = self.points[point_idx]
        self.samples.append((gaze_vector.copy(), np.array(screen_point, dtype=np.float64)))

    def compute_mapping(self):
        """
        用最小二乘法计算仿射变换
        screen = M * gaze + offset
        返回: (M, offset) 或 None（数据不足时）
        """
        if len(self.samples) < 3:
            print("[校准] 样本不足，无法计算映射")
            return None

        # 构建线性方程组: [gx, gy, 1] -> [sx, sy]
        A = []
        b = []
        for gaze, screen in self.samples:
            A.append([gaze[0], gaze[1], 1.0, 0.0, 0.0, 0.0])
            A.append([0.0, 0.0, 0.0, gaze[0], gaze[1], 1.0])
            b.append(screen[0])
            b.append(screen[1])

        A = np.array(A)
        b = np.array(b)

        # 最小二乘求解
        result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

        M = np.array([
            [result[0], result[1]],
            [result[3], result[4]],
        ])
        offset = np.array([result[2], result[5]])

        return M, offset

    def calibrate_from_samples(self):
        """
        从已采集的样本计算并返回校准参数
        返回: (M, offset) 或 None
        """
        # 按校准点分组，取平均值（减少噪声）
        grouped = {}
        for gaze, screen in self.samples:
            key = (int(screen[0]), int(screen[1]))
            if key not in grouped:
                grouped[key] = {"gazes": [], "screen": screen}
            grouped[key]["gazes"].append(gaze)

        # 用每个校准点的平均视线方向
        averaged_samples = []
        for key, data in grouped.items():
            avg_gaze = np.mean(data["gazes"], axis=0)
            averaged_samples.append((avg_gaze, data["screen"]))

        # 临时替换 samples 进行计算
        old_samples = self.samples
        self.samples = averaged_samples
        result = self.compute_mapping()
        self.samples = old_samples

        return result

    def save_calibration(self, M, offset, filepath="calibration.npy"):
        """保存校准数据"""
        np.savez(filepath, M=M, offset=offset)
        print(f"[校准] 已保存到 {filepath}")

    def load_calibration(self, filepath="calibration.npy"):
        """加载校准数据"""
        try:
            data = np.load(filepath)
            M = data["M"]
            offset = data["offset"]
            print(f"[校准] 已从 {filepath} 加载")
            return M, offset
        except FileNotFoundError:
            print(f"[校准] 未找到 {filepath}")
            return None, None

    def clear_samples(self):
        self.samples = []

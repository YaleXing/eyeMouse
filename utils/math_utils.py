"""
数学工具：映射、插值、归一化
"""

import numpy as np


def normalize_point(point, reference_box):
    """
    将关键点坐标相对于一个参考框进行归一化
    point: (x, y) 关键点坐标
    reference_box: (x_min, y_min, x_max, y_max) 参考框
    返回: (nx, ny) 归一化坐标，范围大致在 [-1, 1]
    """
    x, y = point
    x_min, y_min, x_max, y_max = reference_box
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = (x_max - x_min) or 1.0
    h = (y_max - y_min) or 1.0
    nx = (x - cx) / (w / 2.0)
    ny = (y - cy) / (h / 2.0)
    return np.array([nx, ny])


def normalize_to_unit(point, frame_w, frame_h):
    """将像素坐标归一化到 [0, 1]"""
    return np.array([point[0] / frame_w, point[1] / frame_h])


def linear_map(value, in_min, in_max, out_min, out_max):
    """线性映射"""
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min + 1e-8)


def poly_map(value, coeffs):
    """多项式映射（用于非线性校正）"""
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (value ** i)
    return result


def clamp(value, low, high):
    """限制值在 [low, high] 范围内"""
    return max(low, min(high, value))


def lerp(a, b, t):
    """线性插值"""
    return a + (b - a) * t


def eye_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """
    计算眼睛纵横比 (EAR)
    用于判断眼睛开合程度
    """
    top = np.array([landmarks[top_idx].px, landmarks[top_idx].py])
    bottom = np.array([landmarks[bottom_idx].px, landmarks[bottom_idx].py])
    left = np.array([landmarks[left_idx].px, landmarks[left_idx].py])
    right = np.array([landmarks[right_idx].px, landmarks[right_idx].py])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / (horizontal + 1e-8)


def mouth_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """
    计算嘴唇纵横比 (MAR)
    用于张嘴检测
    """
    top = np.array([landmarks[top_idx].px, landmarks[top_idx].py])
    bottom = np.array([landmarks[bottom_idx].px, landmarks[bottom_idx].py])
    left = np.array([landmarks[left_idx].px, landmarks[left_idx].py])
    right = np.array([landmarks[right_idx].px, landmarks[right_idx].py])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / (horizontal + 1e-8)

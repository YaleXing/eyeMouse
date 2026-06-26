"""
MediaPipe Face Mesh 封装
- 468 个面部关键点 + 虹膜关键点
- 输出结构化的关键点数据
"""

import mediapipe as mp
import numpy as np
import sys

sys.path.insert(0, ".")
import config


class FaceLandmark:
    """单个关键点的归一化坐标"""
    __slots__ = ("x", "y", "z", "px", "py")

    def __init__(self, x, y, z, px=0.0, py=0.0):
        self.x = x    # 归一化 x [0, 1]
        self.y = y    # 归一化 y [0, 1]
        self.z = z    # 归一化 z
        self.px = px  # 像素 x
        self.py = py  # 像素 y

    def to_array(self):
        return np.array([self.px, self.py])

    def __getitem__(self, idx):
        if idx == 0:
            return self.px
        elif idx == 1:
            return self.py
        elif idx == 2:
            return self.z
        raise IndexError(f"FaceLandmark index {idx} out of range")


class FaceMeshDetector:
    """MediaPipe Face Mesh 检测器"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.FACE_MESH_MAX_FACES,
            refine_landmarks=config.FACE_MESH_REFINE_LANDMARKS,
            min_detection_confidence=config.FACE_MESH_MIN_DET_CONF,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACK_CONF,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect(self, frame_rgb):
        """
        检测面部关键点
        frame_rgb: RGB 格式的图像 (numpy array)
        返回: list of list[FaceLandmark]（每个人脸一组关键点）
              如果未检测到返回空列表
        """
        h, w = frame_rgb.shape[:2]
        results = self.face_mesh.process(frame_rgb)

        all_faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append(FaceLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        px=lm.x * w,
                        py=lm.y * h,
                    ))
                all_faces.append(landmarks)

        return all_faces

    def draw_landmarks(self, frame, face_landmarks):
        """在图像上绘制面部网格"""
        if face_landmarks is None:
            return frame

        h, w = frame.shape[:2]
        # 构建 MediaPipe 格式
        from mediapipe.framework.formats import landmark_pb2
        proto = landmark_pb2.NormalizedLandmarkList()
        for lm in face_landmarks:
            proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)

        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=proto,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        # 绘制虹膜
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=proto,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1
            ),
        )
        return frame

    def release(self):
        self.face_mesh.close()

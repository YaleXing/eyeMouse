"""
全局配置文件 —— 所有可调参数集中在此
"""

# ── 摄像头 ─────────────────────────────────────────────
CAMERA_INDEX = 0            # 摄像头编号（0 = 默认摄像头）
CAMERA_WIDTH = 1280         # 采集宽度
CAMERA_HEIGHT = 720         # 采集高度
CAMERA_FPS = 30             # 期望帧率

# ── 屏幕 ───────────────────────────────────────────────
SCREEN_WIDTH = 1920         # 屏幕宽度（像素），运行时自动检测
SCREEN_HEIGHT = 1080        # 屏幕高度（像素），运行时自动检测

# ── MediaPipe Face Mesh ────────────────────────────────
FACE_MESH_MAX_FACES = 1
FACE_MESH_REFINE_LANDMARKS = True   # 启用虹膜精细化
FACE_MESH_MIN_DET_CONF = 0.5
FACE_MESH_MIN_TRACK_CONF = 0.5

# ── 视线追踪 ──────────────────────────────────────────
# 左眼关键点索引（MediaPipe Face Mesh）
LEFT_EYE_INNER = 133       # 左眼内角
LEFT_EYE_OUTER = 33        # 左眼外角
LEFT_EYE_TOP = 159         # 左眼上缘
LEFT_EYE_BOTTOM = 145      # 左眼下缘
LEFT_IRIS_CENTER = 468     # 左虹膜中心

# 右眼关键点索引
RIGHT_EYE_INNER = 362      # 右眼内角
RIGHT_EYE_OUTER = 263      # 右眼外角
RIGHT_EYE_TOP = 386        # 右眼上缘
RIGHT_EYE_BOTTOM = 374     # 右眼下缘
RIGHT_IRIS_CENTER = 473    # 右虹膜中心

# 视线平滑
GAZE_SMOOTHING_ALPHA = 0.4  # 指数移动平均系数 (0~1, 越大越灵敏)

# ── 鼠标控制 ──────────────────────────────────────────
MOUSE_MOVE_INTERVAL = 0.016  # 鼠标移动最小间隔（秒），约60fps
MOUSE_SMOOTH_ENABLED = True  # 启用鼠标平滑

# 卡尔曼滤波参数
KALMAN_PROCESS_NOISE = 0.01   # 越小鼠标越稳（降噪越强）
KALMAN_MEASURE_NOISE = 3.0    # 越大平滑越强

# 鼠标移动速度曲线（非线性映射的幂次, 1.0=线性）
MOUSE_SPEED_CURVE = 1.2

# ── 张嘴检测 ──────────────────────────────────────────
# 嘴唇关键点
MOUTH_TOP = 13              # 上唇中点
MOUTH_BOTTOM = 14           # 下唇中点
MOUTH_LEFT = 78             # 左嘴角
MOUTH_RIGHT = 308           # 右嘴角

MOUTH_OPEN_THRESHOLD = 0.06  # MAR 阈值（超过此值视为张嘴）
MOUTH_CLICK_COOLDOWN = 0.5   # 两次点击最小间隔（秒）
MOUTH_HOLD_FRAMES = 2        # 连续 N 帧张嘴才触发点击（防抖）

# ── 校准 ──────────────────────────────────────────────
CALIBRATION_POINTS = 5       # 校准点数（5点：四角+中心）
CALIBRATION_HOLD_TIME = 1.5  # 每个校准点需要注视的秒数
CALIBRATION_SAMPLES = 30     # 每个校准点采集的样本数

# 校准点位置（屏幕比例 0~1）
CALIBRATION_POSITIONS = [
    (0.1, 0.1),   # 左上
    (0.9, 0.1),   # 右上
    (0.5, 0.5),   # 中心
    (0.1, 0.9),   # 左下
    (0.9, 0.9),   # 右下
]

# ── 调试 ──────────────────────────────────────────────
DEBUG_SHOW_VIDEO = True      # 显示摄像头画面
DEBUG_SHOW_LANDMARKS = True  # 显示面部关键点
DEBUG_SHOW_GAZE = True       # 显示视线方向
DEBUG_WINDOW_SCALE = 0.6     # 调试窗口缩放比例

# ── 控制热键 ──────────────────────────────────────────
EXIT_KEY = 27                # Esc 退出
PAUSE_KEY = ord('p')         # P 暂停/恢复
RECALIBRATE_KEY = ord('c')   # C 重新校准

# 方向键微调偏移（像素/步）
OFFSET_STEP = 8

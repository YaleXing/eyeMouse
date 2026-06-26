# EyeMouse — 眼球追踪鼠标控制

通过摄像头追踪眼睛视线方向来控制鼠标光标，张嘴触发鼠标左键点击。

## 原理

- **MediaPipe Face Mesh** 提供 468 个面部关键点 + 虹膜关键点
- 计算虹膜中心相对于眼角的归一化偏移 → 得到视线方向
- 通过校准建立「视线方向 → 屏幕坐标」的映射
- 卡尔曼滤波 + 自适应平滑消除抖动、保持灵敏响应
- 嘴唇纵横比 (MAR) 检测张嘴动作 → 触发鼠标点击

## 环境要求

- Windows 10/11
- Python 3.10+
- 普通 USB 摄像头（推荐 720p 以上）

## 安装

```bash
# 1. 进入项目目录
cd eyeMouse

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 使用

```bash
python main.py
```

### 首次运行

1. 程序启动后自动进入 **5 点校准**
2. 屏幕上会出现红点，请 **注视红点** 直到进度条填满
3. 依次完成 5 个校准点（四角 + 中心）
4. 校准数据自动保存到 `calibration.npy`，下次运行自动加载

### 操作热键

| 按键 | 功能 |
|------|------|
| `ESC` | 退出程序 |
| `P` | 暂停/恢复鼠标控制 |
| `C` | 重新校准 |

### 张嘴点击

- 张嘴超过阈值 → 触发鼠标左键点击
- 有防抖机制（需连续 N 帧张嘴才触发）
- 有冷却间隔（防止连续误触）

## 配置调参

所有参数集中在 [config.py](config.py)，可以调整：

| 参数 | 说明 | 建议 |
|------|------|------|
| `GAZE_SMOOTHING_ALPHA` | 视线平滑系数 | 越大越灵敏，越小越平滑 |
| `MOUTH_OPEN_THRESHOLD` | 张嘴阈值 | 越小越灵敏（容易误触） |
| `MOUTH_CLICK_COOLDOWN` | 点击冷却（秒） | 防止连击 |
| `KALMAN_PROCESS_NOISE` | 卡尔曼过程噪声 | 越小鼠标越稳 |
| `KALMAN_MEASURE_NOISE` | 卡尔曼观测噪声 | 越大鼠标越平滑 |
| `DEBUG_SHOW_VIDEO` | 显示调试窗口 | 调试时打开 |

## 项目结构

```
eyeMouse/
├── main.py               # 入口
├── config.py             # 所有配置参数
├── requirements.txt      # Python 依赖
├── README.md             # 本文件
├── core/
│   ├── camera.py         # 摄像头管理
│   ├── face_mesh.py      # MediaPipe Face Mesh
│   ├── gaze_tracker.py   # 视线追踪
│   ├── mouth_detector.py # 张嘴检测
│   ├── mouse_controller.py # 鼠标控制
│   └── calibrator.py     # 校准系统
└── utils/
    ├── smoothing.py      # 卡尔曼滤波 + EMA
    └── math_utils.py     # 数学工具
```

## 精度优化建议

1. **光线充足** — 确保面部光照均匀，避免逆光
2. **摄像头位置** — 尽量放在屏幕正前方，与眼睛同高
3. **保持距离** — 50-80cm 为最佳距离
4. **校准时头部不动** — 校准过程中保持头部位置稳定
5. **校准后测试** — 如果精度不够，按 `C` 重新校准

## 常见问题

**Q: 鼠标抖动严重？**
调整 `KALMAN_MEASURE_NOISE`（增大）和 `GAZE_SMOOTHING_ALPHA`（减小）

**Q: 张嘴不触发点击？**
降低 `MOUTH_OPEN_THRESHOLD`（如 0.04），或降低 `MOUTH_HOLD_FRAMES`（如 1）

**Q: 误触太多？**
增大 `MOUTH_OPEN_THRESHOLD`（如 0.08），增大 `MOUTH_CLICK_COOLDOWN`

**Q: 检测不到面部？**
确保光线充足，摄像头清晰，`FACE_MESH_MIN_DET_CONF` 可适当降低

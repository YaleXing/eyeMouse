"""
EyeMouse —— 眼球追踪鼠标控制
主入口：启动摄像头 → 校准 → 主追踪循环
"""

import cv2
import time
import numpy as np
import sys
import os

# 确保从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import threading
from core.camera import Camera
from core.face_mesh import FaceMeshDetector
from core.gaze_tracker import GazeTracker
from core.mouth_detector import MouthDetector
from core.mouse_controller import MouseController
from core.calibrator import Calibrator


def beep_done():
    """播放完成提示音（异步，不阻塞主线程）"""
    def _beep():
        import winsound
        winsound.Beep(800, 120)
        winsound.Beep(1000, 120)
        winsound.Beep(1200, 180)
    threading.Thread(target=_beep, daemon=True).start()


def draw_debug_info(frame, landmarks, gaze_pos, mar, mouse_enabled, fps_val, manual_offset=None):
    """在调试画面上绘制信息"""
    h, w = frame.shape[:2]

    # 状态信息
    status_color = (0, 255, 0) if mouse_enabled else (0, 0, 255)
    status_text = "ACTIVE" if mouse_enabled else "PAUSED"
    cv2.putText(frame, f"Status: {status_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # MAR 值 + 可视化条
    is_open = mar > config.MOUTH_OPEN_THRESHOLD
    mar_color = (0, 0, 255) if is_open else (200, 200, 200)
    cv2.putText(frame, f"MAR: {mar:.4f} {'OPEN' if is_open else ''}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 1)
    # MAR 条
    bar_x, bar_y, bar_w, bar_h = 10, 100, 150, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(min(mar / 0.1, 1.0) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 0, 255) if is_open else (0, 200, 0), -1)
    # 阈值线
    thresh_x = bar_x + int(config.MOUTH_OPEN_THRESHOLD / 0.1 * bar_w)
    cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), (255, 255, 0), 1)

    # 手动偏移
    if manual_offset is not None:
        cv2.putText(frame, f"Offset: X={manual_offset[0]:.0f} Y={manual_offset[1]:.0f}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 100), 1)

    # 热键提示
    cv2.putText(frame, "ESC:quit P:pause C:calibrate WASD:offset", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # 视线位置标记（如果在调试窗口中显示）
    if gaze_pos is not None and config.DEBUG_SHOW_GAZE:
        # 将屏幕坐标映射回调试画面坐标
        gx = int(gaze_pos[0] / config.SCREEN_WIDTH * w)
        gy = int(gaze_pos[1] / config.SCREEN_HEIGHT * h)
        gx = max(0, min(w - 1, gx))
        gy = max(0, min(h - 1, gy))
        cv2.circle(frame, (gx, gy), 8, (0, 255, 255), 2)
        cv2.circle(frame, (gx, gy), 2, (0, 255, 255), -1)

    return frame


def run_calibration(camera, face_detector, gaze_tracker, screen_w, screen_h):
    """运行校准流程"""
    calibrator = Calibrator(screen_w, screen_h)

    # 尝试加载已有校准数据
    M, offset = calibrator.load_calibration()
    if M is not None:
        # 加载后仍需设置当前头部基准
        print("[校准] 使用已保存的校准数据，正在设置头部基准...")
        for _ in range(10):
            ret, frame = camera.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(frame_rgb)
            if faces:
                gaze_tracker.set_head_baseline(faces[0])
                break
        return M, offset

    print("[校准] 开始校准")
    print(f"[校准] 共 {len(calibrator.points)} 个校准点，每个需要注视 {config.CALIBRATION_HOLD_TIME} 秒")
    print("[校准] 请自然地转头看向每个红点（不需要保持头部不动）")

    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 校准开始前设置头部基准
    for _ in range(15):
        ret, frame = camera.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect(frame_rgb)
        if faces:
            gaze_tracker.set_head_baseline(faces[0])

    for point_idx in range(len(calibrator.points)):
        collected = 0
        start_time = time.time()

        while collected < config.CALIBRATION_SAMPLES:
            elapsed = time.time() - start_time
            progress = min(elapsed / config.CALIBRATION_HOLD_TIME, 1.0)

            # 读取摄像头帧
            ret, frame = camera.read()
            if not ret:
                continue

            # 检测面部
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(frame_rgb)

            # 绘制校准界面
            cal_screen = calibrator.get_calibration_screen(point_idx, progress)
            cv2.imshow("Calibration", cal_screen)

            # 采集数据（进度超过 30% 后开始采集，让用户有时间对准）
            if faces and progress > 0.3:
                gaze = gaze_tracker.get_raw_gaze(faces[0])
                if gaze is not None:
                    calibrator.add_sample(gaze, point_idx)
                    collected += 1

            # 进度完成后自动进入下一个点
            if progress >= 1.0 and collected > 0:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == config.EXIT_KEY:
                cv2.destroyWindow("Calibration")
                return None, None

        print(f"  校准点 {point_idx + 1}/{len(calibrator.points)} 完成 (采集 {collected} 个样本)")

    cv2.destroyWindow("Calibration")

    # 计算映射
    result = calibrator.calibrate_from_samples()
    if result is None:
        print("[校准] 校准失败，使用默认映射")
        return None, None

    M, offset = result
    calibrator.save_calibration(M, offset)
    return M, offset


def main():
    """主函数"""
    print("=" * 50)
    print("  EyeMouse — 眼球追踪鼠标控制")
    print("=" * 50)

    # ── 自动检测屏幕尺寸 ──
    try:
        import pyautogui
        screen_size = pyautogui.size()
        config.SCREEN_WIDTH = screen_size.width
        config.SCREEN_HEIGHT = screen_size.height
        print(f"[屏幕] {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
    except Exception:
        print(f"[屏幕] 使用配置值: {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

    # ── 初始化模块 ──
    camera = Camera(
        index=config.CAMERA_INDEX,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS,
    )
    if not camera.open():
        print("[错误] 无法打开摄像头，程序退出")
        return

    face_detector = FaceMeshDetector()
    gaze_tracker = GazeTracker()
    mouth_detector = MouthDetector()
    mouse_controller = MouseController(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

    # ── 校准 ──
    M, offset = run_calibration(
        camera, face_detector, gaze_tracker,
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
    )
    if M is not None:
        gaze_tracker.set_calibration(M, offset)
        print("[校准] 校准完成，鼠标控制已激活")
        beep_done()
    else:
        print("[校准] 跳过校准，使用默认映射（精度较低）")

    # ── 主循环 ──
    print("\n[运行] ESC退出 | P暂停 | C重校准 | WASD微调偏移")
    paused = False
    fps_counter = 0
    fps_time = time.time()
    fps_val = 0.0

    try:
        while True:
            # 读取帧
            ret, frame = camera.read()
            if not ret:
                print("[警告] 无法读取摄像头帧")
                time.sleep(0.01)
                continue

            # FPS 计算
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_val = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # 面部检测
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(frame_rgb)

            gaze_pos = None
            mar = 0.0

            if faces and not paused:
                landmarks = faces[0]

                # 视线追踪
                gaze_pos = gaze_tracker.update(
                    landmarks, config.SCREEN_WIDTH, config.SCREEN_HEIGHT
                )

                # 张嘴检测
                clicked = mouth_detector.update(landmarks)
                mar = mouth_detector.get_mar(landmarks)

                # 移动鼠标
                if gaze_pos is not None:
                    mouse_controller.move_to(gaze_pos[0], gaze_pos[1])

                # 张嘴点击
                if clicked:
                    mouse_controller.click()
                    print("[点击] 张嘴触发左键")
                    beep_done()

            elif not faces:
                # 面部丢失时只重置平滑器，保留头部基准
                gaze_tracker.smoother.reset()

            # 调试画面
            if config.DEBUG_SHOW_VIDEO:
                debug_frame = frame.copy()
                if config.DEBUG_SHOW_LANDMARKS and faces:
                    face_detector.draw_landmarks(debug_frame, faces[0])

                debug_frame = draw_debug_info(
                    debug_frame, faces[0] if faces else None,
                    gaze_pos, mar, not paused, fps_val,
                    gaze_tracker.manual_offset,
                )

                # 缩放调试窗口
                scale = config.DEBUG_WINDOW_SCALE
                h, w = debug_frame.shape[:2]
                debug_frame = cv2.resize(debug_frame, (int(w * scale), int(h * scale)))
                cv2.imshow("EyeMouse Debug", debug_frame)

            # 按键处理
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF

            # 方向键微调偏移（WASD 替代方案，更可靠）
            step = config.OFFSET_STEP
            if key == ord('a'):      # ← 左
                gaze_tracker.adjust_offset(-step, 0)
                print(f"[偏移] X={gaze_tracker.manual_offset[0]:.0f} Y={gaze_tracker.manual_offset[1]:.0f}")
            elif key == ord('d'):    # → 右
                gaze_tracker.adjust_offset(step, 0)
                print(f"[偏移] X={gaze_tracker.manual_offset[0]:.0f} Y={gaze_tracker.manual_offset[1]:.0f}")
            elif key == ord('w'):    # ↑ 上
                gaze_tracker.adjust_offset(0, -step)
                print(f"[偏移] X={gaze_tracker.manual_offset[0]:.0f} Y={gaze_tracker.manual_offset[1]:.0f}")
            elif key == ord('s'):    # ↓ 下
                gaze_tracker.adjust_offset(0, step)
                print(f"[偏移] X={gaze_tracker.manual_offset[0]:.0f} Y={gaze_tracker.manual_offset[1]:.0f}")

            if key == config.EXIT_KEY:
                break
            elif key == config.PAUSE_KEY:
                paused = not paused
                if paused:
                    mouse_controller.disable()
                    print("[暂停] 鼠标控制已暂停")
                else:
                    mouse_controller.enable()
                    print("[恢复] 鼠标控制已恢复")
            elif key == config.RECALIBRATE_KEY:
                print("[校准] 重新校准...")
                # 删除旧校准文件
                try:
                    os.remove("calibration.npy")
                except FileNotFoundError:
                    pass
                gaze_tracker.reset()
                mouse_controller.reset()
                M, offset = run_calibration(
                    camera, face_detector, gaze_tracker,
                    config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
                )
                if M is not None:
                    gaze_tracker.set_calibration(M, offset)
                    print("[校准] 重新校准完成")
                    beep_done()

    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        face_detector.release()
        cv2.destroyAllWindows()
        print("[退出] 程序已退出")


if __name__ == "__main__":
    main()

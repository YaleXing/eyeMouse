"""
EyeMouse —— 视线/手势鼠标控制
支持两种模式：
  - 手部模式(h)：食指控制鼠标，握拳点击（更精准）
  - 眼球模式(e)：视线控制鼠标，张嘴点击（免手操作）
按 Tab 切换模式
"""

import cv2
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.camera import Camera
from core.face_mesh import FaceMeshDetector
from core.gaze_tracker import GazeTracker
from core.mouth_detector import MouthDetector
from core.hand_tracker import HandTracker
from core.mouse_controller import MouseController
from core.calibrator import Calibrator


def draw_debug_info(frame, mode, gesture, gaze_pos, mar, finger_pos,
                    mouse_enabled, fps_val, manual_offset=None):
    """在调试画面上绘制信息"""
    h, w = frame.shape[:2]

    # 模式 + 状态
    mode_color = (0, 255, 255) if mode == 'hand' else (255, 200, 0)
    mode_text = "HAND (finger)" if mode == 'hand' else "EYE (gaze)"
    status = "ACTIVE" if mouse_enabled else "PAUSED"
    cv2.putText(frame, f"[{mode_text}] {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    y_offset = 90

    if mode == 'hand':
        # 手势
        gesture_names = {
            'point': 'INDEX (move)',
            'pinch': 'PINCH (click!)',
            'peace': 'PEACE (right click)',
            'open': 'OPEN PALM',
            'fist': 'FIST',
            'other': 'other',
            'none': 'no hand',
        }
        g_text = gesture_names.get(gesture, gesture)
        g_color = (0, 0, 255) if gesture == 'fist' else (200, 200, 200)
        cv2.putText(frame, f"Gesture: {g_text}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, g_color, 1)
        y_offset += 25
        # 指尖位置
        if finger_pos is not None:
            cv2.putText(frame, f"Finger: ({finger_pos[0]:.0f}, {finger_pos[1]:.0f})", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 100), 1)
    else:
        # MAR
        is_open = mar > config.MOUTH_OPEN_THRESHOLD
        mar_color = (0, 0, 255) if is_open else (200, 200, 200)
        cv2.putText(frame, f"MAR: {mar:.4f} {'OPEN' if is_open else ''}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mar_color, 1)
        y_offset += 15
        bar_x, bar_y, bar_w, bar_h = 10, y_offset, 150, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill = int(min(mar / 0.1, 1.0) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      (0, 0, 255) if is_open else (0, 200, 0), -1)
        thresh_x = bar_x + int(config.MOUTH_OPEN_THRESHOLD / 0.1 * bar_w)
        cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), (255, 255, 0), 1)
        y_offset += 20

    # 偏移
    if manual_offset is not None and (manual_offset[0] != 0 or manual_offset[1] != 0):
        cv2.putText(frame, f"Offset: X={manual_offset[0]:.0f} Y={manual_offset[1]:.0f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 100), 1)

    # 热键
    cv2.putText(frame, "TAB:switch V:hide ESC:quit P:pause", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    # 光标位置标记
    cursor = finger_pos if mode == 'hand' else gaze_pos
    if cursor is not None:
        cx = int(cursor[0] / config.SCREEN_WIDTH * w)
        cy = int(cursor[1] / config.SCREEN_HEIGHT * h)
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        color = (0, 255, 0) if mode == 'hand' else (0, 255, 255)
        cv2.circle(frame, (cx, cy), 10, color, 2)
        cv2.circle(frame, (cx, cy), 2, color, -1)

    return frame


def run_calibration(camera, face_detector, gaze_tracker, screen_w, screen_h):
    """运行校准流程（仅眼球模式需要）"""
    calibrator = Calibrator(screen_w, screen_h)

    M, offset = calibrator.load_calibration()
    if M is not None:
        print("[校准] 使用已保存的校准数据，正在设置基准...")
        for _ in range(10):
            ret, frame = camera.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(frame_rgb)
            if faces:
                gaze_tracker.set_baseline(faces[0])
                break
        return M, offset

    print("[校准] 开始校准，请自然地转头看向每个红点")
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for _ in range(15):
        ret, frame = camera.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect(frame_rgb)
        if faces:
            gaze_tracker.set_baseline(faces[0])

    for point_idx in range(len(calibrator.points)):
        collected = 0
        start_time = time.time()
        while collected < config.CALIBRATION_SAMPLES:
            elapsed = time.time() - start_time
            progress = min(elapsed / config.CALIBRATION_HOLD_TIME, 1.0)
            ret, frame = camera.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(frame_rgb)
            cal_screen = calibrator.get_calibration_screen(point_idx, progress)
            cv2.imshow("Calibration", cal_screen)
            if faces and progress > 0.3:
                gaze = gaze_tracker.get_raw_gaze(faces[0])
                if gaze is not None:
                    calibrator.add_sample(gaze, point_idx)
                    collected += 1
            if progress >= 1.0 and collected > 0:
                break
            key = cv2.waitKey(1) & 0xFF
            if key == config.EXIT_KEY:
                cv2.destroyWindow("Calibration")
                return None, None
        print(f"  校准点 {point_idx + 1}/{len(calibrator.points)} 完成 ({collected} samples)")

    cv2.destroyWindow("Calibration")
    result = calibrator.calibrate_from_samples()
    if result is None:
        print("[校准] 校准失败")
        return None, None
    M, offset = result
    calibrator.save_calibration(M, offset)
    return M, offset


def main():
    print("=" * 50)
    print("  EyeMouse — 手势/视线鼠标控制")
    print("=" * 50)

    # 屏幕尺寸
    try:
        import pyautogui
        sz = pyautogui.size()
        config.SCREEN_WIDTH = sz.width
        config.SCREEN_HEIGHT = sz.height
    except Exception:
        pass
    print(f"[屏幕] {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

    # 摄像头
    camera = Camera(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                    config.CAMERA_FPS, flip=config.CAMERA_FLIP)
    if not camera.open():
        print("[错误] 无法打开摄像头")
        return

    # 模块
    face_detector = FaceMeshDetector()
    gaze_tracker = GazeTracker()
    mouth_detector = MouthDetector()
    hand_tracker = HandTracker()
    hand_tracker.set_mirror(config.CAMERA_FLIP)
    mouse_controller = MouseController(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

    # 模式: 'hand' 或 'eye'
    mode = 'hand'
    print(f"[模式] 当前: 手部模式（食指控制） | 按 Tab 切换")

    paused = False
    show_debug = True  # 是否显示调试窗口
    fps_counter = 0
    fps_time = time.time()
    fps_val = 0.0
    fist_click_cooldown = 0  # 握拳点击冷却

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            # FPS
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_val = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            gaze_pos = None
            finger_pos = None
            mar = 0.0
            gesture = 'none'

            if not paused:
                # ── 手部模式 ──
                if mode == 'hand':
                    hand_landmarks = hand_tracker.detect(frame_rgb)
                    if hand_landmarks is not None:
                        gesture = hand_tracker.get_gesture()
                        fnp = hand_tracker.get_normalized_finger_pos()
                        if fnp is not None:
                            screen_pos = hand_tracker.finger_to_screen(
                                fnp, config.SCREEN_WIDTH, config.SCREEN_HEIGHT
                            )
                            # 轻度平滑
                            smoothed = hand_tracker.smoother.update(screen_pos)
                            finger_pos = smoothed
                            mouse_controller.move_to(smoothed[0], smoothed[1])

                        # 捏合点击（拇指+食指靠近）
                        if gesture == 'pinch' and now - fist_click_cooldown > 0.6:
                            mouse_controller.click()
                            fist_click_cooldown = now
                            print("[点击] 捏合触发")

                # ── 眼球模式 ──
                else:
                    faces = face_detector.detect(frame_rgb)
                    if faces:
                        gaze_pos = gaze_tracker.update(
                            faces[0], config.SCREEN_WIDTH, config.SCREEN_HEIGHT
                        )
                        clicked = mouth_detector.update(faces[0])
                        mar = mouth_detector.get_mar(faces[0])
                        if gaze_pos is not None:
                            mouse_controller.move_to(gaze_pos[0], gaze_pos[1])
                        if clicked:
                            mouse_controller.click()
                            print("[点击] 张嘴触发")
                    else:
                        gaze_tracker.smoother.reset()

            # 调试画面
            if show_debug:
                debug = frame.copy()
                if mode == 'hand':
                    hand_tracker.draw_landmarks(debug)
                debug = draw_debug_info(
                    debug, mode, gesture, gaze_pos, mar, finger_pos,
                    not paused, fps_val,
                    gaze_tracker.manual_offset if mode == 'eye' else None,
                )
                scale = config.DEBUG_WINDOW_SCALE
                h, w = debug.shape[:2]
                debug = cv2.resize(debug, (int(w * scale), int(h * scale)))
                cv2.imshow("EyeMouse Debug", debug)
            else:
                cv2.destroyWindow("EyeMouse Debug")

            # 按键
            key = cv2.waitKeyEx(1) & 0xFF

            if key == 9:  # Tab → 切换模式
                mode = 'eye' if mode == 'hand' else 'hand'
                print(f"[模式] 切换到: {'手部模式（食指控制）' if mode == 'hand' else '眼球模式（视线控制）'}")
                if mode == 'eye':
                    # 切到眼球模式时需要校准
                    M, offset = run_calibration(
                        camera, face_detector, gaze_tracker,
                        config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
                    )
                    if M is not None:
                        gaze_tracker.set_calibration(M, offset)
                        print("[校准] 完成")
            elif key == ord('h'):  # h → 直接切手部模式
                mode = 'hand'
                print("[模式] 手部模式")
            elif key == ord('e'):  # e → 直接切眼球模式
                mode = 'eye'
                M, offset = run_calibration(
                    camera, face_detector, gaze_tracker,
                    config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
                )
                if M is not None:
                    gaze_tracker.set_calibration(M, offset)
                print("[模式] 眼球模式")
            elif key == ord('v'):  # v → 隐藏/显示调试画面
                show_debug = not show_debug
                print(f"[画面] {'显示' if show_debug else '隐藏'}")
            elif key == ord('a'):
                gaze_tracker.adjust_offset(-config.OFFSET_STEP, 0)
            elif key == ord('d'):
                gaze_tracker.adjust_offset(config.OFFSET_STEP, 0)
            elif key == ord('w'):
                gaze_tracker.adjust_offset(0, -config.OFFSET_STEP)
            elif key == ord('s'):
                gaze_tracker.adjust_offset(0, config.OFFSET_STEP)
            elif key == config.EXIT_KEY:
                break
            elif key == config.PAUSE_KEY:
                paused = not paused
                if paused:
                    mouse_controller.disable()
                    print("[暂停]")
                else:
                    mouse_controller.enable()
                    print("[恢复]")

    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        face_detector.release()
        hand_tracker.release()
        cv2.destroyAllWindows()
        print("[退出] 程序已退出")


if __name__ == "__main__":
    main()

"""
EyeMouse —— 视线/手势鼠标控制
支持两种模式：
  - 手部模式(h)：食指控制鼠标，捏合点击（更精准）
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
from core.mouse_controller import MouseController


def ask_mirror():
    """启动时弹窗选择镜像"""
    screen = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.putText(screen, "Camera Mirror?", (80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(screen, "[Y] Yes - Mirror (default)", (60, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
    cv2.putText(screen, "[N] No - Normal", (60, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)
    cv2.putText(screen, "Press Y or N", (150, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.imshow("EyeMouse Setup", screen)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('y'), ord('Y'), 13):  # Y or Enter
            cv2.destroyWindow("EyeMouse Setup")
            return True
        elif key in (ord('n'), ord('N')):
            cv2.destroyWindow("EyeMouse Setup")
            return False


def draw_debug_info(frame, mode, gesture, cursor_pos, mar,
                    mouse_enabled, fps_val, mirror_on):
    """调试画面"""
    h, w = frame.shape[:2]

    mode_color = (0, 255, 255) if mode == 'hand' else (255, 200, 0)
    mode_text = "HAND" if mode == 'hand' else "EYE"
    status = "ON" if mouse_enabled else "OFF"
    cv2.putText(frame, f"[{mode_text}] {status}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

    cv2.putText(frame, f"FPS:{fps_val:.0f} Mirror:{'Y' if mirror_on else 'N'}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    y = 75
    if mode == 'hand':
        gn = {'point': 'INDEX(move)', 'pinch': 'PINCH(click!)',
              'peace': 'PEACE', 'open': 'OPEN', 'fist': 'FIST',
              'other': 'other', 'none': 'no hand'}
        gc = (0, 0, 255) if gesture == 'pinch' else (180, 180, 180)
        cv2.putText(frame, f"Gesture: {gn.get(gesture, gesture)}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 1)
    else:
        is_open = mar > config.MOUTH_OPEN_THRESHOLD
        mc = (0, 0, 255) if is_open else (180, 180, 180)
        cv2.putText(frame, f"MAR: {mar:.3f} {'OPEN' if is_open else ''}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mc, 1)

    cv2.putText(frame, "TAB:switch V:hide ESC:quit M:mirror", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # 光标标记
    if cursor_pos is not None:
        cx = int(cursor_pos[0] / config.SCREEN_WIDTH * w)
        cy = int(cursor_pos[1] / config.SCREEN_HEIGHT * h)
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        col = (0, 255, 0) if mode == 'hand' else (0, 255, 255)
        cv2.circle(frame, (cx, cy), 10, col, 2)
        cv2.circle(frame, (cx, cy), 2, col, -1)

    return frame


def run_calibration(camera, face_detector, gaze_tracker, screen_w, screen_h):
    """校准（仅眼球模式）"""
    from core.calibrator import Calibrator
    calibrator = Calibrator(screen_w, screen_h)

    M, offset = calibrator.load_calibration()
    if M is not None:
        print("[校准] 加载已保存数据...")
        for _ in range(10):
            ret, frame = camera.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(rgb)
            if faces:
                gaze_tracker.set_baseline(faces[0])
                break
        return M, offset

    print("[校准] 请自然地转头看向每个红点")
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for _ in range(15):
        ret, frame = camera.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(rgb)
            if faces:
                gaze_tracker.set_baseline(faces[0])

    for pi in range(len(calibrator.points)):
        collected = 0
        t0 = time.time()
        while collected < config.CALIBRATION_SAMPLES:
            elapsed = time.time() - t0
            progress = min(elapsed / config.CALIBRATION_HOLD_TIME, 1.0)
            ret, frame = camera.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(rgb)
            cal_screen = calibrator.get_calibration_screen(pi, progress)
            cv2.imshow("Calibration", cal_screen)
            if faces and progress > 0.3:
                g = gaze_tracker.get_raw_gaze(faces[0])
                if g is not None:
                    calibrator.add_sample(g, pi)
                    collected += 1
            if progress >= 1.0 and collected > 0:
                break
            if cv2.waitKey(1) & 0xFF == config.EXIT_KEY:
                cv2.destroyWindow("Calibration")
                return None, None
        print(f"  点 {pi+1}/{len(calibrator.points)} 完成 ({collected} samples)")

    cv2.destroyWindow("Calibration")
    result = calibrator.calibrate_from_samples()
    if result is None:
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

    # 镜像选择
    mirror = ask_mirror()
    config.CAMERA_FLIP = mirror
    print(f"[镜像] {'开启' if mirror else '关闭'}")

    # 摄像头
    camera = Camera(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                    config.CAMERA_FPS, flip=mirror)
    if not camera.open():
        print("[错误] 无法打开摄像头")
        return

    # 懒加载模块
    face_detector = None
    gaze_tracker = None
    mouth_detector = None
    hand_tracker = None

    def ensure_hand():
        nonlocal hand_tracker
        if hand_tracker is None:
            from core.hand_tracker import HandTracker
            hand_tracker = HandTracker()
            hand_tracker.set_mirror(mirror)

    def ensure_eye():
        nonlocal face_detector, gaze_tracker, mouth_detector
        if face_detector is None:
            from core.face_mesh import FaceMeshDetector
            from core.gaze_tracker import GazeTracker
            from core.mouth_detector import MouthDetector
            face_detector = FaceMeshDetector()
            gaze_tracker = GazeTracker()
            mouth_detector = MouthDetector()

    mouse_controller = MouseController(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

    mode = 'hand'
    print("[模式] 手部模式（食指控制）| Tab切换 | V隐藏画面 | M镜像 | ESC退出")

    paused = False
    show_debug = True
    fps_counter = 0
    fps_time = time.time()
    fps_val = 0.0
    click_cooldown = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_val = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cursor_pos = None
            mar = 0.0
            gesture = 'none'

            if not paused:
                if mode == 'hand':
                    ensure_hand()
                    lm = hand_tracker.detect(frame_rgb)
                    if lm is not None:
                        gesture = hand_tracker.get_gesture()
                        fnp = hand_tracker.get_normalized_finger_pos()
                        if fnp is not None:
                            sp = hand_tracker.finger_to_screen(
                                fnp, config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                            smoothed = hand_tracker.smoother.update(sp)
                            cursor_pos = smoothed
                            mouse_controller.move_to(smoothed[0], smoothed[1])
                        if gesture == 'pinch' and now - click_cooldown > 0.6:
                            mouse_controller.click()
                            click_cooldown = now
                            print("[点击] 捏合")
                else:
                    ensure_eye()
                    faces = face_detector.detect(frame_rgb)
                    if faces:
                        cursor_pos = gaze_tracker.update(
                            faces[0], config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                        clicked = mouth_detector.update(faces[0])
                        mar = mouth_detector.get_mar(faces[0])
                        if cursor_pos is not None:
                            mouse_controller.move_to(cursor_pos[0], cursor_pos[1])
                        if clicked:
                            mouse_controller.click()
                            print("[点击] 张嘴")
                    elif gaze_tracker:
                        gaze_tracker.smoother.reset()

            if show_debug:
                debug = frame.copy()
                if mode == 'hand' and hand_tracker:
                    hand_tracker.draw_landmarks(debug)
                debug = draw_debug_info(
                    debug, mode, gesture, cursor_pos, mar,
                    not paused, fps_val, mirror)
                s = config.DEBUG_WINDOW_SCALE
                dh, dw = debug.shape[:2]
                debug = cv2.resize(debug, (int(dw * s), int(dh * s)))
                cv2.imshow("EyeMouse Debug", debug)
            else:
                try:
                    cv2.destroyWindow("EyeMouse Debug")
                except cv2.error:
                    pass

            key = cv2.waitKeyEx(1) & 0xFF

            if key == 9:  # Tab
                mode = 'eye' if mode == 'hand' else 'hand'
                print(f"[模式] {'手部' if mode == 'hand' else '眼球'}模式")
                if mode == 'eye':
                    ensure_eye()
                    M, off = run_calibration(
                        camera, face_detector, gaze_tracker,
                        config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                    if M is not None:
                        gaze_tracker.set_calibration(M, off)
            elif key == ord('h'):
                mode = 'hand'
                print("[模式] 手部模式")
            elif key == ord('e'):
                mode = 'eye'
                ensure_eye()
                M, off = run_calibration(
                    camera, face_detector, gaze_tracker,
                    config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                if M is not None:
                    gaze_tracker.set_calibration(M, off)
            elif key == ord('v'):
                show_debug = not show_debug
                print(f"[画面] {'显示' if show_debug else '隐藏'}")
            elif key == ord('m'):
                mirror = not mirror
                config.CAMERA_FLIP = mirror
                camera.release()
                camera = Camera(config.CAMERA_INDEX, config.CAMERA_WIDTH,
                                config.CAMERA_HEIGHT, config.CAMERA_FPS, flip=mirror)
                camera.open()
                if hand_tracker:
                    hand_tracker.set_mirror(mirror)
                print(f"[镜像] {'开启' if mirror else '关闭'}")
            elif key == ord('a') and gaze_tracker:
                gaze_tracker.adjust_offset(-config.OFFSET_STEP, 0)
            elif key == ord('d') and gaze_tracker:
                gaze_tracker.adjust_offset(config.OFFSET_STEP, 0)
            elif key == ord('w') and gaze_tracker:
                gaze_tracker.adjust_offset(0, -config.OFFSET_STEP)
            elif key == ord('s') and gaze_tracker:
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
        print("\n[退出]")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        if face_detector:
            face_detector.release()
        if hand_tracker:
            hand_tracker.release()
        cv2.destroyAllWindows()
        print("[退出]")


if __name__ == "__main__":
    main()

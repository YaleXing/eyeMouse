"""
EyeMouse — 手势/视线鼠标控制
手部模式(默认): 食指移动鼠标，捏合点击
眼球模式: 视线移动鼠标，张嘴点击
"""

import cv2
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.camera import Camera
from core.mouse_controller import MouseController


def draw_hud(frame, mode, gesture, cursor, mar, fps, paused):
    h, w = frame.shape[:2]
    mc = (0, 255, 255) if mode == 'hand' else (255, 200, 0)
    mt = "HAND" if mode == 'hand' else "EYE"
    st = "PAUSED" if paused else ""
    cv2.putText(frame, f"[{mt}] {st}  FPS:{fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc, 2)
    y = 50
    if mode == 'hand':
        gn = {'point': 'INDEX', 'pinch': 'PINCH(click)', 'peace': 'PEACE',
              'open': 'OPEN', 'fist': 'FIST', 'none': 'no hand'}
        gc = (0, 0, 255) if gesture == 'pinch' else (180, 180, 180)
        cv2.putText(frame, gn.get(gesture, gesture), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 1)
    else:
        io = mar > config.MOUTH_OPEN_THRESHOLD
        cv2.putText(frame, f"MAR:{mar:.3f}{' OPEN' if io else ''}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if io else (180, 180, 180), 1)
    if cursor is not None:
        cx = int(cursor[0] / config.SCREEN_WIDTH * w)
        cy = int(cursor[1] / config.SCREEN_HEIGHT * h)
        col = (0, 255, 0) if mode == 'hand' else (0, 255, 255)
        cv2.circle(frame, (max(0, min(w-1, cx)), max(0, min(h-1, cy))), 8, col, 2)
    return frame


def run_calibration(camera, face_detector, gaze_tracker, sw, sh):
    from core.calibrator import Calibrator
    cal = Calibrator(sw, sh)
    M, off = cal.load_calibration()
    if M is not None:
        for _ in range(10):
            ret, f = camera.read()
            if ret:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                faces = face_detector.detect(rgb)
                if faces:
                    gaze_tracker.set_baseline(faces[0])
                    break
        return M, off
    print("[校准] 请自然地转头看向每个红点")
    cv2.namedWindow("Cal", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Cal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for _ in range(15):
        ret, f = camera.read()
        if ret:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(rgb)
            if faces:
                gaze_tracker.set_baseline(faces[0])
    for pi in range(len(cal.points)):
        collected = 0
        t0 = time.time()
        while collected < config.CALIBRATION_SAMPLES:
            elapsed = time.time() - t0
            p = min(elapsed / config.CALIBRATION_HOLD_TIME, 1.0)
            ret, f = camera.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect(rgb)
            cv2.imshow("Cal", cal.get_calibration_screen(pi, p))
            if faces and p > 0.3:
                g = gaze_tracker.get_raw_gaze(faces[0])
                if g is not None:
                    cal.add_sample(g, pi)
                    collected += 1
            if p >= 1.0 and collected > 0:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow("Cal")
                return None, None
        print(f"  点{pi+1}/{len(cal.points)} ({collected}样本)")
    cv2.destroyWindow("Cal")
    result = cal.calibrate_from_samples()
    if result is None:
        return None, None
    M, off = result
    cal.save_calibration(M, off)
    return M, off


def main():
    print("=" * 40)
    print("  EyeMouse — 手势/视线鼠标控制")
    print("=" * 40)

    try:
        import pyautogui
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT = pyautogui.size().width, pyautogui.size().height
    except Exception:
        pass

    camera = Camera(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
    if not camera.open():
        return

    mouse = MouseController(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

    # 懒加载
    hand_tracker = None
    face_detector = None
    gaze_tracker = None
    mouth_detector = None

    def ensure_hand():
        nonlocal hand_tracker
        if hand_tracker is None:
            from core.hand_tracker import HandTracker
            hand_tracker = HandTracker()

    def ensure_eye():
        nonlocal face_detector, gaze_tracker, mouth_detector
        if face_detector is None:
            from core.face_mesh import FaceMeshDetector
            from core.gaze_tracker import GazeTracker
            from core.mouth_detector import MouthDetector
            face_detector = FaceMeshDetector()
            gaze_tracker = GazeTracker()
            mouth_detector = MouthDetector()

    mode = 'hand'
    show_debug = False  # 默认隐藏画面
    paused = False
    fps_c = 0
    fps_t = time.time()
    fps = 0.0
    click_cd = 0

    print("[手部模式] 食指移动 | 捏合点击")
    print("[热键] V显示画面 | Tab切换模式 | P暂停 | ESC退出")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            fps_c += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps = fps_c / (now - fps_t)
                fps_c = 0
                fps_t = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cursor = None
            mar = 0.0
            gesture = 'none'

            if not paused:
                if mode == 'hand':
                    ensure_hand()
                    lm = hand_tracker.detect(rgb)
                    if lm:
                        gesture = hand_tracker.get_gesture()
                        fp = hand_tracker.get_finger_pos()
                        if fp is not None:
                            sp = hand_tracker.to_screen(fp, config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                            cursor = hand_tracker.smoother.update(sp)
                            mouse.move_to(cursor[0], cursor[1])
                        if gesture == 'pinch' and now - click_cd > 0.6:
                            mouse.click()
                            click_cd = now
                            print("[点击] 捏合")
                else:
                    ensure_eye()
                    faces = face_detector.detect(rgb)
                    if faces:
                        cursor = gaze_tracker.update(faces[0], config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                        clicked = mouth_detector.update(faces[0])
                        mar = mouth_detector.get_mar(faces[0])
                        if cursor is not None:
                            mouse.move_to(cursor[0], cursor[1])
                        if clicked:
                            mouse.click()
                            print("[点击] 张嘴")
                    elif gaze_tracker:
                        gaze_tracker.smoother.reset()

            if show_debug:
                debug = frame.copy()
                if mode == 'hand' and hand_tracker:
                    hand_tracker.draw(debug)
                debug = draw_hud(debug, mode, gesture, cursor, mar, fps, paused)
                s = config.DEBUG_WINDOW_SCALE
                dh, dw = debug.shape[:2]
                cv2.imshow("EyeMouse", cv2.resize(debug, (int(dw*s), int(dh*s))))
            else:
                try:
                    cv2.destroyWindow("EyeMouse")
                except cv2.error:
                    pass

            key = cv2.waitKeyEx(1) & 0xFF

            if key == 9:  # Tab
                mode = 'eye' if mode == 'hand' else 'hand'
                print(f"[模式] {'手部' if mode=='hand' else '眼球'}")
                if mode == 'eye':
                    ensure_eye()
                    M, off = run_calibration(camera, face_detector, gaze_tracker,
                                            config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                    if M is not None:
                        gaze_tracker.set_calibration(M, off)
            elif key == ord('v'):
                show_debug = not show_debug
                print(f"[画面] {'显示' if show_debug else '隐藏'}")
            elif key == ord('h'):
                mode = 'hand'
                print("[模式] 手部")
            elif key == ord('e'):
                mode = 'eye'
                ensure_eye()
                M, off = run_calibration(camera, face_detector, gaze_tracker,
                                        config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
                if M is not None:
                    gaze_tracker.set_calibration(M, off)
            elif key == ord('p'):
                paused = not paused
                if paused:
                    mouse.disable()
                else:
                    mouse.enable()
                print(f"[{'暂停' if paused else '恢复'}]")
            elif key == 27:
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[错误] {e}")
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

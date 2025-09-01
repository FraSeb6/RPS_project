# models/CNN_real_time/run_realtime_cnn1.py
# -----------------------------------------------------------
# Real-time Rock–Paper–Scissors on macOS (AVFoundation + MediaPipe)
# - Auto-picks the latest CNN_1 model (or use a fixed path)
# - MediaPipe Hands for robust hand box
# - Prediction smoothing (moving average on probs)
# - English on-screen messages
# Keys: [ESC]/q = quit, f = flip preview, i = toggle input preview
# -----------------------------------------------------------

import os
import sys
import glob
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# === CONFIG ===
# Expected input size for CNN_1 (H, W) as used in training
IMG_H, IMG_W = 200, 300
CLASS_NAMES = ["paper", "rock", "scissors"]  # <- adjust only if your training order differs
CONFIDENCE_MIN = 0.55                         # below this => "Not sure"
SMOOTHING_WINDOW = 5                          # moving average over last N predictions
DRAW_PREVIEW = True                           # small model-input preview window

# Project root (assumes this file is inside models/CNN_real_time/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CNN1_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "models", "CNN_1", "outputs")

# If you prefer a fixed model path, set it here and skip auto-discovery:
FIXED_MODEL_PATH = None
# Example:
# FIXED_MODEL_PATH = "/Users/youruser/.../models/CNN_1/outputs/2025-09-01_22-32-43/models/predictor_CNN_1.keras"


def find_latest_cnn1_model(outputs_dir: str) -> str:
    """
    Auto-discover the latest 'predictor_CNN_1.keras' under models/CNN_1/outputs/*/models/
    """
    pattern = os.path.join(outputs_dir, "*", "models", "predictor_CNN_1.keras")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return ""
    # outputs dir names are timestamps, sorted() lexicographically works thanks to the YYYY-MM-DD_HH-mm-ss format
    return candidates[-1]


def load_model() -> tf.keras.Model:
    model_path = FIXED_MODEL_PATH or find_latest_cnn1_model(CNN1_OUTPUTS_DIR)
    if not model_path or not os.path.exists(model_path):
        print("❌ Could not find a CNN_1 model file.")
        print("   Looked under:", CNN1_OUTPUTS_DIR)
        print("   You can set FIXED_MODEL_PATH at the top of this file to the exact .keras path.")
        sys.exit(1)
    print(f"✅ Using model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def select_camera() -> cv2.VideoCapture:
    """
    Try AVFoundation on indexes [0,1,2]; fallback to default backend if needed.
    """
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    # Fallback
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    print("❌ No camera found. Check macOS privacy permissions: System Settings > Privacy & Security > Camera.")
    sys.exit(1)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def crop_hand(frame_bgr, hand_landmarks, margin_ratio=0.25):
    """
    Crop a padded bounding box around the detected hand.
    margin_ratio is relative to the larger side of the box.
    """
    h, w, _ = frame_bgr.shape
    xs = [int(pt.x * w) for pt in hand_landmarks.landmark]
    ys = [int(pt.y * h) for pt in hand_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    side = max(x_max - x_min, y_max - y_min)
    margin = int(side * margin_ratio)

    x1 = clamp(x_min - margin, 0, w - 1)
    y1 = clamp(y_min - margin, 0, h - 1)
    x2 = clamp(x_max + margin, 0, w - 1)
    y2 = clamp(y_max + margin, 0, h - 1)

    hand_bgr = frame_bgr[y1:y2, x1:x2]
    return hand_bgr, (x1, y1, x2, y2)


def preprocess_for_model(hand_bgr):
    """
    Resize to (IMG_W, IMG_H), convert to RGB, normalize to [0,1], add batch dim.
    """
    if hand_bgr.size == 0:
        return None, None
    hand_bgr = cv2.resize(hand_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    hand_rgb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2RGB)
    inp = hand_rgb.astype(np.float32) / 255.0
    return inp[np.newaxis, ...], hand_rgb  # (1, H, W, 3), (H, W, 3)


def text_shadow(img, text, org, font, scale, color, thickness=2):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def main():
    print("==> Real-time RPS (CNN_1 + MediaPipe)")
    model = load_model()

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = select_camera()
    print("Press [ESC]/q to quit, [f] to flip, [i] to toggle input preview")

    # For smoother labels
    prob_buffer = deque(maxlen=SMOOTHING_WINDOW)
    flip = False
    show_preview = DRAW_PREVIEW

    # Windows
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    if show_preview:
        cv2.namedWindow("Model Input (RGB)", cv2.WINDOW_NORMAL)

    last_fps_t = time.time()
    frames = 0
    fps_txt = "FPS: --"

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("⚠️ Could not read from camera.")
                break

            if flip:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            label_text = "No hand"
            preview_rgb = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            box = None

            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                hand_bgr, box = crop_hand(frame_bgr, hand_lm, margin_ratio=0.25)

                if hand_bgr.size != 0:
                    inp, hand_rgb = preprocess_for_model(hand_bgr)
                    if inp is not None:
                        pred = model.predict(inp, verbose=0)[0]  # (3,)
                        prob_buffer.append(pred)
                        # Moving average over last N frames
                        smoothed = np.mean(np.stack(prob_buffer, axis=0), axis=0) if len(prob_buffer) > 0 else pred
                        idx = int(np.argmax(smoothed))
                        conf = float(smoothed[idx])

                        if conf < CONFIDENCE_MIN:
                            label_text = "Not sure"
                        else:
                            label_text = f"{CLASS_NAMES[idx]} ({conf*100:.0f}%)"

                        preview_rgb = hand_rgb.copy()

                # Draw landmarks & box for debug
                mp_draw.draw_landmarks(frame_bgr, hand_lm, mp_hands.HAND_CONNECTIONS)
                if box:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (80, 220, 80), 2)

            # FPS
            frames += 1
            if time.time() - last_fps_t >= 1.0:
                fps_txt = f"FPS: {frames}"
                frames = 0
                last_fps_t = time.time()

            # HUD
            text_shadow(frame_bgr, label_text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 220, 50), 2)
            text_shadow(frame_bgr, fps_txt, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 50), 2)

            cv2.imshow("Webcam", frame_bgr)
            if show_preview:
                cv2.imshow("Model Input (RGB)", preview_rgb)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                break
            elif key == ord('f'):
                flip = not flip
            elif key == ord('i'):
                show_preview = not show_preview
                if not show_preview and cv2.getWindowProperty("Model Input (RGB)", 0) >= 0:
                    cv2.destroyWindow("Model Input (RGB)")
                elif show_preview:
                    cv2.namedWindow("Model Input (RGB)", cv2.WINDOW_NORMAL)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("✅ Bye!")


if __name__ == "__main__":
    main()

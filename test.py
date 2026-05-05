"""
Real-Time Emotion Recognition
==============================
Uses your trained `best_emotion.keras` model + webcam to detect emotions live.

Requirements:
    pip install tensorflow opencv-python numpy

Usage:
    python realtime_emotion.py

Controls:
    Q  /  ESC  →  quit
    S          →  save screenshot
    P          →  pause / resume
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
path = r'D:\XuLyAnh\workspace_v2\models\best_emotion.keras'
# ── Config (edit these if needed) ────────────────────────────────────────────
MODEL_PATH = path   # path to your trained model
CAM_INDEX  = 0                      # camera index (0 = default webcam)
IMG_SIZE   = 48                     # fallback input size if not read from model

# ── Emotion labels (edit to match your training labels) ──────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Colour palette per emotion  (BGR)
COLOURS = {
    "Angry":    (0,   0,   220),
    "Disgust":  (0,   128, 0  ),
    "Fear":     (128, 0,   128),
    "Happy":    (0,   220, 220),
    "Neutral":  (200, 200, 200),
    "Sad":      (220, 100, 0  ),
    "Surprise": (0,   165, 255),
}
DEFAULT_COLOUR = (180, 180, 180)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading model: {MODEL_PATH}")
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded. Input shape: {model.input_shape}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    raise

# Auto-detect image size from model if possible
try:
    _, h, w, *_ = model.input_shape   # (None, H, W, C)
    IMG_SIZE = h if h is not None else IMG_SIZE
except Exception:
    pass  # keep default IMG_SIZE

# Auto-detect channels
try:
    CHANNELS = model.input_shape[-1]
    USE_GRAY = (CHANNELS == 1)
except Exception:
    USE_GRAY = True

print(f"[INFO] Inference size: {IMG_SIZE}x{IMG_SIZE}, grayscale: {USE_GRAY}")

# ── Face detector ─────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize, normalise, and reshape a face crop for the model."""
    if USE_GRAY:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)          # (H, W, 1)
    else:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0         # (H, W, 3)

    return np.expand_dims(face, axis=0)               # (1, H, W, C)


def draw_bar_chart(frame, probs, x, y, width=160, bar_height=14, gap=4):
    """Draw a mini probability bar chart next to the face box."""
    for i, (label, prob) in enumerate(zip(EMOTIONS, probs)):
        top = y + i * (bar_height + gap)
        # Background track
        cv2.rectangle(frame, (x, top), (x + width, top + bar_height),
                      (50, 50, 50), -1)
        # Filled bar
        fill = int(width * prob)
        colour = COLOURS.get(label, DEFAULT_COLOUR)
        cv2.rectangle(frame, (x, top), (x + fill, top + bar_height),
                      colour, -1)
        # Label + percentage
        cv2.putText(frame, f"{label}: {prob*100:.1f}%",
                    (x + 4, top + bar_height - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                    cv2.LINE_AA)


def overlay_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)


def overlay_status(frame, text, colour=(200, 200, 200)):
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

    print("[INFO] Camera opened. Press Q/ESC to quit, S to screenshot, P to pause.")

    paused      = False
    prev_time   = time.time()
    fps         = 0.0
    screenshot_dir = "screenshots"

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):          # Q or ESC
            break
        elif key == ord('p'):
            paused = not paused
            print("[INFO] Paused" if paused else "[INFO] Resumed")
        elif key == ord('s'):
            os.makedirs(screenshot_dir, exist_ok=True)
            fname = os.path.join(screenshot_dir,
                                 f"emotion_{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved: {fname}")

        if paused:
            overlay_status(frame, "PAUSED – press P to resume", (0, 200, 255))
            cv2.imshow("Emotion Recognition  [Q=quit | S=screenshot | P=pause]", frame)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying…")
            continue

        # ── FPS ──────────────────────────────────────────────────────────────
        now      = time.time()
        fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── Face detection ────────────────────────────────────────────────────
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )

        face_count = len(faces) if isinstance(faces, np.ndarray) else 0

        for (fx, fy, fw, fh) in (faces if face_count else []):
            face_crop = frame[fy:fy + fh, fx:fx + fw]
            if face_crop.size == 0:
                continue

            # ── Inference ────────────────────────────────────────────────────
            input_tensor = preprocess_face(face_crop)
            preds        = model.predict(input_tensor, verbose=0)[0]
            top_idx      = int(np.argmax(preds))
            top_label    = EMOTIONS[top_idx] if top_idx < len(EMOTIONS) else str(top_idx)
            top_conf     = float(preds[top_idx])
            colour       = COLOURS.get(top_label, DEFAULT_COLOUR)

            # ── Draw face box ─────────────────────────────────────────────────
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), colour, 2)

            # Label above the box
            label_text = f"{top_label}  {top_conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame,
                          (fx, fy - th - 10), (fx + tw + 6, fy),
                          colour, -1)
            cv2.putText(frame, label_text,
                        (fx + 3, fy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Bar chart to the right of the box (clamp to frame width)
            chart_x = min(fx + fw + 8, frame.shape[1] - 175)
            chart_y = fy
            draw_bar_chart(frame, preds, chart_x, chart_y)

        # ── HUD ───────────────────────────────────────────────────────────────
        overlay_fps(frame, fps)
        overlay_status(frame,
                       f"Faces: {face_count}  |  Q=quit  S=screenshot  P=pause")

        cv2.imshow("Emotion Recognition  [Q=quit | S=screenshot | P=pause]", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
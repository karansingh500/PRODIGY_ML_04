import ctypes
import os
import urllib.request
from collections import deque
from importlib import resources

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import GESTURES, IMG_SIZE, MODEL_SAVE_PATH
from data_loader import to_model_batch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark


def _patch_mediapipe_windows_free():
    """MediaPipe 0.10 on Windows looks up free() in libmediapipe.dll; it lives in the CRT."""
    if os.name == "posix":
        return

    def load_raw_library(signatures=()):
        if mediapipe_c_bindings._shared_lib is None:
            lib_path = str(resources.files("mediapipe.tasks.c") / "libmediapipe.dll")
            mediapipe_c_bindings._shared_lib = ctypes.CDLL(lib_path)
            mediapipe_c_bindings._shared_lib.free = ctypes.CDLL("ucrtbase").free
        lib = mediapipe_c_bindings._shared_lib
        for signature in signatures:
            c_func = getattr(lib, signature.func_name)
            c_func.argtypes = signature.argtypes
            c_func.restype = signature.restype
        lib.free.argtypes = [ctypes.c_void_p]
        lib.free.restype = None
        return lib

    mediapipe_c_bindings.load_raw_library = load_raw_library


_patch_mediapipe_windows_free()

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = os.path.join("models", "hand_landmarker.task")

prediction_history = deque(maxlen=5)
wrist_history = deque(maxlen=14)

FINGER_TIPS = (
    HandLandmark.INDEX_FINGER_TIP,
    HandLandmark.MIDDLE_FINGER_TIP,
    HandLandmark.RING_FINGER_TIP,
    HandLandmark.PINKY_TIP,
)
FINGER_PIPS = (
    HandLandmark.INDEX_FINGER_PIP,
    HandLandmark.MIDDLE_FINGER_PIP,
    HandLandmark.RING_FINGER_PIP,
    HandLandmark.PINKY_PIP,
)


def ensure_hand_model():
    if os.path.exists(HAND_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
    print("Downloading MediaPipe hand landmarker...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)


def _dist(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _palm_size(lm):
    return _dist(lm[HandLandmark.WRIST], lm[HandLandmark.MIDDLE_FINGER_MCP]) + 1e-6


def _finger_extended(lm, tip, pip):
    wrist = lm[HandLandmark.WRIST]
    return _dist(lm[tip], wrist) > _dist(lm[pip], wrist) * 1.08


def _thumb_extended(lm):
    wrist = lm[HandLandmark.WRIST]
    tip = lm[HandLandmark.THUMB_TIP]
    ip = lm[HandLandmark.THUMB_IP]
    return _dist(tip, wrist) > _dist(ip, wrist) * 1.05


def _pointing_down(lm):
    """True when fingertips sit clearly below the wrist (image y grows downward)."""
    wrist = lm[HandLandmark.WRIST]
    middle_tip = lm[HandLandmark.MIDDLE_FINGER_TIP]
    index_tip = lm[HandLandmark.INDEX_FINGER_TIP]
    dx = middle_tip.x - wrist.x
    dy = middle_tip.y - wrist.y
    middle_down = dy > 0.06 and dy > abs(dx) * 0.55
    index_dx = index_tip.x - wrist.x
    index_dy = index_tip.y - wrist.y
    index_down = index_dy > 0.06 and index_dy > abs(index_dx) * 0.55
    mean_tip_y = float(np.mean([lm[i].y for i in FINGER_TIPS]))
    return middle_down or index_down or (mean_tip_y > wrist.y + 0.07)


def _is_ok(lm, n_others):
    """OK: thumb tip touches index tip, forming a loop (other fingers often up)."""
    scale = _palm_size(lm)
    tip_gap = _dist(lm[HandLandmark.THUMB_TIP], lm[HandLandmark.INDEX_FINGER_TIP]) / scale
    loop = _dist(lm[HandLandmark.THUMB_IP], lm[HandLandmark.INDEX_FINGER_PIP]) / scale
    touching = tip_gap < 0.55
    has_circle = loop > tip_gap * 1.15
    return touching and has_circle and n_others >= 1


def _hand_is_moving():
    if len(wrist_history) < 8:
        return False
    pts = np.array(wrist_history, dtype=np.float32)
    path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    recent = float(np.linalg.norm(pts[-1] - pts[-5]))
    return path > 0.07 or span > 0.05 or recent > 0.03


def classify_from_landmarks(lm):
    """Map MediaPipe landmarks to leapGestRecog class indices (static pose)."""
    thumb = _thumb_extended(lm)
    extended = [_finger_extended(lm, tip, pip) for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]
    index_up, middle, ring, pinky = extended
    n_fingers = sum(extended)
    n_others = int(middle) + int(ring) + int(pinky)

    if _is_ok(lm, n_others):
        return GESTURES["07_ok"], 0.92
    if _pointing_down(lm) and n_fingers >= 2:
        return GESTURES["10_down"], 0.9

    thumb_index_gap = _dist(lm[HandLandmark.THUMB_TIP], lm[HandLandmark.INDEX_FINGER_TIP])
    finger_curl = float(
        np.mean([_dist(lm[tip], lm[pip]) for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)])
    )

    if index_up and not middle and not ring and not pinky:
        if thumb:
            return GESTURES["02_l"], 0.9
        return GESTURES["06_index"], 0.9
    if thumb and n_fingers == 0:
        return GESTURES["05_thumb"], 0.88
    if n_fingers <= 1 and not thumb:
        return GESTURES["03_fist"], 0.9
    if n_fingers >= 3:
        if 0.08 < thumb_index_gap < 0.20 and finger_curl < 0.12:
            return GESTURES["09_c"], 0.75
        return GESTURES["01_palm"], 0.88
    return GESTURES["03_fist"], 0.55


def apply_motion_label(class_index):
    """fist_moved / palm_moved are the same poses while the hand is translating."""
    moving = _hand_is_moving()
    if moving and class_index == GESTURES["03_fist"]:
        return GESTURES["04_fist_moved"], 0.9
    if moving and class_index == GESTURES["01_palm"]:
        return GESTURES["08_palm_moved"], 0.9
    return class_index, None


def isolate_hand(frame, landmarks, padding=0.4):
    h, w = frame.shape[:2]
    pts = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks], dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    pad = int(max(bw, bh) * padding)
    x_min = max(0, x - pad)
    y_min = max(0, y - pad)
    x_max = min(w, x + bw + pad)
    y_max = min(h, y + bh + pad)

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.dilate(mask, kernel, iterations=1)

    isolated = np.zeros_like(frame)
    isolated[mask > 0] = frame[mask > 0]
    roi = isolated[y_min:y_max, x_min:x_max]
    return (x_min, y_min, x_max, y_max), roi


def letterbox_square(image):
    if image.size == 0:
        return image
    h, w = image.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas


def smooth_prediction(class_index, confidence):
    prediction_history.append((class_index, confidence))
    classes = [p[0] for p in prediction_history]
    most_common_class = max(set(classes), key=classes.count)
    avg_confidence = np.mean([p[1] for p in prediction_history if p[0] == most_common_class])
    return most_common_class, float(avg_confidence)


def create_landmarker():
    ensure_hand_model()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def main():
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model not found at {MODEL_SAVE_PATH}. Train the model first.")
        return

    model = load_model(MODEL_SAVE_PATH)
    class_labels = {v: k.split("_", 1)[1].upper() for k, v in GESTURES.items()}
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        landmarker.close()
        return

    print("Press 'q' to quit.")
    print("OK: thumb+index circle, other fingers up")
    print("DOWN: point fingers toward the bottom of the screen")
    print("FIST_MOVED / PALM_MOVED: hold fist or palm and wave the hand")

    current_gesture = None
    current_confidence = 0.0
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            bbox, roi = isolate_hand(frame, lm)
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            wrist = lm[HandLandmark.WRIST]
            wrist_history.append((wrist.x, wrist.y))

            lm_class, lm_conf = classify_from_landmarks(lm)
            moved_class, moved_conf = apply_motion_label(lm_class)
            if moved_conf is not None:
                lm_class, lm_conf = moved_class, moved_conf

            class_index = lm_class
            confidence = lm_conf

            if roi.size != 0:
                square = letterbox_square(roi)
                gray_roi = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_roi = clahe.apply(gray_roi)
                reshaped = to_model_batch(gray_roi)
                edges = (reshaped[0, :, :, 0] * 255).astype(np.uint8)

                prediction = model.predict(reshaped, verbose=0)
                cnn_index = int(np.argmax(prediction))

                pose_locked = {
                    GESTURES["07_ok"],
                    GESTURES["10_down"],
                    GESTURES["04_fist_moved"],
                    GESTURES["08_palm_moved"],
                    GESTURES["03_fist"],
                    GESTURES["06_index"],
                    GESTURES["02_l"],
                    GESTURES["05_thumb"],
                }
                if lm_class not in pose_locked and float(prediction[0][cnn_index]) > 0.8:
                    class_index = cnn_index
                    confidence = float(prediction[0][cnn_index])

                smoothed_class, smoothed_confidence = smooth_prediction(class_index, confidence)
                moving = _hand_is_moving()
                if moving and smoothed_class == GESTURES["03_fist"]:
                    smoothed_class = GESTURES["04_fist_moved"]
                elif moving and smoothed_class == GESTURES["01_palm"]:
                    smoothed_class = GESTURES["08_palm_moved"]
                elif not moving and smoothed_class == GESTURES["04_fist_moved"]:
                    smoothed_class = GESTURES["03_fist"]
                elif not moving and smoothed_class == GESTURES["08_palm_moved"]:
                    smoothed_class = GESTURES["01_palm"]
                current_gesture = class_labels[smoothed_class]
                current_confidence = smoothed_confidence
                cv2.putText(
                    frame,
                    f"{current_gesture} ({smoothed_confidence * 100:.1f}%)",
                    (x_min, max(25, y_min - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    edges,
                    f"CNN:{class_labels[cnn_index]}",
                    (2, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    255,
                    1,
                )
                cv2.imshow("What the Model Sees (Edges)", edges)
        else:
            current_gesture = None
            prediction_history.clear()
            wrist_history.clear()
            cv2.imshow("What the Model Sees (Edges)", np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))

        cv2.putText(
            frame,
            "OK: thumb+index O | DOWN: fingers down | wave fist/palm | q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        if current_gesture:
            cv2.putText(
                frame,
                f"Current: {current_gesture} ({current_confidence * 100:.1f}%)",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Exited successfully")


if __name__ == "__main__":
    main()

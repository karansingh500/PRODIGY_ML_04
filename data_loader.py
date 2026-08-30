import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from config import BASE_PATH, IMG_SIZE, GESTURES, NUM_CLASSES

CANNY_T1 = 50
CANNY_T2 = 150


def resolve_dataset_path():
    """Handle both dataset/leapGestRecog/00 and a nested leapGestRecog folder."""
    if os.path.isdir(os.path.join(BASE_PATH, "00")):
        return BASE_PATH
    nested = os.path.join(BASE_PATH, "leapGestRecog")
    if os.path.isdir(os.path.join(nested, "00")):
        return nested
    return BASE_PATH


def preprocess_gray(gray):
    """Match train and webcam: resize, then Canny, then [0, 1]."""
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(resized, CANNY_T1, CANNY_T2)
    return edges.astype("float32") / 255.0


def to_model_batch(gray):
    return preprocess_gray(gray).reshape(1, IMG_SIZE, IMG_SIZE, 1)


def load_data():
    images = []
    labels = []
    dataset_root = resolve_dataset_path()

    print(f"Loading images from {dataset_root} ...")

    total_images = 0
    for subject in range(10):
        subject_folder = f"{subject:02d}"
        subject_path = os.path.join(dataset_root, subject_folder)
        if not os.path.exists(subject_path):
            print(f"Subject folder not found: {subject_path}")
            continue

        print(f"Loading subject {subject}...")
        for gesture_name, label in GESTURES.items():
            gesture_path = os.path.join(subject_path, gesture_name)
            if not os.path.exists(gesture_path):
                continue

            img_files = [
                f for f in os.listdir(gesture_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            for img_name in img_files:
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(preprocess_gray(img))
                labels.append(label)
                total_images += 1

    print(f"Total images loaded: {total_images}")

    if len(images) == 0:
        print("ERROR: No images were loaded! Check dataset path and structure.")
        return None, None, None, None

    X = np.array(images, dtype="float32").reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_int = np.array(labels)
    y = to_categorical(y_int, num_classes=NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_int
    )

    print(f"Data Loaded! Training shapes: {X_train.shape}, {y_train.shape}")
    return X_train, X_test, y_train, y_test

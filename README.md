# PRODIGY_ML_04 — Hand Gesture Recognition

Train a CNN on the **LeapGestRecog** dataset (10 gestures) and run a **live webcam demo**. The training path is a standard image classifier. The webcam path is harder: Leap images are near-infrared, so a color camera does not look like the training set. The demo therefore finds the hand with MediaPipe, then labels the pose from finger landmarks (and still runs the CNN on an edge crop).

## What the project does

1. Load grayscale gesture images from ten subjects (`00`–`09`).
2. Turn each image into a 64×64 Canny edge map.
3. Train a small CNN and save `best_gesture_model.h5`.
4. On the webcam: detect one hand, crop it onto a black background, classify the gesture, draw a box and label.

| Folder | On-screen label | Typical pose |
| --- | --- | --- |
| `01_palm` | PALM | Open palm, fingers up |
| `02_l` | L | Thumb + index |
| `03_fist` | FIST | Closed fist, still |
| `04_fist_moved` | FIST_MOVED | Fist while waving |
| `05_thumb` | THUMB | Thumb up |
| `06_index` | INDEX | Index finger up |
| `07_ok` | OK | Thumb–index circle, other fingers up |
| `08_palm_moved` | PALM_MOVED | Open palm while waving |
| `09_c` | C | C-shape |
| `10_down` | DOWN | Fingers toward the **bottom** of the frame |

---

## Codebase

The repo is split so config, data, model, train, eval, and webcam stay independent.

### `config.py`

Shared constants: dataset path, `best_gesture_model.h5`, `IMG_SIZE = 64`, batch size, epochs, learning rate, and the `GESTURES` dict (folder name → class index `0`–`9`). Every other module imports from here so labels stay consistent.

### `data_loader.py`

- `resolve_dataset_path()` — works if files sit in `dataset/leapGestRecog/00` **or** a nested `leapGestRecog/leapGestRecog/00` (common after unzipping Kaggle).
- `preprocess_gray()` — resize to 64×64, Canny (`50`, `150`), scale to `[0, 1]`. Training and webcam both use this so the CNN always sees edges, not raw pixels.
- `to_model_batch()` — one grayscale crop → shape `(1, 64, 64, 1)` for `model.predict`.
- `load_data()` — walks subjects and gesture folders, stacks `X`, one-hot `y`, **stratified** 80/20 split.

### `model.py`

A compact Sequential CNN:

```text
Input 64×64×1
Conv2D 32 → BatchNorm → MaxPool
Conv2D 64 → BatchNorm → MaxPool
Conv2D 128 → BatchNorm → MaxPool
Flatten → Dense 256 → Dropout 0.5 → Dense 10 (softmax)
```

BatchNorm helps training stay stable; dropout reduces overfitting on similar Leap frames.

### `train.py`

Loads data, compiles with Adam + categorical cross-entropy, then fits with:

- **ModelCheckpoint** — keep only the best `val_accuracy` weights.
- **EarlyStopping** — stop if `val_loss` does not improve for 5 epochs; restore best weights.
- **ReduceLROnPlateau** — cut LR by half if loss stalls.

Prints final test accuracy after training.

### `evaluate_model.py`

Reloads the saved `.h5` and the same split (same `random_state=42`). Prints a sklearn classification report and a confusion matrix. Use this to see which classes mix up on **dataset** images (not the webcam).

### `predict_webcam.py`

Live loop:

1. Open camera `0`, mirror the frame (`flip`) so it feels like a mirror.
2. Run **MediaPipe Hand Landmarker** (21 landmarks) in VIDEO mode.
3. Build a padded box around the landmarks, mask the hand (convex hull) onto black, letterbox to a square.
4. CLAHE + Canny → CNN (debug window: “What the Model Sees”).
5. **Primary label** from landmark geometry (`classify_from_landmarks`): which fingers are extended, OK loop, pointing down.
6. **Motion:** wrist positions over ~14 frames. If the pose is fist/palm **and** the wrist has moved enough → `FIST_MOVED` / `PALM_MOVED`.
7. Majority vote over 5 frames to reduce flicker. Motion is applied again after smoothing so a wave is not voted back to still fist/palm.
8. CNN is not allowed to override clear landmark poses (OK, down, fist, L, index, thumb, moved). It can still help on ambiguous C vs palm.

`models/hand_landmarker.task` is downloaded on first run if missing.

---

## How it works (two pipelines)

```text
LeapGestRecog PNG  →  grayscale  →  64×64 Canny  →  CNN  →  class 0–9
                                                      ↑
Webcam BGR  →  MediaPipe hand  →  isolated crop  →  same Canny
                 ↓
           finger / motion rules  →  on-screen label
```

**Offline:** the network learns Leap edge shapes. Accuracy on the test split is about the CNN only.

**Online:** a phone/laptop camera is RGB, cluttered, and a different camera model. Feeding a bad crop into Canny produced a blob that the CNN almost always called **fist**. Landmark rules fix that for a live demo; the CNN still runs so the intern task (trained model + webcam) stays intact.

---

## Setup and commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install tensorflow opencv-python mediapipe scikit-learn numpy
```

Dataset layout:

```text
dataset/leapGestRecog/00/01_palm/*.png
...
dataset/leapGestRecog/09/10_down/*.png
```

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate_model.py
.\.venv\Scripts\python.exe predict_webcam.py
```

Press **q** to quit the webcam. One hand, reasonable light. For **DOWN**, point fingers at the bottom of the image. For **MOVED** classes, wave the hand; holding still stays `FIST` / `PALM`.

---

## Difficulties faced

### 1. Webcam is not the Leap camera

LeapGestRecog is IR: dark background, high-contrast hand. Webcam frames have faces, walls, and color skin. Even with the same Canny function, the CNN saw a different distribution and collapsed to **fist** (a compact blob looks like a fist in edge space).

**What we did:** isolate the hand on black, match preprocessing as far as possible, and drive the live label from MediaPipe finger pose instead of trusting the CNN alone.

### 2. Skin-color “hand detection” was wrong

The first detector used YCrCb + HSV skin masks. It often boxed the **face**, neck, or furniture. The crop was not a hand, so every gesture looked like a fist.

**What we did:** MediaPipe Hand Landmarker for a real hand box (landmarks used for pose; skeleton is not drawn on the video).

### 3. MediaPipe crashed on Windows

`HandLandmarker.create_from_options` failed with:

```text
AttributeError: function 'free' not found
```

MediaPipe 0.10 looks up `free()` inside `libmediapipe.dll`. On Windows that symbol lives in the C runtime (`ucrtbase`), not in the MediaPipe DLL.

**What we did:** in `predict_webcam.py`, patch `load_raw_library` so `free` is bound to `ucrtbase` before the landmarker loads. Harmless XNNPACK / “feedback tensor” logs can still appear.

### 4. OK, DOWN, and the two “moved” classes did not fire

- **OK** was coded as a pinch with the other fingers *folded*. The Leap OK is a thumb–index **circle** with other fingers **up**.
- **DOWN** is not an upright palm; fingertips must sit **below** the wrist in image coordinates (`y` increases downward).
- **FIST_MOVED / PALM_MOVED** are the same poses as fist/palm plus **translation**. Measuring motion between two consecutive frames was too small, and a 5–7 frame majority vote snapped the label back to still fist/palm.

**What we did:** relative thumb–index loop + other fingers for OK; wrist-to-tip direction for DOWN; wrist path over many frames for motion, re-applied after smoothing.

### 5. Dataset zip nesting

Kaggle zips often add an extra `leapGestRecog` folder. A hardcoded path loaded zero images.

**What we did:** `resolve_dataset_path()` checks both layouts.

### 6. TensorFlow on native Windows

TF 2.x reports that GPU is not used on native Windows. Training and webcam inference run on **CPU**. That is expected, not a broken install.

---

## Limits (still true)

- `palm` vs `palm_moved` (and fist vs fist_moved) need visible motion; a still hand will not get the `_moved` label.
- `C` vs open palm can still mix if the curve is shallow.
- The CNN’s own accuracy should be judged with `evaluate_model.py` on Leap images, not only by watching the webcam.

---

## Files produced

| Path | Meaning |
| --- | --- |
| `best_gesture_model.h5` | Best CNN weights |
| `models/hand_landmarker.task` | MediaPipe detector (auto-download) |

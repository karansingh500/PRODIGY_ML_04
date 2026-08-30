import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from config import GESTURES, MODEL_SAVE_PATH, NUM_CLASSES, README_PATH, RESULTS_DIR
from data_loader import load_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EVAL_START = "<!-- prodigy-eval-start -->"
EVAL_END = "<!-- prodigy-eval-end -->"


def _short_names():
    return [name.split("_", 1)[1] for name, _ in sorted(GESTURES.items(), key=lambda item: item[1])]


def _full_names():
    return [name for name, _ in sorted(GESTURES.items(), key=lambda item: item[1])]


def save_confusion_matrix(cm, labels, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    ax.set_title("Confusion matrix (LeapGestRecog test split)")

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_per_class_bars(report, labels, path):
    precision = [report[name]["precision"] for name in labels]
    recall = [report[name]["recall"] for name in labels]
    f1 = [report[name]["f1-score"] for name in labels]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#4C78A8")
    ax.bar(x, recall, width, label="Recall", color="#F58518")
    ax.bar(x + width, f1, width, label="F1-score", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels([name.split("_", 1)[1] for name in labels], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall, and F1 (test split)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _top_confusions(cm, labels, k=3):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append((int(cm[i, j]), labels[i], labels[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def build_readme_section(accuracy, report, cm, class_names, cm_rel, bars_rel):
    short = [n.split("_", 1)[1] for n in class_names]
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])
    f1s = [(class_names[i], report[class_names[i]]["f1-score"]) for i in range(len(class_names))]
    best = max(f1s, key=lambda item: item[1])
    worst = min(f1s, key=lambda item: item[1])
    confusions = _top_confusions(cm, short)

    confusion_lines = (
        "\n".join(
            f"- `{true_name}` predicted as `{pred_name}`: **{count}** test image{'s' if count != 1 else ''}"
            for count, true_name, pred_name in confusions
        )
        if confusions
        else "- No off-diagonal mistakes on this split."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""{EVAL_START}
## Evaluation results (auto-generated)

This block is **rewritten every time** you run `evaluate_model.py`. Charts are saved under `{RESULTS_DIR}/` and embedded below.

These numbers are from the **held-out 20% LeapGestRecog test split** (same stratified split as training: `random_state=42`). They measure the CNN on infrared dataset images, **not** the webcam demo.

Last run: {stamp}

| Metric | Value |
| --- | --- |
| Test accuracy | **{accuracy * 100:.2f}%** |
| Macro F1 | {macro_f1:.4f} |
| Weighted F1 | {weighted_f1:.4f} |
| Strongest class (F1) | `{best[0]}` ({best[1]:.4f}) |
| Weakest class (F1) | `{worst[0]}` ({worst[1]:.4f}) |

### Confusion matrix

Each **row** is the true gesture; each **column** is what the CNN predicted. A dark diagonal means most Leap test images are classified correctly. Off-diagonal cells are mix-ups. Classes that look similar in a still photo (`fist` vs `fist_moved`, `palm` vs `palm_moved`) often share a few mistakes because the dataset labels motion that a single frame barely shows.

![Confusion matrix]({cm_rel})

Largest mix-ups on this run:

{confusion_lines}

### Per-class precision, recall, and F1

- **Precision** — of the images the model called this class, how many really were.
- **Recall** — of the true images of this class, how many the model found.
- **F1** — balance of the two.

Bars near 1.0 mean that class is easy on the Leap test set. A lower bar is the class to inspect first (often a motion variant or a pose that overlaps another, such as `c` vs `palm`).

![Per-class metrics]({bars_rel})

### How to read this vs the webcam

High scores here only mean the CNN matches LeapGestRecog edges. A color webcam is a different domain, which is why `predict_webcam.py` uses MediaPipe landmarks for the live label and still shows the CNN edge crop in a debug window.

{EVAL_END}
"""


def update_readme(section):
    if os.path.exists(README_PATH):
        with open(README_PATH, "r", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = "# PRODIGY_ML_04\n"

    start = text.find(EVAL_START)
    end = text.find(EVAL_END)
    if start != -1 and end != -1 and end > start:
        text = text[:start] + section.rstrip() + "\n" + text[end + len(EVAL_END) :].lstrip("\n")
        if not text.endswith("\n"):
            text += "\n"
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + section
        if not text.endswith("\n"):
            text += "\n"

    with open(README_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)


def evaluate():
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model not found at {MODEL_SAVE_PATH}. Train the model first.")
        return

    print("Loading test data...")
    _, X_test, _, y_test = load_data()
    if X_test is None:
        return

    print(f"\nLoading saved model: {MODEL_SAVE_PATH}...")
    model = load_model(MODEL_SAVE_PATH)

    print("\nGenerating predictions. Please wait...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    class_names = _full_names()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    bars_path = os.path.join(RESULTS_DIR, "per_class_metrics.png")
    cm_rel = cm_path.replace("\\", "/")
    bars_rel = bars_path.replace("\\", "/")

    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    accuracy = float(report["accuracy"])

    print("\n" + classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))
    print("Saving charts...")
    save_confusion_matrix(cm, _short_names(), cm_path)
    save_per_class_bars(report, class_names, bars_path)

    section = build_readme_section(accuracy, report, cm, class_names, cm_rel, bars_rel)
    update_readme(section)
    print(f"Wrote {cm_path}")
    print(f"Wrote {bars_path}")
    print(f"Updated {README_PATH} with graphs and explanation.")


if __name__ == "__main__":
    evaluate()

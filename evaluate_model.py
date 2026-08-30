import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data
from config import GESTURES, MODEL_SAVE_PATH, NUM_CLASSES

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

    class_names = [name for name, _ in sorted(GESTURES.items(), key=lambda item: item[1])]

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (Per-Class Accuracy)")
    print("=" * 50)
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(f"{'':<15} " + " ".join([f"{i:>5}" for i in range(NUM_CLASSES)]))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<15} " + " ".join([f"{val:>5}" for val in row]))


if __name__ == "__main__":
    evaluate()

import os

# Paths
BASE_PATH = 'dataset/leapGestRecog'
MODEL_SAVE_PATH = 'best_gesture_model.h5'

# Image Parameters
IMG_SIZE = 64 # 64x64 is optimal for this dataset while saving VRAM
CHANNELS = 1  # Grayscale

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# Gesture Mapping
GESTURES = {
    '01_palm': 0, '02_l': 1, '03_fist': 2, '04_fist_moved': 3, 
    '05_thumb': 4, '06_index': 5, '07_ok': 6, '08_palm_moved': 7, 
    '09_c': 8, '10_down': 9
}

NUM_CLASSES = len(GESTURES)
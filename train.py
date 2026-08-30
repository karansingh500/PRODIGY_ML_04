import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_loader import load_data
from model import build_model
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH

def train():
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    model = build_model()
    
    # Updated compile method using base tf.keras
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 3. Callbacks (Best Practices)
    callbacks = [
        # Save the best model only
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        # Stop training if it stops improving for 5 epochs
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        # Reduce learning rate if validation loss plateaus
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    # 4. Train Model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # 5. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    train()
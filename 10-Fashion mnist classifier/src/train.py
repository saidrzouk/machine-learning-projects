import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def ensure_dirs(root: Path):
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)


def build_model():
    model_guide = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ], name='CNN_FashionMNIST_Optimised')
    
    optimizer = Adam(learning_rate=0.001)

    model_guide.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_guide


def main():
    root = Path(__file__).resolve().parents[1]
    ensure_dirs(root)

    (X_train, y_train), (X_test, y_test) = load_data()

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    train_images_f = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images_f  = X_test.reshape(-1, 28, 28, 1).astype('float32')  / 255.0
    model_guide = build_model()
    
    callback = EarlyStopping(monitor='loss',patience=3,verbose=0,restore_best_weights=True)
    
    model_guide.fit(
        train_images_f, y_train,
        epochs=12,
        batch_size=128,
        validation_split=0.2,
        callbacks=[callback],
        verbose=1
    )

    test_loss, test_acc = model_guide.evaluate(test_images_f, y_test, verbose=0)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc*100:.2f}%")

    model_path = root / "models" / "fashion_model.keras"
    model_guide.save(model_path)
    print(f"Saved model -> {model_path}")

    report = {
        "loss": float(test_loss),
        "test_accuracy": test_acc,
        "class_names": [c for c in classes],
    }
    report_path = root / "results" / "fashion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()

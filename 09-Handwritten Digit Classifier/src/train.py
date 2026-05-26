import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def ensure_dirs(root: Path):
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)


def build_model():
    model = Sequential(
        [
            Input(shape=(8, 8)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    root = Path(__file__).resolve().parents[1]
    ensure_dirs(root)

    digits = load_digits()
    X = digits.images / 16.0
    y = digits.target

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    loss, test_accuracy_eval = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    test_pred = np.argmax(predictions, axis=1)
    test_acc = float(accuracy_score(y_test, test_pred))

    model_path = root / "models" / "digits_model.keras"
    model.save(model_path)
    print(f"Saved model -> {model_path}")

    report = {
        "loss": float(loss),
        "test_accuracy_eval": float(test_accuracy_eval),
        "test_accuracy": test_acc,
        "num_train_samples": int(x_train.shape[0]),
        "num_test_samples": int(x_test.shape[0]),
        "class_names": [str(i) for i in range(10)],
    }
    report_path = root / "results" / "digits_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()

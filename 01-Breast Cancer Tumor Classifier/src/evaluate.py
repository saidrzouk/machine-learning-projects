from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from preprocess import load_dataset, prepare_features_target, split_data


def save_confusion_matrix(cm, title: str, out_path: Path, cmap: str):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    root = Path(__file__).resolve().parents[1]
    (root / "results").mkdir(exist_ok=True)

    df = load_dataset()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)

    rf_model = joblib.load(root / "models" / "random_forest.pkl")
    svm_model = joblib.load(root / "models" / "svm_pipeline.pkl")

    rf_pred = rf_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)

    rf_cm = confusion_matrix(y_test, rf_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)

    save_confusion_matrix(
        rf_cm,
        "Random Forest Confusion Matrix",
        root / "results" / "rf_confusion_matrix.png",
        "Blues",
    )
    save_confusion_matrix(
        svm_cm,
        "SVM Confusion Matrix",
        root / "results" / "svm_confusion_matrix.png",
        "Greens",
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "test_accuracy": accuracy_score(y_test, rf_pred),
                "test_f1": f1_score(y_test, rf_pred),
            },
            {
                "model": "SVM",
                "test_accuracy": accuracy_score(y_test, svm_pred),
                "test_f1": f1_score(y_test, svm_pred),
            },
        ]
    )

    print(comparison.sort_values("test_f1", ascending=False))
    comparison.to_csv(root/"results"/"model_comparison.csv")
    print("Saved evaluation plots and model comparison to results /")


if __name__ == "__main__":
    main()

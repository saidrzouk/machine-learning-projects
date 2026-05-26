import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from preprocess import load_dataset, prepare_binary_sentiment, split_data


RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> None:
    (root / "models").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    root = project_root()
    ensure_dirs(root)

    df = load_dataset()
    data = prepare_binary_sentiment(df, text_col="body", rating_col="rating")

    X_train, X_test, y_train, y_test = split_data(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    model_path = root / "models" / "amazon_sentiment_model.joblib"
    report_path = root / "results" / "metrics.json"

    joblib.dump(model, model_path)

    metrics = {
        "test_accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Negative", "Positive"],
            output_dict=True,
        ),
        "config": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "max_features": MAX_FEATURES,
            "ngram_range": NGRAM_RANGE,
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved model  -> {model_path}")
    print(f"Saved report  -> {report_path}")


if __name__ == "__main__":
    main()


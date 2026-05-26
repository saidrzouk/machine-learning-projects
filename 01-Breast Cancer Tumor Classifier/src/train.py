import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import load_dataset, prepare_features_target, split_data

def ensure_dirs(root: Path):
    (root / "models").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)


def main():
    root = Path(__file__).resolve().parents[1]
    ensure_dirs(root)

    df = load_dataset()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20],
        "criterion": ["gini", "entropy"],
    }
    rf_grid = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    rf_grid.fit(X_train, y_train)

    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True)),
        ]
    )
    svm_param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__kernel": ["linear", "rbf"],
        "svc__gamma": ["scale", "auto"],
    }
    svm_grid = GridSearchCV(
        estimator=svm_pipe,
        param_grid=svm_param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    svm_grid.fit(X_train, y_train)

    joblib.dump(rf_grid.best_estimator_, root / "models" / "random_forest.pkl")
    joblib.dump(svm_grid.best_estimator_, root / "models" / "svm_pipeline.pkl")

    metrics = {
        "rf_best_params": rf_grid.best_params_,
        "rf_best_cv_f1": float(rf_grid.best_score_),
        "svm_best_params": svm_grid.best_params_,
        "svm_best_cv_f1": float(svm_grid.best_score_),
    }

    with (root / "models" / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print("Saved models to models/")
    print("Saved CV metrics to models/metrics.json")


if __name__ == "__main__":
    main()

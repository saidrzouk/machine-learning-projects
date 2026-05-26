import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from preprocess import load_dataset, prepare_features_target, split_data


def ensure_dirs(root: Path):
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)


def build_models():
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=10000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("model", RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("model", GradientBoostingRegressor(
                    n_estimators=200,
                    random_state=42,
                )),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("model", XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=42,
                    verbosity=0,
                )),
            ]
        ),
    }


def evaluate_model(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    ensure_dirs(root)

    df = load_dataset()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)

    feature_columns = X.columns.tolist()
    models = build_models()

    results = {}
    best_name = None
    best_score = -1
    best_pipeline = None

    print(f"{'Model':25s} | {'RMSE':>10} | {'MAE':>10} | {'R2':>8}")
    print("-" * 62)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, preds)
        results[name] = metrics

        print(
            f"{name:25s} | {metrics['RMSE']:10.2f} | "
            f"{metrics['MAE']:10.2f} | {metrics['R2']:8.4f}"
        )

        if metrics["R2"] > best_score:
            best_score = metrics["R2"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained successfully.")

    model_path = root / "models" / "best_insurance_model.pkl"
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_pipeline,
            "feature_columns": feature_columns,
            "target_transform": "log1p",
        },
        model_path,
    )
    print(f"Saved model -> {model_path}")

    report = {
        "best_model": best_name,
        "metrics": results,
        "feature_columns": feature_columns,
        "target_transform": "log1p",
    }
    results_path = root / "results" / "insurance_regression_report.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report -> {results_path}")


if __name__ == "__main__":
    main()

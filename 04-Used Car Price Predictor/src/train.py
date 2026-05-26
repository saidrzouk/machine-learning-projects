import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import feature_columns, load_dataset, prepare_features_target, split_data


def ensure_dirs(root: Path):
    (root / "models").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)


def build_preprocessor():
    numeric_features = [
        "mileage",
        "age",
        "model_year",
        "horsepower",
        "has_accident",
        "clean_title_flag",
    ]
    categorical_features = ["brand", "fuel_type", "trans_simple"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        ),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    ensure_dirs(root)

    df = load_dataset()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)

    models = build_models()

    results = {}
    best_name = None
    best_score = float("-inf")
    best_pipeline = None

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

        print(f"{name:20s} -> MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | R2: {r2:.4f}")

        if r2 > best_score:
            best_score = r2
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained successfully.")

    model_path = root / "models" / "best_used_car_model.pkl"
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_pipeline,
            "feature_columns": feature_columns(),
        },
        model_path,
    )
    print(f"Saved model -> {model_path}")

    report = {
        "best_model": best_name,
        "metrics": results,
        "feature_columns": feature_columns(),
    }
    results_path = root / "results" / "regression_report.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report -> {results_path}")


if __name__ == "__main__":
    main()

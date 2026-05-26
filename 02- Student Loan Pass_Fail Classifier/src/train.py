import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report)

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

    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000),
        'Decision Tree'       : DecisionTreeClassifier(random_state=42),
        'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = accuracy_score(y_test, preds)
        print(f"{name:25s} → Accuracy: {results[name]:.4f}")

    best_name = max(results, key=results.get)
    best_model = models[best_name]
    preds = best_model.predict(X_test)

    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_test, preds, target_names=['Fail', 'Pass']))

    # save model
    model_path = root / "models" / f"{best_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved model → {model_path}")

    # save results
    report = classification_report(y_test, preds, target_names=['Fail', 'Pass'], output_dict=True)
    report["best_model"] = best_name
    results_path = root / "results" / "classification_report.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report → {results_path}")

if __name__ == "__main__":
    main()

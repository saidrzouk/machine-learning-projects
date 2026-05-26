from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import json

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_insurance_model.pkl"
REPORT_PATH = ROOT / "results" / "insurance_regression_report.json"
DATA_PATH = ROOT / "data" / "insurance.csv"

st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            margin: 0 auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            text-align: center;
            color: #4b5563;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .prediction-card {
            background: linear-gradient(135deg, #eef6ff 0%, #f8fafc 100%);
            border: 1px solid #dbeafe;
            border-radius: 20px;
            padding: 1.25rem 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='hero-title'>Medical Insurance Cost Predictor</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='hero-subtitle'>Estimate annual medical insurance charges from basic personal and health information.</div>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def load_report():
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_value(sample: dict, column: str, default=0):
    value = sample.get(column, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def engineer_features(raw_row: pd.DataFrame) -> pd.DataFrame:
    df = raw_row.copy()

    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    df["bmi_smoker"] = df["bmi"] * df["smoker"]
    df["age_smoker"] = df["age"] * df["smoker"]
    df["age2"] = df["age"] ** 2
    df["bmi2"] = df["bmi"] ** 2
    df["obese"] = (df["bmi"] >= 30).astype(int)
    df["obese_smoker"] = df["obese"] * df["smoker"]

    return df


bundle = load_model_bundle()
df_ref = load_reference_data()
report = load_report()

if bundle is None:
    st.error("No saved model found in /models. Run `python src/train.py` first.")
    st.stop()

feature_columns = bundle.get("feature_columns") or []
sample = df_ref.iloc[0].to_dict()

with st.sidebar:
    st.header("Model Info")
    st.write(f"Best model: `{bundle.get('model_name', 'Unknown')}`")
    if report is not None:
        st.write("Validation metrics:")
        metrics = report.get("metrics", {})
        best_model = report.get("best_model")
        if best_model and best_model in metrics:
            best_metrics = metrics[best_model]
            r2 = best_metrics.get("R2")
            rmse = best_metrics.get("RMSE")
            mae = best_metrics.get("MAE")
            if isinstance(r2, (int, float)):
                st.write(f"R2: `{r2:.4f}`")
            if isinstance(rmse, (int, float)):
                st.write(f"RMSE: `${rmse:.2f}`")
            if isinstance(mae, (int, float)):
                st.write(f"MAE: `${mae:.2f}`")

st.subheader("Personal and Health Inputs")
with st.form("insurance_form"):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=int(sample_value(sample, "age", 39)), step=1)
        sex = st.selectbox("Sex", ["female", "male"], index=0 if sample.get("sex", "female") == "female" else 1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(sample_value(sample, "bmi", 30.0)), step=0.1)
        children = st.number_input("Children", min_value=0, max_value=10, value=int(sample_value(sample, "children", 0)), step=1)

    with c2:
        smoker = st.selectbox("Smoker", ["no", "yes"], index=0 if sample.get("smoker", "no") == "no" else 1)
        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"],
            index=["northeast", "northwest", "southeast", "southwest"].index(sample.get("region", "southeast")),
        )

    submitted = st.form_submit_button("Predict charges", type="primary")

if submitted:
    raw_input_row = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )

    model_input = engineer_features(raw_input_row)
    model_input = model_input.reindex(columns=feature_columns, fill_value=0)
    model_input = model_input.apply(pd.to_numeric, errors="coerce").fillna(0)

    prediction_log = bundle["pipeline"].predict(model_input)[0]
    prediction = float(np.expm1(prediction_log)) if bundle.get("target_transform") == "log1p" else float(prediction_log)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.metric("Predicted annual insurance cost", f"${prediction:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Input Summary")
    st.dataframe(raw_input_row, width='stretch', hide_index=True)

    st.subheader("Model Features")
    st.dataframe(model_input, width='stretch', hide_index=True)

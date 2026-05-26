from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_concrete_model.pkl"
REPORT_PATH = ROOT / "results" / "regression_report.json"
DATA_PATH = ROOT / "data" / "concrete_data.csv"

st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")

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
    "<div class='hero-title'>Concrete Compressive Strength Predictor</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='hero-subtitle'>Estimate compressive strength in MPa from the concrete mix design.</div>",
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



bundle = load_model_bundle()
df_ref = load_reference_data()

if bundle is None:
    st.error("No saved model found in /models. Run `python src/train.py` first.")
    st.stop()

feature_columns = bundle.get("feature_columns") or [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

sample = df_ref.iloc[0].to_dict()

def sample_value(column: str, default=0):
    value = sample.get(column, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default





st.subheader("Mix Design Inputs")
with st.form("concrete_form"):
    c1, c2 = st.columns(2)

    with c1:
            cement = st.number_input("Cement", min_value=0.0, value=sample_value("cement", 540.0), step=1.0)
            blast_furnace_slag = st.number_input("Blast furnace slag", min_value=0.0, value=sample_value("blast_furnace_slag", 0.0), step=1.0)
            fly_ash = st.number_input("Fly ash", min_value=0.0, value=sample_value("fly_ash", 0.0), step=1.0)
            water = st.number_input("Water", min_value=0.0, value=sample_value("water", 162.0), step=1.0)

    with c2:
            superplasticizer = st.number_input("Superplasticizer", min_value=0.0, value=sample_value("superplasticizer", 2.5), step=0.1)
            coarse_aggregate = st.number_input("Coarse aggregate", min_value=0.0, value=sample_value("coarse_aggregate", 1040.0), step=1.0)
            fine_aggregate = st.number_input("Fine aggregate", min_value=0.0, value=sample_value("fine_aggregate", 676.0), step=1.0)
            age = st.number_input("Age (days)", min_value=1, value=int(sample_value("age", 28)), step=1)

    submitted = st.form_submit_button("Predict strength", type="primary")

if submitted:
    input_row = pd.DataFrame(
        [
            {
                "cement": cement,
                "blast_furnace_slag": blast_furnace_slag,
                "fly_ash": fly_ash,
                "water": water,
                "superplasticizer": superplasticizer,
                "coarse_aggregate": coarse_aggregate,
                "fine_aggregate": fine_aggregate,
                "age": age,
            }
        ]
    )

    for column in feature_columns:
        if column not in input_row.columns:
            input_row[column] = 0

    input_row = input_row[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    prediction = bundle["pipeline"].predict(input_row)[0]

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.metric("Predicted compressive strength", f"{prediction:.2f} MPa")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Input Summary")
    st.dataframe(input_row, use_container_width=True, hide_index=True)


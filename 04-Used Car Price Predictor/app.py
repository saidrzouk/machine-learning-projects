from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_dataset, feature_columns, load_dataset


st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            margin: 0 auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>Used Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:1.05rem;'>Estimate a used car price from the raw listing details.</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_reference_bundle():
    model_path = ROOT / "models" / "best_used_car_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_reference_data():
    return load_dataset()




model_bundle = load_reference_bundle()
df_ref = load_reference_data()

if model_bundle is None:
    st.error("No saved model found in /models. Run `python src/train.py` first.")
    st.stop()

sample = df_ref.iloc[0].to_dict()

brand_options = sorted(df_ref["brand"].dropna().astype(str).unique().tolist())
fuel_options = sorted(df_ref["fuel_type"].dropna().astype(str).unique().tolist())
trans_options = sorted(df_ref["transmission"].dropna().astype(str).unique().tolist())
accident_options = sorted(df_ref["accident"].dropna().astype(str).unique().tolist())
title_options = ["Yes", "No"]

st.subheader("Listing Details")
with st.form("price_form"):
    c1, c2 = st.columns(2)
    with c1:
            brand = st.selectbox("Brand", brand_options, index=brand_options.index(str(sample["brand"])) if str(sample["brand"]) in brand_options else 0)
            model = st.text_input("Model", value=str(sample["model"]))
            model_year = st.number_input(
                "Model year",
                min_value=1980,
                max_value=2025,
                value=int(sample["model_year"]),
                step=1,
            )
            mileage = st.number_input(
                "Mileage (miles)",
                min_value=0,
                max_value=500000,
                value=int(str(sample["milage"]).replace(",", "").split()[0]),
                step=1000,
            )
            fuel_type = st.selectbox("Fuel type", fuel_options, index=fuel_options.index(str(sample["fuel_type"])) if str(sample["fuel_type"]) in fuel_options else 0)
            transmission = st.selectbox("Transmission", trans_options, index=trans_options.index(str(sample["transmission"])) if str(sample["transmission"]) in trans_options else 0)

    with c2:
            engine = st.text_input("Engine", value=str(sample["engine"]))
            ext_col = st.text_input("Exterior color", value=str(sample["ext_col"]))
            int_col = st.text_input("Interior color", value=str(sample["int_col"]))
            accident = st.selectbox("Accident history", accident_options, index=accident_options.index(str(sample["accident"])) if str(sample["accident"]) in accident_options else 0)
            clean_title = st.selectbox("Clean title", title_options, index=0 if str(sample["clean_title"]).lower() == "yes" else 1)

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    record = {
        "brand": brand,
        "model": model,
        "model_year": int(model_year),
        "milage": f"{int(mileage):,} mi.",
        "fuel_type": fuel_type,
        "engine": engine,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "clean_title": clean_title,
    }

    input_df = clean_dataset(pd.DataFrame([record]), drop_price_outliers=False)
    input_df = input_df[feature_columns()]
    price = model_bundle["pipeline"].predict(input_df)[0]

    st.divider()
    st.metric("Predicted price", f"${price:,.0f}")

    summary_col, details_col = st.columns([0.8, 1.2])
    with summary_col:
        st.subheader("Prediction Summary")
        st.write(f"**Brand:** {brand}")
        st.write(f"**Model:** {model}")
        st.write(f"**Year:** {model_year}")
        st.write(f"**Mileage:** {int(mileage):,} miles")
        st.write(f"**Fuel type:** {fuel_type}")
        st.write(f"**Transmission:** {transmission}")
        st.write(f"**Clean title:** {clean_title}")

    with details_col:
        st.subheader("Input Record")
        st.dataframe(pd.DataFrame([record]), use_container_width=True, hide_index=True)

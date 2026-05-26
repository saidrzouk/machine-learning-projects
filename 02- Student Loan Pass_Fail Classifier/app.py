from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Pass Classifier", page_icon="🎓", layout="wide")

ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_PATH  = ROOT / "data" / "student-por.csv"

st.title("🎓 Student Pass/Fail Classifier")
st.write("Enter student information to predict whether they will pass or fail Portuguese.")

@st.cache_resource
def load_model():
    path = MODELS_DIR / "random_forest.pkl"
    return joblib.load(path) if path.exists() else None

@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['G3'])       
    return df

model = load_model()
df_ref = load_reference_data()

if model is None:
    st.error("No saved model found in /models. Run: python src/train.py")
    st.stop()

cat_cols = df_ref.select_dtypes(include='object').columns.tolist()
num_cols = df_ref.select_dtypes(include='number').columns.tolist()
feature_cols = [c for c in df_ref.columns if c != 'pass']

st.subheader("Input Features")
inputs = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_cols):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        if feature in cat_cols:
            options = sorted(df_ref[feature].dropna().unique().tolist())
            inputs[feature] = st.selectbox(feature, options)
        else:
            median_val = float(df_ref[feature].median())
            inputs[feature] = st.number_input(feature, value=median_val, format="%.2f")

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([inputs])

    le = LabelEncoder()
    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = le.fit_transform(
                pd.concat([df_ref[col], input_df[col]], ignore_index=True)
            ).tolist()[-1:]

    pred = model.predict(input_df)[0]
    label = "✅ Pass" if pred == 1 else "❌ Fail"

    st.divider()
    st.subheader(f"Prediction: **{label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        col1, col2 = st.columns(2)
        col1.metric("Fail Probability",  f"{proba[0]:.1%}")
        col2.metric("Pass Probability",  f"{proba[1]:.1%}")
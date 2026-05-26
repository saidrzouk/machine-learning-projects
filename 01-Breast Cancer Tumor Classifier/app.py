from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "datadata.csv"

st.title("Breast Cancer Tumor Classifier")
st.write("Enter tumor feature values and predict whether the tumor is benign or malignant.")

@st.cache_resource
def load_models():
    rf_path = MODELS_DIR / "random_forest.pkl"
    svm_path = MODELS_DIR / "svm_pipeline.pkl"
    rf = joblib.load(rf_path) if rf_path.exists() else None
    svm = joblib.load(svm_path) if svm_path.exists() else None
    return rf, svm

@st.cache_data
def load_feature_columns():
    df = pd.read_csv(DATA_PATH).drop(columns=["id", "Unnamed: 32"], errors="ignore")
    return [c for c in df.columns if c != "diagnosis"], df

rf_model, svm_model = load_models()
feature_cols, df_ref = load_feature_columns()

if rf_model is None and svm_model is None:
    st.error("No saved models found in /models. Run: python src/train.py")
    st.stop()

model_choice = st.selectbox(
    "Select model",
    [m for m in ["Random Forest", "SVM"] if (m == "Random Forest" and rf_model) or (m == "SVM" and svm_model)]
)

st.subheader("Input Features")
inputs = {}

# Use dataset medians as defaults
medians = df_ref[feature_cols].median(numeric_only=True)

col1, col2 = st.columns(2)
for i, feature in enumerate(feature_cols):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        inputs[feature] = st.number_input(
            feature,
            value=float(medians[feature]),
            format="%.6f"
        )

input_df = pd.DataFrame([inputs])

if st.button("Predict"):
    model = rf_model if model_choice == "Random Forest" else svm_model
    pred = model.predict(input_df)[0]
    label = "Malignant" if pred == 1 else "Benign"
    st.success(f"Prediction: **{label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        st.write(f"Benign probability: `{proba[0]:.3f}`")
        st.write(f"Malignant probability: `{proba[1]:.3f}`")
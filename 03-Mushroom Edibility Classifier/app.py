from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄", layout="wide")

ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_PATH  = ROOT / "data" / "mushrooms.csv"

st.title("🍄 Mushroom Edibility Classifier")
st.write("Select mushroom features to predict whether it is edible or poisonous.")

# ── Loaders ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = MODELS_DIR / "decision_tree.pkl"
    return joblib.load(path) if path.exists() else None

@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['class', 'veil-type']) 
    return df

# ── Load ──────────────────────────────────────────────────────────
model  = load_model()
df_ref = load_reference_data()

if model is None:
    st.error("No saved model found in /models. Run: python src/train.py")
    st.stop()

cat_cols     = df_ref.select_dtypes(include='object').columns.tolist()
feature_cols = df_ref.columns.tolist()

# ── Input Form ────────────────────────────────────────────────────
st.subheader("Mushroom Features")
inputs = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_cols):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        options = sorted(df_ref[feature].dropna().unique().tolist())
        inputs[feature] = st.selectbox(feature, options)

# ── Encode & Predict ──────────────────────────────────────────────
if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([inputs])

    le = LabelEncoder()
    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = le.fit_transform(
                pd.concat([df_ref[col], input_df[col]], ignore_index=True)
            ).tolist()[-1:]

    pred = model.predict(input_df)[0]

    st.divider()

    if pred == 0:
        st.success("### ✅ Edible")
        st.write("This mushroom is predicted to be **safe to eat**.")
    else:
        st.error("### ☠️ Poisonous")
        st.warning("⚠️ This mushroom is predicted to be **poisonous**. Do NOT consume it.")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        c1, c2 = st.columns(2)
        c1.metric("Edible Probability",    f"{proba[0]:.1%}")
        c2.metric("Poisonous Probability", f"{proba[1]:.1%}")
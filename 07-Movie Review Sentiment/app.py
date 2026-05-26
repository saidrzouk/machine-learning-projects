from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from src.preprocess import clean_text


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"


st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def predict_sentiment(review: str, model):
    cleaned = clean_text(review)
    probability = float(model.predict_proba([cleaned])[0][1])
    label = "Positive" if probability >= 0.5 else "Negative"
    confidence = probability if label == "Positive" else 1.0 - probability
    return label, probability, confidence, cleaned


def reset_review():
    st.session_state["review_input"] = ""


st.title("Movie Review Sentiment Analyzer")
st.write("Enter a review below and the app will predict whether it is positive or negative.")

model = load_model()
metrics = load_metrics()

if model is None:
    st.warning("Saved model artifacts are missing. Run `python src/train.py` first.")
    st.stop()

with st.sidebar:
    st.subheader("Model Info")
    if metrics:
        test_accuracy = metrics.get("test_accuracy", "n/a")
        if isinstance(test_accuracy, (int, float)):
            st.write(f"Test accuracy: `{test_accuracy:.4f}`")
        else:
            st.write(f"Test accuracy: `{test_accuracy}`")
        st.write(f"Max features: `{metrics.get('config', {}).get('max_features', 'n/a')}`")

review = st.text_area(
    "Movie review",
    height=200,
    placeholder="Write a review here, for example: The movie was surprisingly emotional and well acted.",
    key="review_input",
)

col1, col2 = st.columns(2)

with col1:
    run_prediction = st.button("Predict", type="primary", use_container_width=True)

with col2:
    st.button("Clear", use_container_width=True, on_click=reset_review)

if run_prediction:
    review_text = st.session_state.get("review_input", "")
    if not review_text.strip():
        st.error("Please enter a review first.")
    else:
        label, probability, confidence, cleaned = predict_sentiment(review_text, model)

        if label == "Positive":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

        with st.expander("See cleaned input"):
            st.code(cleaned or "(empty after preprocessing)")

        st.caption(
            f"Raw positive probability: {probability:.3f}. Values above 0.5 are treated as Positive."
        )

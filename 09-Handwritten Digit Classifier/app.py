from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "digits_model.keras"
CLASS_NAMES = [str(i) for i in range(10)]

st.set_page_config(page_title="Digits Recognition", layout="wide")
st.title("Sklearn Digits Recognition")
st.write("Upload an image of a handwritten digit and predict 0 to 9.")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image):
    img = image.convert("L").resize((8, 8))
    arr = np.array(img).astype("float32")
    arr = 16.0 - ((arr / 255.0) * 16.0)
    arr = np.clip(arr, 0.0, 16.0)
    return (arr / 16.0).reshape(1, 8, 8)


model = get_model()
if model is None:
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=280)

    input_tensor = preprocess_image(image)
    probs = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    st.subheader("Prediction")
    st.metric("Predicted class", pred_label)
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.subheader("All class probabilities")
    prob_table = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}
    st.json(prob_table)

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fashion_model.keras"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.set_page_config(page_title="Fashion Recognition", layout="wide")
st.title("Fashion MNIST Recognition")
st.write("Upload an image of clothing")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image):
    img = image.convert("L").resize((28, 28))          
    arr = np.array(img).astype("float32")
    arr = 255.0 - arr                                       
    arr = np.clip(arr, 0.0, 255.0)
    return (arr / 255.0).reshape(1, 28, 28, 1)         


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
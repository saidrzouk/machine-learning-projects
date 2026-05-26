# Digits Image Recognition

A computer vision project that trains a sklearn model on the digits dataset and serves predictions with Streamlit.

## Project Structure

- `src/train.py` - model training and evaluation script
- `models/` - saved trained model (`digits_model.keras`)
- `results/` - evaluation report (`digits_report.json`)
- `app.py` - Streamlit prediction app
- `cifar_recognition.ipynb` - notebook experiments

## Train

```bash
python src/train.py
```

This saves:

- `models/digits_model.keras`
- `results/digits_report.json`

## Run the app

```bash
streamlit run app.py
```

## Notes

- Input images are converted to grayscale and resized to `8x8` before prediction.
- The model predicts one digit class from `0` to `9`.
- Generated model files and result reports are ignored by Git.

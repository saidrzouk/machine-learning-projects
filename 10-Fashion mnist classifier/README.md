# Fashion MNIST Clothing Classifier

A deep learning project that trains a convolutional neural network on the Fashion MNIST dataset and serves predictions with Streamlit.

## Project Structure

- `src/train.py` - model training and evaluation script
- `models/` - saved trained model (`fashion_model.keras`)
- `results/` - evaluation report (`fashion_report.json`)
- `app.py` - Streamlit prediction app
- `fashion_mnis_Clothing_classification.ipynb` - notebook experiments

## Train

```bash
python src/train.py
```

This saves:

- `models/fashion_model.keras`
- `results/fashion_report.json`

## Run the app

```bash
streamlit run app.py
```

## Notes

- The model predicts one clothing category from 10 Fashion MNIST classes.
- Uploaded images are converted to grayscale, resized to `28x28`, and normalized before prediction.
- There are multiple notebook copies in this folder, so keeping one final notebook name later would make the project cleaner on GitHub.

# Mushroom Edibility Classifier

A small machine learning project that predicts whether a mushroom is edible or poisonous using the UCI mushroom dataset.

## Project Structure

- `data/` - raw dataset files
- `src/` - preprocessing and training scripts
- `models/` - saved trained models
- `results/` - evaluation outputs such as reports
- `app.py` - Streamlit demo app
- `Mushroom_Edibility_Classifier.ipynb` - exploratory analysis notebook

## What the notebook covers

- dataset loading and inspection
- class balance checks
- feature visualizations
- preprocessing and encoding
- train/test split
- model benchmarking
- evaluation plots and feature importance

## Setup

Install the project dependencies, then run the training script to generate a model and evaluation report.

```bash
python src/train.py
```

## Run the app

After training, launch the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- The `veil-type` feature is dropped because it is constant in this dataset.
- The notebook treats `?` in `stalk-root` as missing/unknown during analysis.
- Generated model artifacts and evaluation files are ignored by Git.

# Concrete Compressive Strength Predictor

A machine learning project that estimates concrete compressive strength from the concrete mix design.

## Project Structure

- `data/` - raw dataset files
- `src/` - preprocessing and training scripts
- `models/` - saved trained model bundles
- `results/` - evaluation outputs such as reports
- `app.py` - Streamlit demo app
- `Concrete_Compressive_Strength_Predictor.ipynb` - exploratory analysis and model experimentation notebook

## What the notebook covers

- dataset inspection and descriptive statistics
- feature distribution plots
- train/test split and scaling
- regression model benchmarking
- evaluation with MAE, RMSE, and R2

## Setup

Train the model bundle first:

```bash
python src/train.py
```

This creates:

- `models/best_concrete_model.pkl`
- `results/regression_report.json`

## Run the app

After training, launch the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- The dataset target is `concrete_compressive_strength`.
- The preprocessing code strips the trailing space from `fine_aggregate ` in the CSV header.
- Generated model files and evaluation reports are ignored by Git.


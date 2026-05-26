# Medical Insurance Cost Predictor

A machine learning project that estimates annual medical insurance charges from basic personal and health information.

## Project Structure

- `data/` - raw dataset files
- `src/` - preprocessing and training scripts
- `models/` - saved trained model bundles
- `results/` - evaluation outputs such as reports
- `app.py` - Streamlit demo app
- `Medical_Insurance_Cost.ipynb` - exploratory analysis and model experimentation notebook

## What the notebook covers

- dataset inspection and descriptive statistics
- target distribution analysis
- feature engineering for insurance charge prediction
- train/test split and scaling for linear models
- regression model benchmarking
- evaluation with MAE, RMSE, and R2

## Models included in training

The training script benchmarks the same models used in the notebook:

- `Linear Regression`
- `Ridge`
- `Lasso`
- `Random Forest`
- `Gradient Boosting`
- `XGBoost`

## Setup

Train the model bundle first:

```bash
python src/train.py
```

This creates:

- `models/best_insurance_model.pkl`
- `results/insurance_regression_report.json`

## Run the app

After training, launch the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- The dataset target is `charges`.
- The notebook and training pipeline use a `log1p` transform on the target for modeling.
- The app converts predicted log values back to dollar amounts for display.
- Generated model files and evaluation reports are ignored by Git.

# Used Car Price Predictor

A machine learning project that estimates the price of a used car from its listing details.

## Project Structure

- `data/` - raw dataset files
- `images/` - saved notebook plots and visualizations
- `src/` - preprocessing, training, and prediction scripts
- `models/` - saved trained model bundles
- `results/` - evaluation outputs such as reports
- `app.py` - Streamlit demo app
- `Used_Car_Price_Predictor.ipynb` - exploratory analysis and model experimentation notebook

## What the notebook covers

- dataset inspection and cleaning
- feature engineering for mileage, age, horsepower, and listing flags
- price distribution analysis and brand comparisons
- train/test split and model benchmarking
- regression evaluation with MAE, RMSE, and R2

## Setup

Install the project dependencies, then train a model bundle:

```bash
python src/train.py
```

This creates:

- `models/best_used_car_model.pkl`
- `results/regression_report.json`

## Run the app

After training, launch the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- The raw dataset uses `milage` in the CSV, and the preprocessing code normalizes it to `mileage`.
- The model uses the cleaned fields and derived features from `src/preprocess.py`.
- Generated model files and evaluation reports are ignored by Git.
- Saved plots live in `images/`.

# Student Loan Pass/Fail Classifier

This project trains a classification model to predict student pass/fail outcomes and serves predictions through a Streamlit app.

## Requirements

- Python 3.10+
- pip

Python packages:

- pandas
- scikit-learn
- streamlit
- joblib

Install dependencies:

```bash
pip install pandas scikit-learn streamlit joblib
```

## Project Structure

- `app.py`: Streamlit inference app
- `src/preprocess.py`: Data loading and preprocessing
- `src/train.py`: Model training and evaluation
- `data/student-por.csv`: Input dataset
- `models/`: Saved trained models
- `results/`: Evaluation report outputs

## Run

Train models:

```bash
python src/train.py
```

Run app:

```bash
streamlit run app.py
```

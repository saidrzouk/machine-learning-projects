# Breast Cancer Tumor Classifier

This project predicts whether a breast tumor is benign (`B`) or malignant (`M`) using the Wisconsin Breast Cancer dataset.

## Project Structure

```text
Breast Cancer Tumor Classifier/
├── data.csv
├── Breast_Cancer_Tumor_Classifier.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── random_forest.pkl
│   ├── svm_pipeline.pkl
│   └── metrics.json
└── results/
    ├── rf_confusion_matrix.png
    ├── svm_confusion_matrix.png
    └── model_comparison.csv
```

## What This Project Includes

- Exploratory data analysis (in notebook)
- Feature correlation and visualization
- Two classifiers:
  - Random Forest
  - SVM (with `StandardScaler` in a pipeline)
- Hyperparameter tuning with `GridSearchCV`
- Saved best models
- Saved evaluation plots and metrics

## Installation

```bash
pip install -r requirements.txt
```

## Run From Terminal

From the project root:

```bash
python src/train.py
python src/evaluate.py
```

This will:
- train and tune both models,
- save models in `models/`,
- save confusion matrix plots and comparison table in `results/`.

## Dataset

- File used: `data.csv`
- Target column: `diagnosis` (`B` = benign, `M` = malignant)

## Notes

- Random Forest does not require scaling.
- SVM is trained with scaling via a pipeline to avoid data leakage.
- Random seeds are fixed for reproducibility.

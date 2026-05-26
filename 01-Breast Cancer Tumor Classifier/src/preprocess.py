from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_name: str = "data/data.csv") -> pd.DataFrame:
    csv_path = project_root() / csv_name
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    return df


def prepare_features_target(df: pd.DataFrame):
    model_df = df.copy()
    model_df["diagnosis"] = model_df["diagnosis"].map({"B": 0, "M": 1})
    X = model_df.drop(columns="diagnosis")
    y = model_df["diagnosis"]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

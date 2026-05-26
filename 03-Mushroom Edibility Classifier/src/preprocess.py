from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_name: str = "data/mushrooms.csv") -> pd.DataFrame:
    csv_path = project_root() / csv_name
    df = pd.read_csv(csv_path)
    return df


def prepare_features_target(df: pd.DataFrame):
    df = df.copy()

    df = df.drop(columns=['veil-type'])
    le = LabelEncoder()
    df_enc = df.apply(le.fit_transform)

    X = df_enc.drop(columns=['class'])
    y = df_enc['class']
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 2):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
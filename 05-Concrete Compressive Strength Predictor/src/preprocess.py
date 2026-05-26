from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "concrete_compressive_strength"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_name: str = "data/concrete_data.csv") -> pd.DataFrame:
    csv_path = project_root() / csv_name
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def prepare_features_target(df: pd.DataFrame):
    df = df.copy()

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

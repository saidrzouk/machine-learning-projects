from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_name: str = "data/student-por.csv") -> pd.DataFrame:
    csv_path = project_root() / csv_name
    df = pd.read_csv(csv_path)
    return df


def prepare_features_target(df: pd.DataFrame):
    df = df.copy()
    df['pass'] = (df['G3'] >= 10).astype(int)

    df_model = df.drop(columns=['G3'])

    cat_cols = df_model.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop(columns=['pass'])
    y = df_model['pass']
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
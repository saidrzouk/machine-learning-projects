from pathlib import Path

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN ="log_charges"

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_name: str = "data/insurance.csv") -> pd.DataFrame:
    csv_path = project_root() / csv_name
    df = pd.read_csv(csv_path)
    return df

def prepare_features_target(df: pd.DataFrame):
    df = df.copy()

    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex"]    = df["sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    df["bmi_smoker"]    = df["bmi"] * df["smoker"]      
    df["age_smoker"]    = df["age"] * df["smoker"]
    df["age2"]          = df["age"] ** 2               
    df["bmi2"]          = df["bmi"] ** 2
    df["obese"]         = (df["bmi"] >= 30).astype(int)
    df["obese_smoker"]  = df["obese"] * df["smoker"] 

    df["log_charges"] = np.log1p(df["charges"])

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
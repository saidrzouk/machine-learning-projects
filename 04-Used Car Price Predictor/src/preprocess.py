from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REFERENCE_YEAR = 2024


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(csv_name: str = "data/used_cars.csv") -> pd.DataFrame:
    csv_path = project_root() / csv_name
    return pd.read_csv(csv_path)


def simplify_transmission(value) -> str:
    text = str(value).lower()
    if "a/t" in text or "auto" in text or "dual shift" in text:
        return "Automatic"
    if "m/t" in text or "manual" in text:
        return "Manual"
    return "Other"


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_dataset(
    df: pd.DataFrame,
    reference_year: int = REFERENCE_YEAR,
    drop_price_outliers: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    if "milage" in df.columns and "mileage" not in df.columns:
        df = df.rename(columns={"milage": "mileage"})

    if "price" in df.columns:
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

    if "mileage" in df.columns:
        df["mileage"] = pd.to_numeric(
            df["mileage"].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        )

    if "fuel_type" in df.columns:
        fuel = df["fuel_type"].astype(str).str.strip()
        bad_fuel = fuel.str.lower().isin({"not supported", "nan", "none"}) | (fuel.str.len() == 1)
        df["fuel_type"] = fuel.mask(bad_fuel, np.nan).fillna("Unknown")

    if "accident" in df.columns:
        df["accident"] = df["accident"].fillna("None reported")
        df["has_accident"] = np.where(
            df["accident"].astype(str).str.contains("none", case=False, na=True),
            0,
            1,
        )
    else:
        df["has_accident"] = 0

    if "clean_title" in df.columns:
        df["clean_title"] = df["clean_title"].fillna("No")
        df["clean_title_flag"] = np.where(
            df["clean_title"].astype(str).str.lower() == "yes",
            1,
            0,
        )
    else:
        df["clean_title_flag"] = 0

    if "transmission" in df.columns:
        df["trans_simple"] = df["transmission"].apply(simplify_transmission)
    else:
        df["trans_simple"] = "Other"

    if "model_year" in df.columns:
        df["model_year"] = _to_numeric(df["model_year"])
        df["age"] = reference_year - df["model_year"]
    else:
        df["age"] = np.nan

    if "engine" in df.columns:
        df["horsepower"] = pd.to_numeric(
            df["engine"].astype(str).str.extract(r"(\d+\.?\d*)HP")[0],
            errors="coerce",
        )
        if df["horsepower"].notna().any():
            df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
        else:
            df["horsepower"] = 0
    else:
        df["horsepower"] = 0

    required_columns = ["brand", "fuel_type", "mileage", "model_year", "age", "horsepower"]
    available_required = [col for col in required_columns if col in df.columns]
    if available_required:
        df = df.dropna(subset=available_required)

    if drop_price_outliers and "price" in df.columns and not df.empty:
        q_low = df["price"].quantile(0.01)
        q_high = df["price"].quantile(0.99)
        df = df[(df["price"] >= q_low) & (df["price"] <= q_high)].copy()

    return df.reset_index(drop=True)


def feature_columns() -> list[str]:
    return [
        "mileage",
        "age",
        "model_year",
        "horsepower",
        "has_accident",
        "clean_title_flag",
        "brand",
        "fuel_type",
        "trans_simple",
    ]


def prepare_features_target(
    df: pd.DataFrame,
    reference_year: int = REFERENCE_YEAR,
    drop_price_outliers: bool = True,
):
    cleaned = clean_dataset(
        df,
        reference_year=reference_year,
        drop_price_outliers=drop_price_outliers,
    )

    if "price" not in cleaned.columns:
        raise ValueError("The dataset must contain a price column for training.")

    X = cleaned[feature_columns()].copy()
    y = cleaned["price"].copy()
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

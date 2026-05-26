from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "20191226-reviews.csv"


def load_dataset(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Amazon reviews dataset."""
    return pd.read_csv(csv_path)


def clean_text(text: str) -> str:
    """Basic text normalization for review sentiment."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_binary_sentiment(
    df: pd.DataFrame,
    text_col: str = "body",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """Convert star ratings into binary sentiment labels.

    1-2 stars -> negative (0)
    4-5 stars -> positive (1)
    3 stars are dropped as neutral.
    """
    required = {text_col, rating_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df[[text_col, rating_col]].copy()
    data[text_col] = data[text_col].fillna("").astype(str)
    data[rating_col] = pd.to_numeric(data[rating_col], errors="coerce")
    data = data.dropna(subset=[text_col, rating_col])

    data = data[data[rating_col] != 3].copy()
    data["sentiment"] = (data[rating_col] >= 4).astype(int)
    data["text"] = data[text_col].apply(clean_text)
    data = data[["text", "sentiment"]].reset_index(drop=True)
    return data


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split sentiment data into train and test sets."""
    X = df["text"]
    y = df["sentiment"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


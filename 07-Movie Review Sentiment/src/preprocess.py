from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "IMDB_Dataset.csv"
NEGATIONS = {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "cannot", "can't"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS


def load_dataset(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    sample_size: int | None = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the IMDB dataset and optionally sample a smaller subset."""
    df = pd.read_csv(csv_path)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Normalize review text for tokenization."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOP_WORDS]
    return " ".join(tokens).strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the review column and convert sentiment labels to integers."""
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Expected columns 'review' and 'sentiment' in the dataset.")

    cleaned = df.copy()
    label_map = {"positive": 1, "negative": 0}
    cleaned["sentiment"] = cleaned["sentiment"].map(label_map)

    if cleaned["sentiment"].isna().any():
        raise ValueError("Unexpected sentiment values found in the dataset.")

    cleaned["sentiment"] = cleaned["sentiment"].astype(int)
    cleaned["review"] = cleaned["review"].apply(clean_text)
    return cleaned


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the cleaned dataset into train and test partitions."""
    X = df["review"]
    y = df["sentiment"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )



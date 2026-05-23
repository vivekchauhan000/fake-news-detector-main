import re
from typing import Optional

import pandas as pd


def clean_text(text: str) -> str:
    """Clean the input text for model training and prediction."""
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_path: str, text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
    """Load a CSV dataset and drop missing values."""
    df = pd.read_csv(csv_path)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Expected columns '{text_column}' and '{label_column}' in dataset.")

    df = df[[text_column, label_column]].dropna().reset_index(drop=True)
    df[text_column] = df[text_column].astype(str).map(clean_text)
    return df


def prepare_xy(df: pd.DataFrame, text_column: str = "text", label_column: str = "label"):
    """Return feature and target arrays from a cleaned DataFrame."""
    X = df[text_column].tolist()
    y = df[label_column].astype(str).tolist()
    return X, y

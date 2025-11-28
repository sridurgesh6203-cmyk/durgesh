# data_loader.py
"""
Simple dataset loader for the workshop.
Expected CSV columns: 'review' (text), optionally 'label' (gold label)
Place your Kaggle CSV in the project folder as imdb.csv (or change the path below).
"""

import pandas as pd
from typing import Tuple

DEFAULT_CSV = "imdb.csv"

def load_data(path: str = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load dataset and perform basic cleaning.
    Returns a pandas DataFrame with at least a 'review' column.
    """
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Heuristics to find text column
    text_cols = [c for c in df.columns if 'review' in c.lower() or 'text' in c.lower()]
    if not text_cols:
        # fallback: take the first string column
        string_cols = [c for c in df.columns if df[c].dtype == object]
        if not string_cols:
            raise ValueError("No text-like column found in the CSV. Ensure a 'review' column exists.")
        text_col = string_cols[0]
    else:
        text_col = text_cols[0]

    # rename to standard name
    if text_col != 'review':
        df = df.rename(columns={text_col: 'review'})

    # drop rows with missing reviews
    df = df.dropna(subset=['review']).reset_index(drop=True)

    return df

def sample_data(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return a small random sample for fast local testing / demo."""
    return df.sample(min(n, len(df))).reset_index(drop=True)

if __name__ == "__main__":
    df = load_data()
    print("Loaded dataset with", len(df), "rows")
    print(df.head())

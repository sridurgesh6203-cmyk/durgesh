# utils.py
import pandas as pd
from typing import List, Dict

def add_predictions_to_df(df: pd.DataFrame, preds: List[Dict]) -> pd.DataFrame:
    """
    Given a DataFrame with 'review' and a list of predictions (dicts with label,score),
    attach prediction columns and return new DataFrame.
    """
    df = df.copy()
    labels = [p.get('label', '') for p in preds]
    scores = [float(p.get('score', 0.0)) for p in preds]
    df['prediction'] = labels
    df['prediction_score'] = scores
    return df

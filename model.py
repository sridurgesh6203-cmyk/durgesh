# model.py
"""
Model wrapper using Hugging Face pipelines.
We use a ready-made sentiment-analysis pipeline so we don't train here.
"""

from transformers import pipeline
from typing import List, Dict
import math
from tqdm import tqdm

# Choose a well-known small finetuned model for sentiment (works out of the box)
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Initialize pipeline (cached by HF locally)
sentiment_pipeline = pipeline("sentiment-analysis", model=DEFAULT_MODEL)

def analyze_text(text: str) -> Dict:
    """
    Analyze a single text string and return label + score.
    Returns: {"label": "POSITIVE"/"NEGATIVE", "score": float}
    """
    if not isinstance(text, str) or text.strip() == "":
        return {"label": "NEUTRAL", "score": 0.0}
    out = sentiment_pipeline(text[:1000])  # cut very long input for speed
    # pipeline returns a list with dict(s)
    return out[0]

def analyze_batch(texts: List[str], batch_size: int = 16) -> List[Dict]:
    """
    Analyze a list of texts in batches (to avoid memory spikes).
    """
    results = []
    n = len(texts)
    for i in tqdm(range(0, n, batch_size), desc="Running model"):
        batch = texts[i:i+batch_size]
        res = sentiment_pipeline(batch)
        results.extend(res)
    return results

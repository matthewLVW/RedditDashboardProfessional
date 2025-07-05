# nlp_core/sentiment.py

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import logging

# Initialize global models (load once for efficiency)
_vader = SentimentIntensityAnalyzer()
try:
    _roberta_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        return_all_scores=False
    )
except Exception as e:
    logging.error(f"Failed to load RoBERTa sentiment model: {e}")
    _roberta_pipeline = None

from typing import Any, Dict, List, cast

from typing import Any, List, Dict, cast

def fused_sentiment(text: str) -> float:
    """
    Combine VADER and RoBERTa sentiment into one score [-1, 1],
    with type-checker-friendly handling of transformers' loose return types.
    """
    if not text:
        return 0.0

    # ── VADER ─────────────────────────────────────────────────────────────
    vader_score: float = _vader.polarity_scores(text)["compound"]

    # ── RoBERTa ───────────────────────────────────────────────────────────
    from collections.abc import Iterable
    roberta_score = 0.0
    if _roberta_pipeline is not None:
        try:
            raw = _roberta_pipeline(text, truncation=True)       # type: ignore[arg-type]

            # Ensure we have an iterable to pass into list()
            if isinstance(raw, Iterable):
                results: List[Any] = list(raw)
            elif raw is None:
                results = []
            else:  # Tensor or single obj → wrap in list
                results = [raw]

            if results:
                first: Any = results[0]
                conf: float

                if isinstance(first, dict):
                    conf = float(first.get("score", 0.5))
                else:  # unexpected type (e.g., Tensor) → neutral
                    conf = 0.5

                roberta_score = (conf - 0.5) * 2  # map [0,1] → [-1,1]

        except Exception:
            pass  # leave roberta_score = 0

    # ── Fuse & squash ─────────────────────────────────────────────────────
    combined = np.tanh(0.6 * vader_score + 0.4 * roberta_score)
    return float(combined)

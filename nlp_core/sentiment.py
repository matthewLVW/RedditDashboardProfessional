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

def fused_sentiment(text: str) -> float:
    """
    Compute a sentiment score by combining VADER and RoBERTa predictions.

    Args:
        text (str): Input text.

    Returns:
        float: Sentiment score in [-1, 1].
    """
    if not text:
        return 0.0
    # VADER score
    vader_score = _vader.polarity_scores(text)["compound"]
    # RoBERTa score
    if _roberta_pipeline:
        try:
            result = _roberta_pipeline(text, truncation=True)
            roberta_conf = result[0]["score"]  # between 0 and 1
            # Scale to [-1,1] around neutral
            roberta_score = (roberta_conf - 0.5) * 2
        except Exception:
            roberta_score = 0.0
    else:
        roberta_score = 0.0
    # Combine scores (weights can be tuned/calibrated)
    combined = np.tanh(0.6 * vader_score + 0.4 * roberta_score)
    return float(combined)

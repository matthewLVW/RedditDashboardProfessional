# nlp_core/sarcasm.py

from typing import Optional
from transformers import pipeline
import logging

try:
    _sarcasm_pipeline = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sarcasm-twitter")
except Exception as e:
    logging.error(f"Could not load sarcasm model: {e}")
    _sarcasm_pipeline = None

def detect_sarcasm(text: str) -> Optional[float]:
    """
    Predict sarcasm probability of a text.
    
    Args:
        text (str): Input text.

    Returns:
        Optional[float]: Probability that the text is sarcastic, or None if unavailable.
    """
    if not text or not _sarcasm_pipeline:
        return None
    try:
        result = _sarcasm_pipeline(text, truncation=True)[0]
        # Some pipelines return a label, so convert label to a score
        if "score" in result:
            score = result["score"]
        elif "label" in result:
            # If label is 'SARCASM' or similar
            score = float("SARCASM" in result["label"].upper())
        else:
            score = 0.0
        return score
    except Exception:
        return None

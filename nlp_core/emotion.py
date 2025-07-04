# nlp_core/emotion.py

from transformers import pipeline
from typing import Dict

try:
    _emotion_pipeline = pipeline(
        "text-classification",
        model="joeddav/bert-base-go-emotions",
        return_all_scores=False
    )
except Exception:
    _emotion_pipeline = None

def detect_emotions(text: str) -> Dict[str, float]:
    """
    Detect emotions in text using GoEmotions (28 classes + neutral).

    Args:
        text (str): Input text.

    Returns:
        Dict[str, float]: Mapping from emotion label to confidence score.
    """
    if not text or not _emotion_pipeline:
        return {}
    try:
        # Some emotion models output multiple labels with scores
        results = _emotion_pipeline(text, truncation=True)
        if isinstance(results, list):
            # Convert list of dicts to label:score mapping
            return {res["label"]: res["score"] for res in results}
        elif isinstance(results, dict):
            return {results["label"]: results["score"]}
        else:
            return {}
    except Exception:
        return {}

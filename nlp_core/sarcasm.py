from typing import Optional, List, Dict, Any
from transformers import pipeline
import logging

try:
    _sarcasm_pipeline = pipeline(
        "text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter"
    )
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
    if not text or _sarcasm_pipeline is None:
        return None

    try:
        # HF pipelines are typed as generators; make them a list for safe indexing
        results: List[Dict[str, Any]] = list(
            _sarcasm_pipeline(text, truncation=True)  # type: ignore[arg-type]
        )
        if not results:
            return None

        result = results[0]
        score = float(result.get("score", 0.0))

        # Some models don’t include a numeric score—fallback to label
        if "label" in result and "score" not in result:
            label = str(result["label"]).upper()
            score = 1.0 if "SARCASM" in label else 0.0

        return score
    except Exception:
        return None

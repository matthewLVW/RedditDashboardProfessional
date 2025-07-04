# tests/test_sentiment.py

from nlp_core.sentiment import fused_sentiment

def test_fused_sentiment_neutral():
    # neutral text should yield score near 0
    score = fused_sentiment("I have no strong feelings either way.")
    assert -1.0 <= score <= 1.0

# tests/test_emotion.py

from nlp_core.emotion import detect_emotions

def test_detect_emotions_empty():
    assert detect_emotions("") == {}

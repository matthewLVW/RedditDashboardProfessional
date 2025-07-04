# tests/test_cleaning.py

from nlp_core.cleaning import clean_text

def test_clean_removed_empty():
    assert clean_text("") == ""
    assert clean_text("[deleted]") == ""
    assert clean_text("[removed]") == ""

def test_clean_basic():
    text = "Hello World! Visit http://example.com 😊"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "emoji" not in cleaned
    assert cleaned == "hello world! visit"

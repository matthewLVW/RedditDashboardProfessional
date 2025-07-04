# tests/test_spacy_pipe.py

from nlp_core.spacy_pipe import process_texts

def test_process_texts_empty():
    docs = process_texts([])
    assert docs == []

def test_process_texts_non_list():
    import pytest
    with pytest.raises(TypeError):
        process_texts("not a list")

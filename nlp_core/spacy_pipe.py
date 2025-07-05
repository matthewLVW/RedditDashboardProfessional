# nlp_core/spacy_pipe.py

import spacy
from spacy.tokens import Doc
from spacy.attrs import POS
from typing import List, Dict, Union

# Load a transformer-backed English model for better performance on complex language
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    # Fallback to small model if transformer model is not available
    nlp = spacy.load("en_core_web_sm")

def analyze_text(doc: Doc) -> Dict:
    """
    Extract features from a spaCy Doc for analysis or statistics.

    Args:
        doc (spacy.tokens.Doc): Parsed document.

    Returns:
        Dict: A dictionary of extracted features (e.g., entity counts, sentence lengths).
    """
    features = {
        "num_tokens": len(doc),
        "num_sentences": len(list(doc.sents)),
        "entities": [ent.label_ for ent in doc.ents],
        "pos_counts": doc.count_by(POS),
        "noun_chunks": [chunk.text for chunk in doc.noun_chunks]
    }
    return features

def process_texts(texts: Union[List[str], str]) -> List[Doc]:
    """
    Apply the spaCy pipeline to a list of texts.

    Args:
        texts (List[str] or str): List of input strings.

    Returns:
        List[Doc]: List of spaCy Doc objects.
    """
    if not isinstance(texts, list):
        raise TypeError("Input to process_texts must be a list of strings.")
    docs = []
    for text in texts:
        if not text:
            docs.append(Doc(nlp.vocab, words=[]))
        else:
            docs.append(nlp(text))
    return docs

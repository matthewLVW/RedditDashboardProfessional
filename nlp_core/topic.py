# nlp_core/topic.py

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Initialize embedding model
try:
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _embed_model = None

# Initialize BERTopic; dimensionality reduction and clustering under the hood
try:
    _topic_model = BERTopic(verbose=False, embedding_model=_embed_model)
except Exception:
    _topic_model = None

def get_topics(docs: List[str]) -> Tuple[List[int], List[float]]:
    """
    Fit BERTopic on a batch of documents and return topic assignments and probabilities.

    Args:
        docs (List[str]): List of document texts.

    Returns:
        Tuple[List[int], List[float]]: (topics, probabilities) for each document.
    """
    if not docs or not _topic_model:
        raise ValueError("No documents provided or BERTopic model unavailable.")
    topics, probs = _topic_model.fit_transform(docs)
    return topics, probs

def extract_keywords(topic_model= _topic_model, top_n: int = 5) -> List[List[str]]:
    """
    Extract keywords for each topic.

    Args:
        topic_model (BERTopic): Trained BERTopic model.
        top_n (int): Number of keywords per topic.

    Returns:
        List[List[str]]: Keywords for each topic.
    """
    if not topic_model:
        return []
    return [topic_model.get_topic(topic) for topic in range(len(topic_model.get_topic_info())-1)]

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from typing import List, Any, Optional, Tuple
from collections.abc import Mapping

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    _topic_model: BERTopic = BERTopic(verbose=False, embedding_model=embed_model)
except Exception as e:
    raise RuntimeError(f"BERTopic initialization failed: {e}") from e

def get_topics(docs: List[str]) -> Tuple[List[int], List[Any]]:
    if not docs:
        raise ValueError("No documents provided.")
    topics, probs = _topic_model.fit_transform(docs)
    topics_list = list(topics)
    if probs is None:
        probs_list = []
    else:
        try:
            probs_list = probs.tolist()  # numpy/pandas
        except Exception:
            try:
                probs_list = list(probs)  # generic iterable
            except Exception:
                probs_list = []
    return topics_list, probs_list

def extract_keywords(
    topic_model: Optional[BERTopic] = _topic_model,
    top_n: int = 5
) -> List[List[str]]:
    if topic_model is None:
        return []
    info = topic_model.get_topic_info()
    num = len(info) - 1 if len(info) > 0 else 0
    results: List[List[str]] = []
    for t in range(num):
        data = topic_model.get_topic(t)
        # Robust skip for weird outputs
        if not data or data is True:
            continue
        # Always turn mapping into a list, else coerce to list for slicing
        if isinstance(data, Mapping):
            items = list(data.values())
        else:
            items = list(data)
        sliced = items[:top_n]
        keywords = [word for word, _ in sliced]
        results.append(keywords)
    return results

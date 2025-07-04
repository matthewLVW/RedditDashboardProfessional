# nlp_core/embeddings.py

import numpy as np
from typing import List
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
import faiss

class EmbeddingStore:
    """
    Stores embeddings in a FAISS index for nearest-neighbor search.
    """
    def __init__(self, model_name: str = "hkunlp/instructor-xl"):
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
        else:
            self.model = None
        self.index = None
        self.dimension = None

    def build_index(self, texts: List[str]):
        """
        Build FAISS index from a list of texts.

        Args:
            texts (List[str]): Texts to index.
        """
        if not self.model:
            raise RuntimeError("No embedding model available.")
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype='float32')
        self.dimension = embeddings.shape[1]
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Search the index for nearest neighbors to the query text.

        Args:
            query (str): Input query string.
            k (int): Number of neighbors to return.

        Returns:
            List[int]: Indices of nearest neighbors.
        """
        if self.index is None or self.model is None:
            raise RuntimeError("Index or model not initialized.")
        query_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')
        D, I = self.index.search(query_vec, k)
        return I[0].tolist()

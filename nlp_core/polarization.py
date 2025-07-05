# nlp_core/polarization.py

import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import List

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p (np.ndarray): First probability distribution.
        q (np.ndarray): Second probability distribution.

    Returns:
        float: JS divergence.
    """
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must be non-negative.")
    # Normalize distributions
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    # Jensen-Shannon (scipy returns sqrt(JS), so square it)
    return float(jensenshannon(p_norm, q_norm) ** 2)

def moral_foundation_vectors(comments: List[str]) -> np.ndarray:
    """
    Placeholder for mapping comments to Moral Foundations embeddings (requires a model).
    Returns random vectors for demonstration.
    """
    import numpy as np
    # Assuming 5-dimensional Moral Foundations space
    return np.random.rand(len(comments), 5)

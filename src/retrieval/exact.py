"""Exact cosine-similarity retrieval."""

import numpy as np


def search(gallery, query, top_k):
    """Perform exhaustive cosine-similarity search."""
    similarities = gallery @ query
    k = min(top_k, len(similarities))

    if k == len(similarities):
        indices = np.argsort(-similarities)
    else:
        indices = np.argpartition(-similarities, k - 1)[:k]
        indices = indices[np.argsort(-similarities[indices])]

    return indices, similarities[indices], len(gallery)
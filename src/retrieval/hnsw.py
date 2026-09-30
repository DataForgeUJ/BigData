"""HNSW approximate nearest-neighbor retrieval."""

import hnswlib
import numpy as np


def build_index(embeddings, output_path, space, M, ef_construction, ef_search):
    """Build and save an HNSW index."""
    index = hnswlib.Index(space=space, dim=embeddings.shape[1])

    index.init_index(
        max_elements=len(embeddings),
        ef_construction=ef_construction,
        M=M,
    )

    index.add_items(embeddings, np.arange(len(embeddings)))
    index.set_ef(ef_search)
    index.save_index(str(output_path))

    return index


def search(index, query, top_k):
    """Search an HNSW index."""
    k = min(top_k, index.get_current_count())

    labels, distances = index.knn_query(query.reshape(1, -1), k=k)

    indices = labels[0].astype(int)
    similarities = 1.0 - distances[0]

    return indices, similarities, np.nan


def load_index(path, dimension, space):
    """Load a saved HNSW index."""
    index = hnswlib.Index(space=space, dim=dimension)
    index.load_index(str(path))
    return index
"""Random Hyperplane LSH retrieval."""

import numpy as np


class RandomHyperplaneLSH:
    """Random Hyperplane Locality-Sensitive Hashing index."""

    def __init__(self, dim, num_bits, num_tables, seed=42):
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.seed = seed
        self.tables = []
        self.planes = []

        rng = np.random.default_rng(seed)

        for _ in range(num_tables):
            plane = np.zeros((num_bits, dim), dtype=np.float32)

            for bit in range(num_bits):
                dims = rng.choice(dim, size=min(32, dim), replace=False)
                values = rng.normal(size=len(dims)).astype(np.float32)
                plane[bit, dims] = values

            self.planes.append(plane)
            self.tables.append({})

    def _hash_vectors(self, vectors, plane):
        projections = vectors @ plane.T
        bits = (projections >= 0).astype(np.uint32)
        codes = np.zeros(len(vectors), dtype=np.uint32)

        for bit in range(self.num_bits):
            codes |= bits[:, bit] << bit

        return codes

    def fit(self, embeddings):
        """Build LSH hash tables."""
        self.tables = [{} for _ in range(self.num_tables)]

        for table_id, plane in enumerate(self.planes):
            codes = self._hash_vectors(embeddings, plane)

            for index, code in enumerate(codes):
                code = int(code)
                self.tables[table_id].setdefault(code, []).append(index)

            print(f"    LSH table {table_id + 1}: {len(self.tables[table_id])} buckets")

    def query_candidates(self, query):
        """Return unique candidate gallery indices."""
        candidates = set()

        for table_id, plane in enumerate(self.planes):
            code = int(self._hash_vectors(query[None], plane)[0])
            candidates.update(self.tables[table_id].get(code, []))

        return np.array(sorted(candidates), dtype=np.int64)


def search(lsh, gallery, query, top_k):
    """Search LSH candidates using exact cosine similarity."""
    candidates = lsh.query_candidates(query)

    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), 0

    candidates = candidates[(candidates >= 0) & (candidates < len(gallery))]

    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), 0

    similarities = gallery[candidates] @ query
    k = min(top_k, len(candidates))
    order = np.argsort(-similarities)[:k]

    return candidates[order], similarities[order], len(candidates)
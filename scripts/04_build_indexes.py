"""Build Exact, LSH, and HNSW retrieval indexes."""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hnswlib
import numpy as np
import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path("configs/base.yaml")
INDEX_DIR = Path("artifacts/indexes")


# ============================================================
# Helpers
# ============================================================

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_embeddings(embedding_dir):
    embeddings = np.load(embedding_dir / "train_embeddings.npy")
    metadata = pd.read_csv(embedding_dir / "train_metadata.csv")

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding/metadata mismatch: "
            f"{len(embeddings)} embeddings vs {len(metadata)} metadata rows"
        )

    return embeddings.astype(np.float32), metadata


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def create_gallery_subset(embeddings, metadata, fraction, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(embeddings))[:int(len(embeddings) * fraction)]

    return embeddings[indices], metadata.iloc[indices].reset_index(drop=True), indices


def save_exact_gallery(gallery_dir, embeddings, metadata):
    exact_dir = gallery_dir / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)

    np.save(exact_dir / "gallery_embeddings.npy", embeddings)
    metadata.to_csv(exact_dir / "gallery_metadata.csv", index=False)


# ============================================================
# Random Hyperplane LSH
# ============================================================

class RandomHyperplaneLSH:
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
        self.tables = [{} for _ in range(self.num_tables)]

        for table_id, plane in enumerate(self.planes):
            codes = self._hash_vectors(embeddings, plane)

            for index, code in enumerate(codes):
                code = int(code)
                self.tables[table_id].setdefault(code, []).append(index)

            print(
                f"    LSH table {table_id + 1}: "
                f"{len(self.tables[table_id])} buckets"
            )

    def query_candidates(self, query):
        candidates = set()

        for table_id, plane in enumerate(self.planes):
            code = int(self._hash_vectors(query[None], plane)[0])
            candidates.update(self.tables[table_id].get(code, []))

        return np.array(sorted(candidates), dtype=np.int64)


# ============================================================
# HNSW
# ============================================================

def build_hnsw_index(
    embeddings,
    output_path,
    space,
    M,
    ef_construction,
    ef_search,
):
    index = hnswlib.Index(space=space, dim=embeddings.shape[1])

    index.init_index(
        max_elements=len(embeddings),
        ef_construction=ef_construction,
        M=M,
    )

    index.add_items(
        embeddings,
        np.arange(len(embeddings)),
    )

    index.set_ef(ef_search)
    index.save_index(str(output_path))

    return index


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()

    seed = config["seed"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    retrieval_cfg = config["retrieval"]

    embedding_mode = data_cfg.get("embedding_mode", "finetuned")
    embedding_dir = Path(data_cfg["embeddings_dir"]) / embedding_mode
    index_root = INDEX_DIR / embedding_mode

    fractions = retrieval_cfg["gallery_fractions"]

    lsh_cfg = retrieval_cfg["lsh"]
    hnsw_cfg = retrieval_cfg["hnsw"]

    print("=" * 60)
    print("STEP 04: BUILD RETRIEVAL INDEXES")
    print("=" * 60)
    print(f"Embedding mode : {embedding_mode}")
    print(f"Embedding dim  : {model_cfg['embedding_dim']}")
    print(f"Seed            : {seed}")
    print(f"Gallery sizes   : {fractions}")
    print()

    embeddings, metadata = load_embeddings(embedding_dir)

    if embeddings.shape[1] != model_cfg["embedding_dim"]:
        raise ValueError(
            f"Expected embedding dimension "
            f"{model_cfg['embedding_dim']}, "
            f"got {embeddings.shape[1]}"
        )

    if model_cfg.get("normalize_embeddings", True):
        embeddings = normalize_embeddings(embeddings)

    print(f"Loaded {len(embeddings):,} training embeddings")
    print(f"Dimension: {embeddings.shape[1]}")
    print()

    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(len(embeddings))

    for fraction in fractions:
        gallery_size = int(len(embeddings) * fraction)

        selected_indices = all_indices[:gallery_size]

        gallery_embeddings = embeddings[selected_indices]
        gallery_metadata = metadata.iloc[selected_indices].reset_index(drop=True)

        gallery_dir = index_root / f"gallery_{int(fraction * 100)}"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 60)
        print(
            f"Gallery {fraction:.0%}: "
            f"{gallery_size:,} / {len(embeddings):,}"
        )

        # ----------------------------------------------------
        # Save selected indices
        # ----------------------------------------------------

        np.save(
            gallery_dir / "selected_indices.npy",
            selected_indices,
        )

        # ----------------------------------------------------
        # Exact Search
        # ----------------------------------------------------

        save_exact_gallery(
            gallery_dir,
            gallery_embeddings,
            gallery_metadata,
        )

        # ----------------------------------------------------
        # LSH
        # ----------------------------------------------------

        lsh_dir = gallery_dir / "lsh"
        lsh_dir.mkdir(parents=True, exist_ok=True)

        print("  Building LSH...")

        start = time.perf_counter()

        lsh = RandomHyperplaneLSH(
            dim=gallery_embeddings.shape[1],
            num_bits=lsh_cfg["num_bits"],
            num_tables=lsh_cfg["num_tables"],
            seed=seed,
        )

        lsh.fit(gallery_embeddings)

        with open(lsh_dir / "lsh_index.pkl", "wb") as f:
            pickle.dump(lsh, f)

        lsh_build_time = time.perf_counter() - start

        # ----------------------------------------------------
        # HNSW
        # ----------------------------------------------------

        hnsw_dir = gallery_dir / "hnsw"
        hnsw_dir.mkdir(parents=True, exist_ok=True)

        print("  Building HNSW...")

        start = time.perf_counter()

        build_hnsw_index(
            gallery_embeddings,
            hnsw_dir / "hnsw_index.bin",
            space=hnsw_cfg["space"],
            M=hnsw_cfg["M"],
            ef_construction=hnsw_cfg["ef_construction"],
            ef_search=hnsw_cfg["ef_search"],
        )

        hnsw_build_time = time.perf_counter() - start

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        configuration = {
            "embedding_mode": embedding_mode,
            "embedding_dimension": model_cfg["embedding_dim"],
            "normalize_embeddings": model_cfg.get(
                "normalize_embeddings", True
            ),
            "gallery_fraction": fraction,
            "gallery_size": gallery_size,
            "seed": seed,
            "lsh": {
                "num_bits": lsh_cfg["num_bits"],
                "num_tables": lsh_cfg["num_tables"],
                "build_time_s": lsh_build_time,
            },
            "hnsw": {
                "space": hnsw_cfg["space"],
                "M": hnsw_cfg["M"],
                "ef_construction": hnsw_cfg["ef_construction"],
                "ef_search": hnsw_cfg["ef_search"],
                "build_time_s": hnsw_build_time,
            },
        }

        with open(
            gallery_dir / "configuration.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(configuration, f, indent=2)

        print(f"  LSH build time : {lsh_build_time:.3f}s")
        print(f"  HNSW build time: {hnsw_build_time:.3f}s")

    print()
    print("=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Indexes saved to: {index_root}")


if __name__ == "__main__":
    main()
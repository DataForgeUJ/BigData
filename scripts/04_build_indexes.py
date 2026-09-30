"""Build Exact, LSH, and HNSW retrieval indexes."""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.retrieval.hnsw import build_index as build_hnsw_index
from src.retrieval.lsh import RandomHyperplaneLSH


CONFIG_PATH = Path("configs/base.yaml")
INDEX_ROOT = Path("artifacts/indexes")

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = config.get("seed", 42)
DATA_CFG = config["data"]
MODEL_CFG = config["model"]
RETRIEVAL_CFG = config["retrieval"]

EMBEDDING_MODE = DATA_CFG.get("embedding_mode", "finetuned")
EMBEDDING_ROOT = Path(DATA_CFG["embeddings_dir"]) / EMBEDDING_MODE


def normalize_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def load_embeddings(name):
    path = EMBEDDING_ROOT / f"{name}_embeddings.npy"

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    return normalize_embeddings(np.load(path))


def load_metadata(name):
    path = EMBEDDING_ROOT / f"{name}_metadata.csv"

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def create_gallery_subsets(embeddings, metadata, fractions):
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(embeddings))
    subsets = {}

    for fraction in fractions:
        size = int(len(embeddings) * fraction)
        indices = permutation[:size]
        subsets[fraction] = indices

    return subsets


def build_lsh(gallery, output_path):
    lsh_cfg = RETRIEVAL_CFG["lsh"]

    lsh = RandomHyperplaneLSH(
        dim=gallery.shape[1],
        num_bits=int(lsh_cfg["num_bits"]),
        num_tables=int(lsh_cfg["num_tables"]),
        seed=SEED,
    )

    start = time.perf_counter()
    lsh.fit(gallery)
    build_time = time.perf_counter() - start

    with open(output_path, "wb") as f:
        pickle.dump(lsh, f)

    return build_time


def build_hnsw(gallery, output_path):
    hnsw_cfg = RETRIEVAL_CFG["hnsw"]

    start = time.perf_counter()

    build_hnsw_index(
        gallery,
        output_path,
        hnsw_cfg["space"],
        int(hnsw_cfg["M"]),
        int(hnsw_cfg["ef_construction"]),
        int(hnsw_cfg["ef_search"]),
    )

    return time.perf_counter() - start


def main():
    print("=" * 70)
    print("STEP 04 - BUILD RETRIEVAL INDEXES")
    print("=" * 70)
    print(f"Embedding mode : {EMBEDDING_MODE}")
    print(f"Embedding dim  : {MODEL_CFG['embedding_dim']}")

    train_embeddings = load_embeddings("train")
    train_metadata = load_metadata("train")

    print(f"Training gallery: {len(train_embeddings):,} images")
    print(f"Identities      : {train_metadata['identity'].nunique():,}")

    fractions = RETRIEVAL_CFG.get(
        "gallery_fractions",
        [0.25, 0.50, 0.75, 1.00],
    )

    subsets = create_gallery_subsets(
        train_embeddings,
        train_metadata,
        fractions,
    )

    mode_root = INDEX_ROOT / EMBEDDING_MODE
    mode_root.mkdir(parents=True, exist_ok=True)

    for fraction, indices in subsets.items():
        gallery_name = f"gallery_{int(fraction * 100)}"
        gallery_dir = mode_root / gallery_name
        exact_dir = gallery_dir / "exact"
        lsh_dir = gallery_dir / "lsh"
        hnsw_dir = gallery_dir / "hnsw"

        for directory in [exact_dir, lsh_dir, hnsw_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        gallery = train_embeddings[indices]
        gallery_metadata = train_metadata.iloc[indices].reset_index(drop=True)

        print("\n" + "=" * 70)
        print(f"Gallery: {fraction:.0%} ({len(gallery):,} images)")
        print("=" * 70)

        np.save(gallery_dir / "selected_indices.npy", indices)
        np.save(exact_dir / "gallery_embeddings.npy", gallery)
        gallery_metadata.to_csv(
            exact_dir / "gallery_metadata.csv",
            index=False,
        )

        print("\nBuilding LSH...")
        lsh_time = build_lsh(
            gallery,
            lsh_dir / "lsh_index.pkl",
        )
        print(f"  Build time: {lsh_time:.4f} s")

        print("\nBuilding HNSW...")
        hnsw_time = build_hnsw(
            gallery,
            hnsw_dir / "hnsw_index.bin",
        )
        print(f"  Build time: {hnsw_time:.4f} s")

        configuration = {
            "embedding_mode": EMBEDDING_MODE,
            "embedding_dim": int(MODEL_CFG["embedding_dim"]),
            "normalize_embeddings": bool(
                MODEL_CFG.get("normalize_embeddings", True)
            ),
            "gallery_fraction": fraction,
            "gallery_size": len(gallery),
            "gallery_identities": int(
                gallery_metadata["identity"].nunique()
            ),
            "lsh": {
                "num_bits": int(RETRIEVAL_CFG["lsh"]["num_bits"]),
                "num_tables": int(RETRIEVAL_CFG["lsh"]["num_tables"]),
                "build_time_s": lsh_time,
            },
            "hnsw": {
                "space": RETRIEVAL_CFG["hnsw"]["space"],
                "M": int(RETRIEVAL_CFG["hnsw"]["M"]),
                "ef_construction": int(
                    RETRIEVAL_CFG["hnsw"]["ef_construction"]
                ),
                "ef_search": int(
                    RETRIEVAL_CFG["hnsw"]["ef_search"]
                ),
                "build_time_s": hnsw_time,
            },
        }

        (gallery_dir / "configuration.json").write_text(
            json.dumps(configuration, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nIndex building interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(e)
        sys.exit(1)
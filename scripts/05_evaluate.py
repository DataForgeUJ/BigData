"""Evaluate Exact, LSH, and HNSW retrieval with open-set recognition."""

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
INDEX_ROOT = Path("artifacts/indexes")
RESULT_ROOT = Path("artifacts/results")


# ============================================================
# Configuration / Data
# ============================================================

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_embeddings(embedding_dir, filename):
    return np.load(embedding_dir / filename).astype(np.float32)


def load_metadata(embedding_dir, filename):
    return pd.read_csv(embedding_dir / filename)


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


# ============================================================
# LSH
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
                dims = rng.choice(
                    dim,
                    size=min(32, dim),
                    replace=False,
                )
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

    def query_candidates(self, query):
        candidates = set()

        for table_id, plane in enumerate(self.planes):
            code = int(
                self._hash_vectors(
                    query[None],
                    plane,
                )[0]
            )
            candidates.update(
                self.tables[table_id].get(code, [])
            )

        return np.array(
            sorted(candidates),
            dtype=np.int64,
        )


# ============================================================
# Search
# ============================================================

def exact_search(
    query,
    gallery_embeddings,
    top_k,
    exclude_index=None,
):
    scores = gallery_embeddings @ query

    if exclude_index is not None:
        scores[exclude_index] = -np.inf

    k = min(top_k, len(scores))
    indices = np.argpartition(-scores, k - 1)[:k]
    indices = indices[np.argsort(-scores[indices])]

    return indices, scores[indices]


def lsh_search(
    query,
    gallery_embeddings,
    lsh,
    top_k,
    exclude_index=None,
):
    candidates = lsh.query_candidates(query)

    if exclude_index is not None:
        candidates = candidates[candidates != exclude_index]

    if len(candidates) == 0:
        return np.array([], dtype=np.int64), np.array([])

    scores = gallery_embeddings[candidates] @ query

    k = min(top_k, len(candidates))
    order = np.argsort(-scores)[:k]

    return candidates[order], scores[order]


def hnsw_search(
    query,
    index,
    top_k,
    exclude_index=None,
):
    k = min(
        top_k + (1 if exclude_index is not None else 0),
        index.get_current_count(),
    )

    labels, distances = index.knn_query(
        query[None],
        k=k,
    )

    labels = labels[0]
    distances = distances[0]

    if exclude_index is not None:
        keep = labels != exclude_index
        labels = labels[keep]
        distances = distances[keep]

    scores = 1.0 - distances

    return labels[:top_k], scores[:top_k]


def search_one(
    method,
    query,
    gallery_embeddings,
    top_k,
    lsh=None,
    hnsw=None,
    exclude_index=None,
):
    if method == "Exact":
        return exact_search(
            query,
            gallery_embeddings,
            top_k,
            exclude_index,
        )

    if method == "LSH":
        return lsh_search(
            query,
            gallery_embeddings,
            lsh,
            top_k,
            exclude_index,
        )

    if method == "HNSW":
        return hnsw_search(
            query,
            hnsw,
            top_k,
            exclude_index,
        )

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Open-Set Calibration
# ============================================================

def calibrate_threshold(
    method,
    known_queries,
    unknown_queries,
    gallery_embeddings,
    top_k,
    lsh=None,
    hnsw=None,
):
    scores = []

    # Known queries
    for i, query in enumerate(known_queries):
        _, result_scores = search_one(
            method,
            query,
            gallery_embeddings,
            top_k=1,
            lsh=lsh,
            hnsw=hnsw,
            exclude_index=i,
        )

        if len(result_scores):
            scores.append(
                (float(result_scores[0]), 1)
            )

    # Unknown queries
    for query in unknown_queries:
        _, result_scores = search_one(
            method,
            query,
            gallery_embeddings,
            top_k=1,
            lsh=lsh,
            hnsw=hnsw,
        )

        if len(result_scores):
            scores.append(
                (float(result_scores[0]), 0)
            )

    if not scores:
        return 0.0

    values = np.array(
        [x[0] for x in scores],
        dtype=float,
    )
    labels = np.array(
        [x[1] for x in scores],
        dtype=int,
    )

    thresholds = np.unique(values)

    best_threshold = thresholds[0]
    best_j = -np.inf

    for threshold in thresholds:
        predicted = values >= threshold

        tp = np.sum((predicted == 1) & (labels == 1))
        fn = np.sum((predicted == 0) & (labels == 1))
        fp = np.sum((predicted == 1) & (labels == 0))
        tn = np.sum((predicted == 0) & (labels == 0))

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        j = tpr - fpr

        if j > best_j:
            best_j = j
            best_threshold = threshold

    return float(best_threshold)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    predictions,
    true_identities,
    known_flags,
):
    predictions = np.asarray(predictions)
    true_identities = np.asarray(true_identities)
    known_flags = np.asarray(known_flags)

    known_mask = known_flags == 1
    unknown_mask = known_flags == 0

    # BAKS: correct identification among known queries
    if known_mask.any():
        baks = np.mean(
            predictions[known_mask]
            == true_identities[known_mask]
        )
    else:
        baks = np.nan

    # BAUS: correctly rejected unknown queries
    if unknown_mask.any():
        baus = np.mean(
            predictions[unknown_mask] == -1
        )
    else:
        baus = np.nan

    if np.isfinite(baks) and np.isfinite(baus):
        geomean = np.sqrt(baks * baus)
    else:
        geomean = np.nan

    return baks, baus, geomean


# ============================================================
# Utilities
# ============================================================

def get_directory_size_mb(path):
    total = 0

    if path.exists():
        for file in path.rglob("*"):
            if file.is_file():
                total += file.stat().st_size

    return total / (1024 ** 2)


def load_lsh(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_hnsw(path, dim, space):
    index = hnswlib.Index(
        space=space,
        dim=dim,
    )
    index.load_index(str(path))
    return index


# ============================================================
# Main Evaluation
# ============================================================

def main():
    config = load_config()

    seed = config["seed"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    retrieval_cfg = config["retrieval"]
    open_set_cfg = config["open_set"]
    evaluation_cfg = config["evaluation"]

    embedding_mode = data_cfg.get(
        "embedding_mode",
        "finetuned",
    )

    embedding_dir = (
        Path(data_cfg["embeddings_dir"])
        / embedding_mode
    )

    index_root = INDEX_ROOT / embedding_mode
    result_root = RESULT_ROOT / embedding_mode
    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_k = retrieval_cfg["top_k"]
    gallery_fractions = retrieval_cfg["gallery_fractions"]

    lsh_cfg = retrieval_cfg["lsh"]
    hnsw_cfg = retrieval_cfg["hnsw"]

    # --------------------------------------------------------
    # Load query data
    # --------------------------------------------------------

    train_embeddings = load_embeddings(
        embedding_dir,
        "train_embeddings.npy",
    )

    train_metadata = load_metadata(
        embedding_dir,
        "train_metadata.csv",
    )

    validation_embeddings = load_embeddings(
        embedding_dir,
        "validation_embeddings.npy",
    )

    validation_metadata = load_metadata(
        embedding_dir,
        "validation_metadata.csv",
    )

    if model_cfg.get("normalize_embeddings", True):
        train_embeddings = normalize_embeddings(
            train_embeddings
        )
        validation_embeddings = normalize_embeddings(
            validation_embeddings
        )

    print("=" * 60)
    print("STEP 05: RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Embedding mode : {embedding_mode}")
    print(f"Embedding dim  : {model_cfg['embedding_dim']}")
    print(f"Top-k          : {top_k}")
    print(f"Seed           : {seed}")
    print()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    methods = [
        "Exact",
        "LSH",
        "HNSW",
    ]

    # --------------------------------------------------------
    # Gallery evaluation
    # --------------------------------------------------------

    for fraction in gallery_fractions:
        gallery_name = f"gallery_{int(fraction * 100)}"
        gallery_dir = index_root / gallery_name

        print("-" * 60)
        print(f"Gallery: {fraction:.0%}")

        exact_dir = gallery_dir / "exact"
        lsh_dir = gallery_dir / "lsh"
        hnsw_dir = gallery_dir / "hnsw"

        gallery_embeddings = np.load(
            exact_dir / "gallery_embeddings.npy"
        ).astype(np.float32)

        gallery_metadata = pd.read_csv(
            exact_dir / "gallery_metadata.csv"
        )

        # ----------------------------------------------------
        # Load indexes
        # ----------------------------------------------------

        lsh = load_lsh(
            lsh_dir / "lsh_index.pkl"
        )

        hnsw = load_hnsw(
            hnsw_dir / "hnsw_index.bin",
            dim=model_cfg["embedding_dim"],
            space=hnsw_cfg["space"],
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        calibration_size = min(
            1000,
            len(train_embeddings),
            len(validation_embeddings),
        )

        known_queries = train_embeddings[
            :calibration_size
        ]

        unknown_queries = validation_embeddings[
            :calibration_size
        ]

        # For 100% gallery, leave-one-out is possible.
        # For smaller galleries, only use known queries whose
        # identities occur in the gallery.

        gallery_ids = set(
            gallery_metadata["identity"]
        )

        known_indices = [
            i
            for i, identity in enumerate(
                train_metadata["identity"]
            )
            if identity in gallery_ids
        ]

        known_indices = known_indices[
            :calibration_size
        ]

        known_queries = train_embeddings[
            known_indices
        ]

        # ----------------------------------------------------
        # Evaluate each method
        # ----------------------------------------------------

        for method in methods:
            print(f"  Evaluating {method}...")

            # Threshold
            if open_set_cfg["threshold"] is not None:
                threshold = float(
                    open_set_cfg["threshold"]
                )
            else:
                threshold = calibrate_threshold(
                    method,
                    known_queries,
                    unknown_queries,
                    gallery_embeddings,
                    top_k=1,
                    lsh=lsh,
                    hnsw=hnsw,
                )

            # ------------------------------------------------
            # Test queries
            # ------------------------------------------------

            test_limit = min(
                2000,
                len(validation_embeddings),
            )

            query_embeddings = validation_embeddings[
                :test_limit
            ]

            query_metadata = validation_metadata.iloc[
                :test_limit
            ].reset_index(drop=True)

            predictions = []
            true_ids = []
            known_flags = []
            latencies = []
            comparisons = []

            for query, (_, row) in zip(
                query_embeddings,
                query_metadata.iterrows(),
            ):
                start = time.perf_counter()

                indices, scores = search_one(
                    method,
                    query,
                    gallery_embeddings,
                    top_k=top_k,
                    lsh=lsh,
                    hnsw=hnsw,
                )

                latency = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(latency)

                if len(indices):
                    best_index = int(indices[0])
                    best_score = float(scores[0])

                    if best_score >= threshold:
                        predicted_identity = (
                            gallery_metadata.iloc[
                                best_index
                            ]["identity"]
                        )
                    else:
                        predicted_identity = -1
                else:
                    best_score = -np.inf
                    predicted_identity = -1

                predictions.append(
                    predicted_identity
                )

                true_identity = row["identity"]
                true_ids.append(true_identity)

                known_flags.append(
                    int(
                        true_identity in gallery_ids
                    )
                )

                comparisons.append(
                    len(indices)
                )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            predictions_array = np.array(
                predictions,
                dtype=object,
            )

            true_ids_array = np.array(
                true_ids,
                dtype=object,
            )

            known_flags_array = np.array(
                known_flags,
                dtype=int,
            )

            baks, baus, geomean = calculate_metrics(
                predictions_array,
                true_ids_array,
                known_flags_array,
            )

            latency_array = np.array(
                latencies
            )

            avg_latency = float(
                np.mean(latency_array)
            )

            p95_latency = float(
                np.percentile(
                    latency_array,
                    95,
                )
            )

            qps = (
                1000.0 / avg_latency
                if avg_latency > 0
                else 0.0
            )

            avg_comparisons = float(
                np.mean(comparisons)
            )

            index_size = get_directory_size_mb(
                gallery_dir
            )

            # ------------------------------------------------
            # Build times
            # ------------------------------------------------

            with open(
                gallery_dir / "configuration.json",
                "r",
                encoding="utf-8",
            ) as f:
                index_config = json.load(f)

            lsh_build_time = index_config.get(
                "lsh",
                {},
            ).get(
                "build_time_s",
                np.nan,
            )

            hnsw_build_time = index_config.get(
                "hnsw",
                {},
            ).get(
                "build_time_s",
                np.nan,
            )

            if method == "LSH":
                build_time = lsh_build_time
            elif method == "HNSW":
                build_time = hnsw_build_time
            else:
                build_time = 0.0

            results.append(
                {
                    "gallery_fraction": fraction,
                    "gallery_size": len(
                        gallery_embeddings
                    ),
                    "method": method,
                    "top_k": top_k,
                    "threshold": threshold,
                    "baks": baks,
                    "baus": baus,
                    "geomean": geomean,
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "qps": qps,
                    "comparisons": avg_comparisons,
                    "index_build_time_s": build_time,
                    "index_size_mb": index_size,
                }
            )

            print(
                f"    threshold={threshold:.4f} | "
                f"BAKS={baks:.4f} | "
                f"BAUS={baus:.4f} | "
                f"latency={avg_latency:.3f} ms | "
                f"QPS={qps:.2f}"
            )

    # ========================================================
    # Save Results
    # ========================================================

    results_df = pd.DataFrame(results)

    output_csv = (
        result_root / "finetuned_results.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    evaluation_configuration = {
        "seed": seed,
        "embedding_mode": embedding_mode,
        "embedding_dimension": model_cfg[
            "embedding_dim"
        ],
        "normalize_embeddings": model_cfg.get(
            "normalize_embeddings",
            True,
        ),
        "top_k": top_k,
        "gallery_fractions": gallery_fractions,
        "open_set": open_set_cfg,
        "evaluation_metrics": evaluation_cfg[
            "metrics"
        ],
        "lsh": lsh_cfg,
        "hnsw": hnsw_cfg,
    }

    with open(
        result_root / "evaluation_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evaluation_configuration,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
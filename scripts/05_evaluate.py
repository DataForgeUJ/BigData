"""Evaluate Exact, LSH, and HNSW retrieval performance."""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import (
    calculate_auroc,
    calculate_baks,
    calculate_baus,
    calculate_geomean,
    calculate_qps,
)
from src.retrieval.exact import search as exact_search
from src.retrieval.hnsw import load_index as load_hnsw
from src.retrieval.hnsw import search as hnsw_search
from src.retrieval.lsh import search as lsh_search


CONFIG_PATH = Path("configs/base.yaml")
INDEX_ROOT = Path("artifacts/indexes")
RESULT_ROOT = Path("artifacts/results")

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = config.get("seed", 42)
DATA_CFG = config["data"]
MODEL_CFG = config["model"]
RETRIEVAL_CFG = config["retrieval"]

EMBEDDING_MODE = DATA_CFG.get("embedding_mode", "finetuned")
EMBEDDING_ROOT = Path(DATA_CFG["embeddings_dir"]) / EMBEDDING_MODE
RESULT_DIR = RESULT_ROOT / EMBEDDING_MODE
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = int(RETRIEVAL_CFG.get("top_k", 5))
GALLERY_FRACTIONS = RETRIEVAL_CFG.get(
    "gallery_fractions", [0.25, 0.50, 0.75, 1.00]
)
OPEN_SET_THRESHOLD = config["open_set"].get("threshold")
CALIBRATION_LIMIT = 1000


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


def get_directory_size_mb(path):
    if not path.exists():
        return np.nan
    return sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    ) / (1024 * 1024)


def load_lsh(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def search(method, query, top_k, gallery, lsh=None, hnsw=None):
    if method == "Exact":
        return exact_search(gallery, query, top_k)
    if method == "LSH":
        return lsh_search(lsh, gallery, query, top_k)
    if method == "HNSW":
        return hnsw_search(hnsw, query, top_k)
    raise ValueError(f"Unknown method: {method}")


def calibrate_threshold(
    method, gallery, gallery_metadata, validation_embeddings, lsh=None, hnsw=None
):
    gallery_ids = gallery_metadata["identity"].to_numpy()
    counts = pd.Series(gallery_ids).value_counts()
    valid_indices = [
        i for i, identity in enumerate(gallery_ids) if counts[identity] >= 2
    ]

    rng = np.random.default_rng(SEED)

    if len(valid_indices) > CALIBRATION_LIMIT:
        valid_indices = rng.choice(
            valid_indices, size=CALIBRATION_LIMIT, replace=False
        )

    print(f"  Known calibration candidates : {len(valid_indices)}")
    known_scores = []

    for gallery_index in valid_indices:
        query = gallery[gallery_index]
        true_identity = gallery_ids[gallery_index]

        indices, similarities, _ = search(
            method, query, TOP_K + 1, gallery, lsh=lsh, hnsw=hnsw
        )

        keep = indices != gallery_index
        indices, similarities = indices[keep], similarities[keep]

        if len(indices):
            best = int(np.argmax(similarities))
            if gallery_ids[indices[best]] == true_identity:
                known_scores.append(float(similarities[best]))

    unknown_limit = min(CALIBRATION_LIMIT, len(validation_embeddings))
    validation_indices = rng.choice(
        len(validation_embeddings),
        size=unknown_limit,
        replace=False,
    )

    print(f"  Unknown calibration candidates: {len(validation_indices)}")
    unknown_scores = []

    for validation_index in validation_indices:
        _, similarities, _ = search(
            method,
            validation_embeddings[validation_index],
            1,
            gallery,
            lsh=lsh,
            hnsw=hnsw,
        )
        if len(similarities):
            unknown_scores.append(float(similarities[0]))

    if not known_scores:
        raise RuntimeError(f"{method}: no known calibration scores were produced.")
    if not unknown_scores:
        raise RuntimeError(f"{method}: no unknown calibration scores were produced.")

    known_scores = np.asarray(known_scores, dtype=np.float32)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float32)

    scores = np.concatenate([known_scores, unknown_scores])
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))

    best_threshold, best_geomean = None, -1.0

    for threshold in thresholds:
        known_acceptance = np.mean(known_scores >= threshold)
        unknown_rejection = np.mean(unknown_scores < threshold)
        geomean = np.sqrt(known_acceptance * unknown_rejection)

        if geomean > best_geomean:
            best_threshold = float(threshold)
            best_geomean = geomean

    calibration = {
        "threshold": best_threshold,
        "known_count": len(known_scores),
        "unknown_count": len(unknown_scores),
        "known_mean": float(np.mean(known_scores)),
        "known_median": float(np.median(known_scores)),
        "unknown_mean": float(np.mean(unknown_scores)),
        "unknown_median": float(np.median(unknown_scores)),
        "calibration_geomean": float(best_geomean),
    }

    print(f"  Threshold: {best_threshold:.6f}")
    print(f"  Calibration GeoMean: {best_geomean:.4f}")

    return calibration


def evaluate_method(
    method,
    gallery,
    gallery_metadata,
    test_embeddings,
    test_metadata,
    threshold,
    lsh=None,
    hnsw=None,
):
    gallery_ids = gallery_metadata["identity"].to_numpy()
    test_ids = test_metadata["identity"].to_numpy()
    gallery_identity_set = set(gallery_ids)

    top1_correct = top5_correct = recall_hits = 0
    known_count = unknown_count = known_accepted_correct = unknown_rejected = 0
    known_scores, unknown_scores, latencies, comparisons = [], [], [], []

    for query_index, query in enumerate(test_embeddings):
        true_identity = test_ids[query_index]
        is_known = true_identity in gallery_identity_set

        if is_known:
            known_count += 1
        else:
            unknown_count += 1

        start = time.perf_counter()
        indices, similarities, comparison_count = search(
            method, query, TOP_K, gallery, lsh=lsh, hnsw=hnsw
        )
        latencies.append((time.perf_counter() - start) * 1000.0)

        if not np.isnan(comparison_count):
            comparisons.append(comparison_count)

        if len(indices) == 0:
            if not is_known:
                unknown_scores.append(-1.0)
            continue

        predicted_ids = [gallery_ids[i] for i in indices]
        best_similarity = float(similarities[0])

        if is_known:
            if predicted_ids[0] == true_identity:
                top1_correct += 1
            if true_identity in predicted_ids:
                top5_correct += 1
                recall_hits += 1

        accepted = best_similarity >= threshold

        if is_known:
            if accepted and predicted_ids[0] == true_identity:
                known_accepted_correct += 1
            known_scores.append(best_similarity)
        else:
            if not accepted:
                unknown_rejected += 1
            unknown_scores.append(best_similarity)

    top1 = top1_correct / known_count if known_count else np.nan
    top5 = top5_correct / known_count if known_count else np.nan
    recall_at_k = recall_hits / known_count if known_count else np.nan

    baks = calculate_baks(known_accepted_correct, known_count)
    baus = calculate_baus(unknown_rejected, unknown_count)
    geomean = calculate_geomean(baks, baus)

    avg_latency = float(np.mean(latencies)) if latencies else np.nan
    p95_latency = float(np.percentile(latencies, 95)) if latencies else np.nan
    qps = calculate_qps(avg_latency)
    avg_comparisons = float(np.mean(comparisons)) if comparisons else np.nan

    return {
        "top1": top1,
        "top5": top5,
        "baks": baks,
        "baus": baus,
        "geomean": geomean,
        "auroc": calculate_auroc(known_scores, unknown_scores),
        "recall_at_k": recall_at_k,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "qps": qps,
        "comparisons": avg_comparisons,
        "known_queries": known_count,
        "unknown_queries": unknown_count,
    }


def main():
    print("=" * 70)
    print("STEP 05 - RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Embedding mode : {EMBEDDING_MODE}")
    print(f"Top-k          : {TOP_K}")
    print(f"Gallery sizes  : {GALLERY_FRACTIONS}\n")

    print("Loading embeddings...")
    train_embeddings = load_embeddings("train")
    validation_embeddings = load_embeddings("validation")
    test_embeddings = load_embeddings("test")

    train_metadata = load_metadata("train")
    validation_metadata = load_metadata("validation")
    test_metadata = load_metadata("test")

    print(
        f"Train      : {len(train_embeddings):,} images / "
        f"{train_metadata['identity'].nunique():,} identities"
    )
    print(
        f"Validation : {len(validation_embeddings):,} images / "
        f"{validation_metadata['identity'].nunique():,} identities"
    )
    print(
        f"Test       : {len(test_embeddings):,} images / "
        f"{test_metadata['identity'].nunique():,} identities\n"
    )

    all_results = []

    for fraction in GALLERY_FRACTIONS:
        gallery_name = f"gallery_{int(fraction * 100)}"
        gallery_dir = INDEX_ROOT / EMBEDDING_MODE / gallery_name
        exact_dir = gallery_dir / "exact"
        lsh_dir = gallery_dir / "lsh"
        hnsw_dir = gallery_dir / "hnsw"

        print("=" * 70)
        print(f"Gallery: {fraction:.0%}")
        print("=" * 70)

        gallery = normalize_embeddings(
            np.load(exact_dir / "gallery_embeddings.npy")
        )
        gallery_metadata = pd.read_csv(
            exact_dir / "gallery_metadata.csv"
        )
        gallery_size = len(gallery)

        print(f"Gallery size : {gallery_size:,}")
        print("Loading LSH...")
        lsh = load_lsh(lsh_dir / "lsh_index.pkl")

        print("Loading HNSW...")
        hnsw_cfg = RETRIEVAL_CFG["hnsw"]
        hnsw = load_hnsw(
            hnsw_dir / "hnsw_index.bin",
            int(MODEL_CFG["embedding_dim"]),
            hnsw_cfg["space"],
        )

        sizes = {
            "Exact": get_directory_size_mb(exact_dir),
            "LSH": get_directory_size_mb(lsh_dir),
            "HNSW": get_directory_size_mb(hnsw_dir),
        }

        configuration_path = gallery_dir / "configuration.json"

        if configuration_path.exists():
            configuration = json.loads(
                configuration_path.read_text(encoding="utf-8")
            )
            build_times = {
                "Exact": 0.0,
                "LSH": float(configuration["lsh"]["build_time_s"]),
                "HNSW": float(configuration["hnsw"]["build_time_s"]),
            }
        else:
            build_times = {"Exact": 0.0, "LSH": np.nan, "HNSW": np.nan}

        methods = [
            ("Exact", None, None),
            ("LSH", lsh, None),
            ("HNSW", None, hnsw),
        ]

        for method, lsh_object, hnsw_object in methods:
            print(f"\nEvaluating {method}...")

            if OPEN_SET_THRESHOLD is not None:
                threshold = float(OPEN_SET_THRESHOLD)
                calibration = {
                    "threshold": threshold,
                    "known_count": np.nan,
                    "unknown_count": np.nan,
                    "calibration_geomean": np.nan,
                }
            else:
                calibration = calibrate_threshold(
                    method,
                    gallery,
                    gallery_metadata,
                    validation_embeddings,
                    lsh=lsh_object,
                    hnsw=hnsw_object,
                )
                threshold = calibration["threshold"]

            print(f"  Evaluating {len(test_embeddings):,} test queries...")

            metrics = evaluate_method(
                method,
                gallery,
                gallery_metadata,
                test_embeddings,
                test_metadata,
                threshold,
                lsh=lsh_object,
                hnsw=hnsw_object,
            )

            all_results.append({
                "gallery_fraction": fraction,
                "gallery_size": gallery_size,
                "method": method,
                "top_k": TOP_K,
                "threshold": threshold,
                **metrics,
                "index_build_time_s": build_times[method],
                "index_size_mb": sizes[method],
                "calibration_known": calibration["known_count"],
                "calibration_unknown": calibration["unknown_count"],
                "calibration_geomean": calibration["calibration_geomean"],
            })

            print(f"  Threshold : {threshold:.6f}")
            print(f"  Top-1     : {metrics['top1']:.4f}")
            print(f"  Top-5     : {metrics['top5']:.4f}")
            print(f"  BAKS      : {metrics['baks']:.4f}")
            print(f"  BAUS      : {metrics['baus']:.4f}")
            print(f"  GeoMean   : {metrics['geomean']:.4f}")
            print(f"  AUROC     : {metrics['auroc']:.4f}")
            print(f"  Recall@{TOP_K}: {metrics['recall_at_k']:.4f}")
            print(f"  Avg latency: {metrics['avg_latency_ms']:.4f} ms")
            print(f"  P95 latency: {metrics['p95_latency_ms']:.4f} ms")
            print(f"  QPS       : {metrics['qps']:.2f}")
            print(f"  Comparisons: {metrics['comparisons']}")
            print(f"  Index size: {sizes[method]:.2f} MB")
            print(f"  Build time: {build_times[method]:.4f} s")

    results_df = pd.DataFrame(all_results)

    results_path = RESULT_DIR / f"{EMBEDDING_MODE}_evaluation.csv"
    results_df.to_csv(results_path, index=False)

    json_path = RESULT_DIR / f"{EMBEDDING_MODE}_evaluation.json"
    json_path.write_text(
        json.dumps(all_results, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    columns = [
        "gallery_fraction", "gallery_size", "method", "threshold",
        "top1", "top5", "baks", "baus", "geomean", "auroc",
        "recall_at_k", "avg_latency_ms", "p95_latency_ms", "qps",
        "comparisons", "index_build_time_s", "index_size_mb",
    ]

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(results_df[columns].to_string(index=False))
    print(f"\nCSV saved to: {results_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(e)
        sys.exit(1)
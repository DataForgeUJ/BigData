"""Evaluation metrics for open-set retrieval."""

import numpy as np


def calculate_auroc(known_scores, unknown_scores):
    """Calculate AUROC using rank statistics."""
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)

    if not len(known_scores) or not len(unknown_scores):
        return np.nan

    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores)),
    ])

    order = np.argsort(scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    ranks = np.empty(len(scores), dtype=float)

    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[i:j] = (i + j + 1) / 2.0
        i = j

    positive_ranks = ranks[sorted_labels == 1]
    n_positive, n_negative = len(known_scores), len(unknown_scores)

    return float(
        (positive_ranks.sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def calculate_baks(known_accepted_correct, known_count):
    """Calculate Balanced Accuracy for Known Samples."""
    return known_accepted_correct / known_count if known_count else np.nan


def calculate_baus(unknown_rejected, unknown_count):
    """Calculate Balanced Accuracy for Unknown Samples."""
    return unknown_rejected / unknown_count if unknown_count else np.nan


def calculate_geomean(baks, baus):
    """Calculate open-set geometric mean."""
    if np.isnan(baks) or np.isnan(baus):
        return np.nan
    return float(np.sqrt(baks * baus))


def calculate_qps(avg_latency_ms):
    """Calculate queries per second."""
    return 1000.0 / avg_latency_ms if avg_latency_ms > 0 else np.nan
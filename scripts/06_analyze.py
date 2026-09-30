"""Analyze retrieval evaluation results and generate report figures."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RESULT_ROOT = Path("artifacts/results")

MODES = ["frozen", "finetuned"]

METHODS = ["Exact", "LSH", "HNSW"]

FIGURE_DPI = 200


# ============================================================
# Helpers
# ============================================================

def load_results(mode):
    path = (
        RESULT_ROOT
        / mode
        / f"{mode}_evaluation.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "gallery_fraction",
        "gallery_size",
        "method",
        "top1",
        "top5",
        "geomean",
        "auroc",
        "avg_latency_ms",
        "p95_latency_ms",
        "qps",
        "comparisons",
        "index_build_time_s",
        "index_size_mb",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{mode}: missing columns: {missing}"
        )

    return df


def save_figure(path):
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()


def plot_metric(
    df,
    metric,
    ylabel,
    title,
    output_path,
):
    plt.figure(figsize=(8, 5))

    for method in METHODS:
        subset = df[df["method"] == method]

        plt.plot(
            subset["gallery_fraction"] * 100,
            subset[metric],
            marker="o",
            label=method,
        )

    plt.xlabel("Gallery Size (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([25, 50, 75, 100])
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path)


def plot_mode_comparison(
    frozen,
    finetuned,
    metric,
    ylabel,
    title,
    output_path,
):
    plt.figure(figsize=(8, 5))

    for method in METHODS:
        frozen_subset = frozen[
            frozen["method"] == method
        ]

        finetuned_subset = finetuned[
            finetuned["method"] == method
        ]

        plt.plot(
            frozen_subset["gallery_fraction"] * 100,
            frozen_subset[metric],
            marker="o",
            linestyle="--",
            label=f"Frozen - {method}",
        )

        plt.plot(
            finetuned_subset["gallery_fraction"] * 100,
            finetuned_subset[metric],
            marker="o",
            label=f"Fine-tuned - {method}",
        )

    plt.xlabel("Gallery Size (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([25, 50, 75, 100])
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path)


def create_summary(df):
    """Create a compact summary using the 100% gallery."""

    summary = df[
        df["gallery_fraction"] == 1.0
    ].copy()

    columns = [
        "method",
        "top1",
        "top5",
        "baks",
        "baus",
        "geomean",
        "auroc",
        "avg_latency_ms",
        "p95_latency_ms",
        "qps",
        "comparisons",
        "index_build_time_s",
        "index_size_mb",
    ]

    return summary[columns].sort_values(
        "method"
    )


def create_combined_summary(
    frozen,
    finetuned,
):
    frozen_summary = create_summary(
        frozen
    ).copy()

    finetuned_summary = create_summary(
        finetuned
    ).copy()

    frozen_summary.insert(
        0,
        "embedding_mode",
        "frozen",
    )

    finetuned_summary.insert(
        0,
        "embedding_mode",
        "finetuned",
    )

    return pd.concat(
        [
            frozen_summary,
            finetuned_summary,
        ],
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("STEP 06 - RETRIEVAL ANALYSIS")
    print("=" * 70)
    print()

    results = {}

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    for mode in MODES:

        print(
            f"Loading {mode} results..."
        )

        results[mode] = load_results(
            mode
        )

        print(
            f"  Rows : "
            f"{len(results[mode])}"
        )

    frozen = results["frozen"]
    finetuned = results["finetuned"]

    print()

    # --------------------------------------------------------
    # Generate per-mode figures
    # --------------------------------------------------------

    for mode in MODES:

        df = results[mode]

        figure_dir = (
            RESULT_ROOT
            / mode
            / "figures"
        )

        figure_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Generating {mode} figures..."
        )

        plot_metric(
            df,
            "top1",
            "Top-1 Identification Accuracy",
            f"{mode.title()} - Top-1 vs Gallery Size",
            figure_dir / "top1_vs_gallery.png",
        )

        plot_metric(
            df,
            "top5",
            "Top-5 Identification Accuracy",
            f"{mode.title()} - Top-5 vs Gallery Size",
            figure_dir / "top5_vs_gallery.png",
        )

        plot_metric(
            df,
            "geomean",
            "Open-Set GeoMean",
            f"{mode.title()} - GeoMean vs Gallery Size",
            figure_dir / "geomean_vs_gallery.png",
        )

        plot_metric(
            df,
            "auroc",
            "AUROC",
            f"{mode.title()} - AUROC vs Gallery Size",
            figure_dir / "auroc_vs_gallery.png",
        )

        plot_metric(
            df,
            "avg_latency_ms",
            "Average Latency (ms)",
            f"{mode.title()} - Latency vs Gallery Size",
            figure_dir / "latency_vs_gallery.png",
        )

        plot_metric(
            df,
            "qps",
            "Queries per Second",
            f"{mode.title()} - QPS vs Gallery Size",
            figure_dir / "qps_vs_gallery.png",
        )

        plot_metric(
            df,
            "index_size_mb",
            "Index Size (MB)",
            f"{mode.title()} - Index Size vs Gallery Size",
            figure_dir / "index_size_vs_gallery.png",
        )

        plot_metric(
            df,
            "index_build_time_s",
            "Index Build Time (s)",
            f"{mode.title()} - Build Time vs Gallery Size",
            figure_dir / "build_time_vs_gallery.png",
        )

        summary = create_summary(df)

        summary_path = (
            RESULT_ROOT
            / mode
            / f"{mode}_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        print(
            f"  Summary saved: "
            f"{summary_path}"
        )

    # --------------------------------------------------------
    # Fine-tuned vs frozen figures
    # --------------------------------------------------------

    comparison_dir = (
        RESULT_ROOT
        / "comparison"
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Generating frozen vs fine-tuned comparisons..."
    )

    plot_mode_comparison(
        frozen,
        finetuned,
        "top1",
        "Top-1 Identification Accuracy",
        "Frozen vs Fine-tuned - Top-1",
        comparison_dir
        / "top1_frozen_vs_finetuned.png",
    )

    plot_mode_comparison(
        frozen,
        finetuned,
        "top5",
        "Top-5 Identification Accuracy",
        "Frozen vs Fine-tuned - Top-5",
        comparison_dir
        / "top5_frozen_vs_finetuned.png",
    )

    plot_mode_comparison(
        frozen,
        finetuned,
        "geomean",
        "Open-Set GeoMean",
        "Frozen vs Fine-tuned - GeoMean",
        comparison_dir
        / "geomean_frozen_vs_finetuned.png",
    )

    plot_mode_comparison(
        frozen,
        finetuned,
        "auroc",
        "AUROC",
        "Frozen vs Fine-tuned - AUROC",
        comparison_dir
        / "auroc_frozen_vs_finetuned.png",
    )

    plot_mode_comparison(
        frozen,
        finetuned,
        "avg_latency_ms",
        "Average Latency (ms)",
        "Frozen vs Fine-tuned - Latency",
        comparison_dir
        / "latency_frozen_vs_finetuned.png",
    )

    plot_mode_comparison(
        frozen,
        finetuned,
        "qps",
        "Queries per Second",
        "Frozen vs Fine-tuned - QPS",
        comparison_dir
        / "qps_frozen_vs_finetuned.png",
    )

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    combined_summary = create_combined_summary(
        frozen,
        finetuned,
    )

    combined_path = (
        comparison_dir
        / "frozen_vs_finetuned_summary.csv"
    )

    combined_summary.to_csv(
        combined_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("100% GALLERY SUMMARY")
    print("=" * 70)

    display_columns = [
        "embedding_mode",
        "method",
        "top1",
        "top5",
        "geomean",
        "auroc",
        "avg_latency_ms",
        "qps",
        "index_size_mb",
    ]

    print(
        combined_summary[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 70)
    print("STEP 06 COMPLETE")
    print("=" * 70)

    print(
        f"Frozen figures   : "
        f"{RESULT_ROOT / 'frozen' / 'figures'}"
    )

    print(
        f"Fine-tuned figures: "
        f"{RESULT_ROOT / 'finetuned' / 'figures'}"
    )

    print(
        f"Comparison figures: "
        f"{comparison_dir}"
    )

    print(
        f"Summary CSV      : "
        f"{combined_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Analysis interrupted by user."
        )
        sys.exit(1)

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(e)
        sys.exit(1)
"""Analyze retrieval evaluation results and generate report figures."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_ROOT = Path("artifacts/results")
MODES = ["frozen", "finetuned"]
METHODS = ["Exact", "LSH", "HNSW"]
FIGURE_DPI = 200

REQUIRED_COLUMNS = [
    "gallery_fraction", "gallery_size", "method", "top1", "top5",
    "geomean", "auroc", "avg_latency_ms", "p95_latency_ms", "qps",
    "comparisons", "index_build_time_s", "index_size_mb",
]

MODE_METRICS = [
    ("top1", "Top-1 Identification Accuracy", "top1_vs_gallery.png", "Top-1 vs Gallery Size"),
    ("top5", "Top-5 Identification Accuracy", "top5_vs_gallery.png", "Top-5 vs Gallery Size"),
    ("geomean", "Open-Set GeoMean", "geomean_vs_gallery.png", "GeoMean vs Gallery Size"),
    ("auroc", "AUROC", "auroc_vs_gallery.png", "AUROC vs Gallery Size"),
    ("avg_latency_ms", "Average Latency (ms)", "latency_vs_gallery.png", "Latency vs Gallery Size"),
    ("qps", "Queries per Second", "qps_vs_gallery.png", "QPS vs Gallery Size"),
    ("index_size_mb", "Index Size (MB)", "index_size_vs_gallery.png", "Index Size vs Gallery Size"),
    ("index_build_time_s", "Index Build Time (s)", "build_time_vs_gallery.png", "Build Time vs Gallery Size"),
]

COMPARISON_METRICS = [
    ("top1", "Top-1 Identification Accuracy", "top1_frozen_vs_finetuned.png", "Top-1"),
    ("top5", "Top-5 Identification Accuracy", "top5_frozen_vs_finetuned.png", "Top-5"),
    ("geomean", "Open-Set GeoMean", "geomean_frozen_vs_finetuned.png", "GeoMean"),
    ("auroc", "AUROC", "auroc_frozen_vs_finetuned.png", "AUROC"),
    ("avg_latency_ms", "Average Latency (ms)", "latency_frozen_vs_finetuned.png", "Latency"),
    ("qps", "Queries per Second", "qps_frozen_vs_finetuned.png", "QPS"),
]


def load_results(mode):
    path = RESULT_ROOT / mode / f"{mode}_evaluation.csv"

    if not path.exists():
        raise FileNotFoundError(f"Results file not found:\n{path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"{mode}: missing columns: {missing}")

    return df


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_metric(df, metric, ylabel, title, output_path):
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


def plot_mode_comparison(frozen, finetuned, metric, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))

    for method in METHODS:
        frozen_subset = frozen[frozen["method"] == method]
        finetuned_subset = finetuned[finetuned["method"] == method]

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
    columns = [
        "method", "top1", "top5", "baks", "baus", "geomean", "auroc",
        "avg_latency_ms", "p95_latency_ms", "qps", "comparisons",
        "index_build_time_s", "index_size_mb",
    ]

    return (
        df[df["gallery_fraction"] == 1.0][columns]
        .copy()
        .sort_values("method")
    )


def create_combined_summary(frozen, finetuned):
    summaries = []

    for mode, df in [("frozen", frozen), ("finetuned", finetuned)]:
        summary = create_summary(df)
        summary.insert(0, "embedding_mode", mode)
        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)


def main():
    print("=" * 70)
    print("STEP 06 - RETRIEVAL ANALYSIS")
    print("=" * 70)

    results = {}

    for mode in MODES:
        print(f"\nLoading {mode} results...")
        results[mode] = load_results(mode)
        print(f"  Rows : {len(results[mode])}")

    frozen, finetuned = results["frozen"], results["finetuned"]

    for mode, df in results.items():
        figure_dir = RESULT_ROOT / mode / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {mode} figures...")

        for metric, ylabel, filename, title in MODE_METRICS:
            plot_metric(
                df,
                metric,
                ylabel,
                f"{mode.title()} - {title}",
                figure_dir / filename,
            )

        summary_path = RESULT_ROOT / mode / f"{mode}_summary.csv"
        create_summary(df).to_csv(summary_path, index=False)
        print(f"  Summary saved: {summary_path}")

    comparison_dir = RESULT_ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating frozen vs fine-tuned comparisons...")

    for metric, ylabel, filename, title in COMPARISON_METRICS:
        plot_mode_comparison(
            frozen,
            finetuned,
            metric,
            ylabel,
            f"Frozen vs Fine-tuned - {title}",
            comparison_dir / filename,
        )

    combined_summary = create_combined_summary(frozen, finetuned)
    combined_path = comparison_dir / "frozen_vs_finetuned_summary.csv"
    combined_summary.to_csv(combined_path, index=False)

    display_columns = [
        "embedding_mode", "method", "top1", "top5", "geomean",
        "auroc", "avg_latency_ms", "qps", "index_size_mb",
    ]

    print("\n" + "=" * 70)
    print("100% GALLERY SUMMARY")
    print("=" * 70)
    print(combined_summary[display_columns].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 06 COMPLETE")
    print("=" * 70)
    print(f"Frozen figures     : {RESULT_ROOT / 'frozen' / 'figures'}")
    print(f"Fine-tuned figures : {RESULT_ROOT / 'finetuned' / 'figures'}")
    print(f"Comparison figures : {comparison_dir}")
    print(f"Summary CSV        : {combined_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(e)
        sys.exit(1)
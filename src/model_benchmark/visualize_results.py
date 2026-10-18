
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Generate images without opening GUI windows.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "model_benchmark"
OUTPUT_DIR = ROOT / "reports" / "model_benchmark"

PREDICTIONS_FILE = RESULTS_DIR / "predictions_and_metrics.csv"
LANGUAGE_FILE = RESULTS_DIR / "language_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and validate the benchmark reports."""
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing benchmark results: {PREDICTIONS_FILE}\n"
            "Run src/model_benchmark/run_benchmark.py first."
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)

    required = {
        "sample_id",
        "language",
        "model",
        "reference",
        "prediction",
        "normalized_wer",
        "normalized_cer",
        "latency_seconds",
        "status",
    }
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Benchmark CSV is missing columns: {sorted(missing)}"
        )

    predictions = predictions[predictions["status"] == "success"].copy()

    if predictions.empty:
        raise ValueError("No successful benchmark rows are available.")

    predictions["normalized_wer"] = pd.to_numeric(
        predictions["normalized_wer"], errors="coerce"
    )
    predictions["normalized_cer"] = pd.to_numeric(
        predictions["normalized_cer"], errors="coerce"
    )
    predictions["latency_seconds"] = pd.to_numeric(
        predictions["latency_seconds"], errors="coerce"
    )

    return predictions


def save_figure(filename):
    """Save the current Matplotlib figure."""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_model_error_comparison(data):
    """Compare mean WER and CER for each model."""
    metrics = data.groupby("model")[
        ["normalized_wer", "normalized_cer"]
    ].mean()

    ax = metrics.plot(
        kind="bar",
        figsize=(9, 5),
        rot=0,
    )
    ax.set_title("ASR Model Error Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean normalized error rate")
    ax.legend(["WER", "CER"])
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Error rates can legitimately exceed 1.0 (100%).
    save_figure("model_error_comparison.png")


def plot_latency_comparison(data):
    """Compare inference latency for each sample and model."""
    pivot = data.pivot_table(
        index="sample_id",
        columns="model",
        values="latency_seconds",
        aggfunc="mean",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 5),
        rot=0,
    )
    ax.set_title("ASR Inference Latency by Sample")
    ax.set_xlabel("Audio sample")
    ax.set_ylabel("Latency (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    save_figure("latency_comparison.png")


def plot_language_performance(data):
    """Show model error rates separately for each language."""
    summary = data.groupby(["language", "model"], as_index=False).agg(
        mean_wer=("normalized_wer", "mean"),
        mean_cer=("normalized_cer", "mean"),
    )

    languages = list(summary["language"].drop_duplicates())
    models = list(summary["model"].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(languages))
    width = 0.8 / max(len(models), 1)

    for index, model in enumerate(models):
        model_data = (
            summary[summary["model"] == model]
            .set_index("language")
            .reindex(languages)
        )

        offset = (index - (len(models) - 1) / 2) * width

        axes[0].bar(
            x + offset,
            model_data["mean_wer"],
            width,
            label=model,
        )
        axes[1].bar(
            x + offset,
            model_data["mean_cer"],
            width,
            label=model,
        )

    for ax, title, metric in [
        (axes[0], "Word Error Rate", "Mean normalized WER"),
        (axes[1], "Character Error Rate", "Mean normalized CER"),
    ]:
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Language-wise ASR Performance (Preliminary: one sample per language)"
    )
    save_figure("language_performance.png")


def plot_transcript_comparison(data):
    """Create a readable text report image for reference/prediction review."""
    rows = data[
        ["sample_id", "language", "model", "reference", "prediction"]
    ]

    fig_height = max(5, len(rows) * 1.6)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    content = []
    for row in rows.itertuples(index=False):
        content.append(
            f"Sample: {row.sample_id} | Language: {row.language}\n"
            f"Model: {row.model}\n"
            f"Reference: {row.reference}\n"
            f"Prediction: {row.prediction}"
        )

    ax.text(
        0.01,
        0.99,
        "\n\n".join(content),
        va="top",
        ha="left",
        fontsize=9,
        wrap=True,
        family="DejaVu Sans",
    )
    ax.set_title(
        "ASR Reference vs Prediction — Manual Review Required",
        pad=16,
    )

    save_figure("transcript_comparison.png")


def main():
    data = load_data()

    print(f"Successful benchmark rows: {len(data)}")
    print(f"Models: {data['model'].nunique()}")
    print(f"Samples: {data['sample_id'].nunique()}")
    print("\nGenerating visualizations...")

    plot_model_error_comparison(data)
    plot_latency_comparison(data)
    plot_language_performance(data)
    plot_transcript_comparison(data)

    print("\nAll visualizations generated.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        "\nCaution: results use only three audio samples. "
        "Script differences affect code-switched error rates."
    )


if __name__ == "__main__":
    main()

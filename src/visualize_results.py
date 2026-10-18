
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "results" / "asr_metrics_normalized.csv"
OUTPUT_DIR = ROOT / "reports"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    required = {
        "sample_id",
        "language",
        "wer_normalized",
        "cer_normalized",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # WER chart
    plt.figure(figsize=(9, 5))
    plt.bar(df["sample_id"], df["wer_normalized"])
    plt.title("Word Error Rate by Audio Sample")
    plt.xlabel("Sample")
    plt.ylabel("WER")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wer_comparison.png", dpi=150)
    plt.close()

    # CER chart
    plt.figure(figsize=(9, 5))
    plt.bar(df["sample_id"], df["cer_normalized"])
    plt.title("Character Error Rate by Audio Sample")
    plt.xlabel("Sample")
    plt.ylabel("CER")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cer_comparison.png", dpi=150)
    plt.close()

    # Summary report
    summary = pd.DataFrame([{
        "samples_evaluated": len(df),
        "mean_normalized_wer": df["wer_normalized"].mean(),
        "mean_normalized_cer": df["cer_normalized"].mean(),
        "median_normalized_wer": df["wer_normalized"].median(),
        "median_normalized_cer": df["cer_normalized"].median(),
    }])

    summary.to_csv(
        OUTPUT_DIR / "benchmark_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n--- Benchmark Summary ---")
    print(summary.to_string(index=False))
    print(f"\nWER chart: {OUTPUT_DIR / 'wer_comparison.png'}")
    print(f"CER chart: {OUTPUT_DIR / 'cer_comparison.png'}")
    print(f"Summary: {OUTPUT_DIR / 'benchmark_summary.csv'}")


if __name__ == "__main__":
    main()

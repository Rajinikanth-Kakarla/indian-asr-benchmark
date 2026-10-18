
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "results" / "model_benchmark" / "predictions_and_metrics.csv"
OUTPUT_FILE = ROOT / "results" / "model_benchmark" / "language_summary.csv"


def main():
    data = pd.read_csv(INPUT_FILE)

    required = {
        "sample_id",
        "language",
        "model",
        "normalized_wer",
        "normalized_cer",
        "latency_seconds",
        "status",
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    successful = data[data["status"] == "success"].copy()

    summary = (
        successful.groupby(["language", "model"], as_index=False)
        .agg(
            samples=("sample_id", "nunique"),
            mean_normalized_wer=("normalized_wer", "mean"),
            mean_normalized_cer=("normalized_cer", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\n=== Per-Language ASR Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

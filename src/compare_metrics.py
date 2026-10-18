
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "results" / "asr_metrics.csv"
NORMALIZED_PATH = ROOT / "results" / "asr_metrics_normalized.csv"
OUTPUT_PATH = ROOT / "results" / "metrics_comparison.csv"


def main():
    for path in (RAW_PATH, NORMALIZED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required report not found: {path}")

    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    normalized = pd.read_csv(NORMALIZED_PATH, encoding="utf-8-sig")

    raw_required = {"sample_id", "language", "wer", "cer"}
    norm_required = {
        "sample_id",
        "language",
        "wer_normalized",
        "cer_normalized",
    }

    if not raw_required.issubset(raw.columns):
        raise ValueError("Raw metrics report has unexpected columns.")

    if not norm_required.issubset(normalized.columns):
        raise ValueError("Normalized metrics report has unexpected columns.")

    comparison = raw.merge(
        normalized,
        on=["sample_id", "language"],
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(raw) or len(comparison) != len(normalized):
        raise ValueError("Reports do not contain matching samples.")

    comparison["wer_change_pp"] = (
        comparison["wer_normalized"] - comparison["wer"]
    ) * 100

    comparison["cer_change_pp"] = (
        comparison["cer_normalized"] - comparison["cer"]
    ) * 100

    comparison.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n--- Raw vs Normalized Comparison ---")
    print(
        comparison[
            [
                "sample_id",
                "wer",
                "wer_normalized",
                "wer_change_pp",
                "cer",
                "cer_normalized",
                "cer_change_pp",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n--- Mean Scores ---")
    print(f"Raw mean WER: {comparison['wer'].mean() * 100:.2f}%")
    print(
        "Normalized mean WER: "
        f"{comparison['wer_normalized'].mean() * 100:.2f}%"
    )
    print(f"Raw mean CER: {comparison['cer'].mean() * 100:.2f}%")
    print(
        "Normalized mean CER: "
        f"{comparison['cer_normalized'].mean() * 100:.2f}%"
    )
    print(f"\nSaved comparison report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

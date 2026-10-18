
from pathlib import Path

import pandas as pd
from jiwer import wer, cer


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "results" / "asr_predictions.csv"
OUTPUT_PATH = ROOT / "results" / "asr_metrics.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Predictions not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    required = {"sample_id", "language", "reference_text", "predicted_text", "status"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    successful = df[df["status"] == "success"].copy()

    if successful.empty:
        raise ValueError("No successful transcriptions to evaluate.")

    rows = []

    for _, row in successful.iterrows():
        reference = str(row["reference_text"])
        prediction = str(row["predicted_text"])

        rows.append({
            "sample_id": row["sample_id"],
            "language": row["language"],
            "reference_text": reference,
            "predicted_text": prediction,
            "wer": wer(reference, prediction),
            "cer": cer(reference, prediction),
        })

    metrics = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n--- Per-sample ASR Metrics ---")
    print(metrics[["sample_id", "language", "wer", "cer"]].to_string(index=False))

    print("\n--- Overall Metrics ---")
    print(f"Samples evaluated: {len(metrics)}")
    print(f"Mean WER: {metrics['wer'].mean():.4f} ({metrics['wer'].mean() * 100:.2f}%)")
    print(f"Mean CER: {metrics['cer'].mean():.4f} ({metrics['cer'].mean() * 100:.2f}%)")
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

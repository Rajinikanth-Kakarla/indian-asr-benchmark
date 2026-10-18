
from pathlib import Path
import re
import unicodedata

import pandas as pd
from jiwer import wer, cer


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "results" / "asr_predictions.csv"
OUTPUT_PATH = ROOT / "results" / "asr_metrics_normalized.csv"


def normalize_text(text):
    """Normalize Unicode, casing, punctuation, and whitespace."""
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()

    # Replace punctuation with spaces instead of joining adjacent words.
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("P"):
            cleaned.append(" ")
        else:
            cleaned.append(char)

    text = "".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Predictions not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    required = {
        "sample_id",
        "language",
        "reference_text",
        "predicted_text",
        "status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    successful = df[df["status"] == "success"].copy()
    if successful.empty:
        raise ValueError("No successful transcriptions to evaluate.")

    rows = []

    for _, row in successful.iterrows():
        reference = normalize_text(row["reference_text"])
        prediction = normalize_text(row["predicted_text"])

        rows.append({
            "sample_id": row["sample_id"],
            "language": row["language"],
            "normalized_reference": reference,
            "normalized_prediction": prediction,
            "wer_normalized": wer(reference, prediction),
            "cer_normalized": cer(reference, prediction),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n--- Normalized ASR Metrics ---")
    print(
        metrics[
            ["sample_id", "language", "wer_normalized", "cer_normalized"]
        ].to_string(index=False)
    )

    print("\n--- Overall Normalized Metrics ---")
    print(f"Samples evaluated: {len(metrics)}")
    print(f"Mean normalized WER: {metrics['wer_normalized'].mean():.4f}")
    print(f"Mean normalized CER: {metrics['cer_normalized'].mean():.4f}")
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

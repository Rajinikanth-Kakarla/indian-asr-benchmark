
from pathlib import Path
import unicodedata

import pandas as pd
from jiwer import wer, cer

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "results" / "model_benchmark" / "predictions_and_metrics.csv"
OUTPUT_FILE = ROOT / "results" / "model_benchmark" / "codeswitch_analysis.csv"


def normalize(text):
    """Normalize Unicode and punctuation without changing writing systems."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    result = []

    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"L", "M", "N"} or char.isspace():
            result.append(char)
        else:
            result.append(" ")

    return " ".join("".join(result).split())


def script_profile(text):
    """Count letters by Unicode script; this is diagnostic, not language detection."""
    counts = {"Latin": 0, "Devanagari": 0, "Telugu": 0, "Tamil": 0, "Other": 0}

    for char in str(text):
        if not unicodedata.category(char).startswith("L"):
            continue

        code = ord(char)

        if 0x0041 <= code <= 0x024F:
            counts["Latin"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["Devanagari"] += 1
        elif 0x0C00 <= code <= 0x0C7F:
            counts["Telugu"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            counts["Tamil"] += 1
        else:
            counts["Other"] += 1

    return counts


def main():
    data = pd.read_csv(INPUT_FILE)
    mixed = data[data["language"] == "Telugu-Hindi"].copy()

    if mixed.empty:
        raise ValueError("No Telugu-Hindi samples found in the benchmark results.")

    rows = []

    for row in mixed.itertuples(index=False):
        reference = normalize(row.reference)
        prediction = normalize(row.prediction)

        ref_scripts = script_profile(row.reference)
        pred_scripts = script_profile(row.prediction)

        rows.append({
            "sample_id": row.sample_id,
            "model": row.model,
            "reference": row.reference,
            "prediction": row.prediction,
            "normalized_wer_same_script": wer(reference, prediction),
            "normalized_cer_same_script": cer(reference, prediction),
            "reference_script_counts": str(ref_scripts),
            "prediction_script_counts": str(pred_scripts),
            "transliteration_applied": False,
            "interpretation": (
                "Diagnostic only: script differences may affect error rates; "
                "no transliteration or semantic equivalence was assumed."
            ),
        })

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n=== Code-Switch Script Analysis ===")
    print(report.to_string(index=False))
    print(f"\nSaved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

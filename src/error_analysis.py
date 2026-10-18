
from pathlib import Path

import pandas as pd
from jiwer import process_words, process_characters


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "results" / "asr_metrics_normalized.csv"
OUTPUT_DIR = ROOT / "results"

WORD_ERRORS_PATH = OUTPUT_DIR / "word_error_details.csv"
CHAR_ERRORS_PATH = OUTPUT_DIR / "character_error_details.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Normalized metrics not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    required = {
        "sample_id",
        "language",
        "normalized_reference",
        "normalized_prediction",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    word_rows = []
    char_rows = []

    for _, row in df.iterrows():
        reference = str(row["normalized_reference"])
        prediction = str(row["normalized_prediction"])

        word_result = process_words(reference, prediction)
        char_result = process_characters(reference, prediction)

        word_rows.append({
            "sample_id": row["sample_id"],
            "language": row["language"],
            "reference_text": reference,
            "predicted_text": prediction,
            "substitutions": word_result.substitutions,
            "deletions": word_result.deletions,
            "insertions": word_result.insertions,
            "wer": word_result.wer,
        })

        char_rows.append({
            "sample_id": row["sample_id"],
            "language": row["language"],
            "character_substitutions": char_result.substitutions,
            "character_deletions": char_result.deletions,
            "character_insertions": char_result.insertions,
            "cer": char_result.cer,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(word_rows).to_csv(
        WORD_ERRORS_PATH, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(char_rows).to_csv(
        CHAR_ERRORS_PATH, index=False, encoding="utf-8-sig"
    )

    print("\n--- Word-Level Error Analysis ---")
    print(pd.DataFrame(word_rows)[[
        "sample_id", "substitutions", "deletions", "insertions", "wer"
    ]].to_string(index=False))

    print("\n--- Character-Level Error Analysis ---")
    print(pd.DataFrame(char_rows)[[
        "sample_id",
        "character_substitutions",
        "character_deletions",
        "character_insertions",
        "cer",
    ]].to_string(index=False))

    print(f"\nWord error report: {WORD_ERRORS_PATH}")
    print(f"Character error report: {CHAR_ERRORS_PATH}")


if __name__ == "__main__":
    main()

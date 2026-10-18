
from pathlib import Path

import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "manifest_validation.csv"

REQUIRED_COLUMNS = [
    "sample_id",
    "audio_path",
    "language",
    "reference_text",
]


def validate_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH,
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    results = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"].strip()
        audio_path = row["audio_path"].strip()
        language = row["language"].strip()
        reference = row["reference_text"].strip()

        errors = []

        if not sample_id:
            errors.append("missing_sample_id")

        if not language:
            errors.append("missing_language")

        if not reference:
            errors.append("missing_reference_transcript")

        audio_file = PROJECT_ROOT / audio_path

        if not audio_path:
            errors.append("missing_audio_path")
        elif not audio_file.is_file():
            errors.append("audio_file_not_found")
        else:
            try:
                info = sf.info(str(audio_file))

                if info.frames <= 0:
                    errors.append("empty_audio")

                if info.samplerate <= 0:
                    errors.append("invalid_sample_rate")

                if info.channels <= 0:
                    errors.append("invalid_channel_count")

                duration = (
                    info.frames / info.samplerate
                    if info.samplerate > 0
                    else 0
                )

            except Exception as exc:
                duration = None
                errors.append(
                    f"unreadable_audio:{type(exc).__name__}"
                )
        if audio_path and not audio_file.is_file():
            duration = None
        elif not audio_path:
            duration = None

        results.append({
            "sample_id": sample_id,
            "audio_path": audio_path,
            "language": language,
            "duration_seconds": duration,
            "status": "PASS" if not errors else "FAIL",
            "errors": "; ".join(errors),
        })

    report = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    report.to_csv(REPORT_PATH, index=False)

    total = len(report)
    passed = (
        int((report["status"] == "PASS").sum())
        if total else 0
    )
    failed = total - passed

    print(f"Manifest rows: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Report: {REPORT_PATH}")

    if total == 0:
        print("No samples found. Add real audio and transcripts.")
    elif failed:
        print("Fix the reported errors before running ASR.")
    else:
        print("All manifest entries passed validation.")


if __name__ == "__main__":
    validate_manifest()

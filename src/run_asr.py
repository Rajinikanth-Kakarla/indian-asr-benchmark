
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
OUTPUT_PATH = ROOT / "results" / "asr_predictions.csv"

MODEL_SIZE = "base"


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")

    required = {
        "sample_id",
        "audio_path",
        "language",
        "reference_text",
    }
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper model: {MODEL_SIZE}")
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )

    results = []

    for _, row in manifest.iterrows():
        audio_path = Path(str(row["audio_path"]))

        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path

        print(f"\nTranscribing: {row['sample_id']}")

        result = {
            "sample_id": row["sample_id"],
            "language": row["language"],
            "reference_text": row["reference_text"],
            "predicted_text": "",
            "detected_language": "",
            "status": "",
            "error": "",
        }

        try:
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
            )

            predicted_text = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            result["predicted_text"] = predicted_text
            result["detected_language"] = info.language
            result["status"] = "success"

            print(f"Detected language: {info.language}")
            print(f"Prediction: {predicted_text}")

        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            print(f"ERROR: {exc}")

        results.append(result)

        # Save after every sample so completed work is retained.
        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    successful = sum(r["status"] == "success" for r in results)
    failed = len(results) - successful

    print("\n--- ASR Summary ---")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

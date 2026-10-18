
from pathlib import Path
import time

import pandas as pd
from faster_whisper import WhisperModel
from jiwer import wer, cer


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.csv"
RESULT_DIR = ROOT / "results" / "telephony"
OUTPUT_FILE = RESULT_DIR / "resampling_comparison.csv"

MODEL_SIZE = "base"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")

    required_columns = {
        "sample_id", "audio_path", "language", "reference_text"
    }
    if not required_columns.issubset(manifest.columns):
        raise ValueError(
            f"Manifest must contain: {sorted(required_columns)}"
        )

    if manifest["reference_text"].isna().any():
        raise ValueError("Some reference transcripts are missing")

    print(f"Loading Faster-Whisper model: {MODEL_SIZE}")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    conditions = {
        "original_44100hz": lambda row: ROOT / row["audio_path"],
        "resampled_16000hz": lambda row: (
            ROOT / "data" / "processed" / "16000hz"
            / f"{row['sample_id']}.wav"
        ),
        "resampled_8000hz": lambda row: (
            ROOT / "data" / "processed" / "8000hz"
            / f"{row['sample_id']}.wav"
        ),
    }

    results = []

    for _, row in manifest.iterrows():
        reference = str(row["reference_text"])

        for condition, path_builder in conditions.items():
            audio_path = path_builder(row)

            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Missing audio for {row['sample_id']}: {audio_path}"
                )

            start = time.perf_counter()

            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
            )
            prediction = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            latency_seconds = time.perf_counter() - start

            results.append({
                "sample_id": row["sample_id"],
                "language_label": row["language"],
                "condition": condition,
                "audio_path": str(audio_path.relative_to(ROOT)),
                
                "sample_rate_hz": (
    44100 if condition == "original_44100hz"
    else 16000 if condition == "resampled_16000hz"
    else 8000
),
                "detected_language": info.language,
                "reference_text": reference,
                "predicted_text": prediction,
                "wer": wer(reference, prediction),
                "cer": cer(reference, prediction),
                "inference_latency_seconds": round(
                    latency_seconds, 4
                ),
            })

            print(
                f"{row['sample_id']} | {condition} | "
                f"WER={results[-1]['wer']:.3f} | "
                f"CER={results[-1]['cer']:.3f} | "
                f"Latency={latency_seconds:.2f}s"
            )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nMean metrics by audio condition:")
    summary = df.groupby("condition").agg(
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        mean_latency_seconds=("inference_latency_seconds", "mean"),
    ).round(4)

    print(summary.to_string())
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

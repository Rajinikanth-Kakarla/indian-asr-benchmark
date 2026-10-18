
from pathlib import Path
import time

import pandas as pd
from faster_whisper import WhisperModel
from jiwer import wer, cer


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.csv"
CLEAN_DIR = ROOT / "data" / "processed" / "16000hz"
NOISE_DIR = ROOT / "data" / "processed" / "noise"
RESULT_DIR = ROOT / "results" / "telephony"

DETAILS_FILE = RESULT_DIR / "noise_robustness_results.csv"
SUMMARY_FILE = RESULT_DIR / "noise_robustness_summary.csv"

MODEL_SIZE = "base"


def get_conditions(sample_id):
    return {
        "clean_16khz": CLEAN_DIR / f"{sample_id}.wav",
        "noise_20db": (
            NOISE_DIR / "snr_20db" / f"{sample_id}.wav"
        ),
        "noise_10db": (
            NOISE_DIR / "snr_10db" / f"{sample_id}.wav"
        ),
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")

    required = {"sample_id", "language", "reference_text"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")

    if manifest["reference_text"].isna().any():
        raise ValueError("Missing reference transcript detected")

    print(f"Loading Faster-Whisper model: {MODEL_SIZE}")
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )

    results = []

    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        reference = str(row["reference_text"])

        for condition, audio_path in get_conditions(sample_id).items():
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Missing {condition} audio: {audio_path}"
                )

            start = time.perf_counter()

            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
            )
            prediction = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            latency = time.perf_counter() - start

            result = {
                "sample_id": sample_id,
                "language_label": row["language"],
                "condition": condition,
                "reference_text": reference,
                "predicted_text": prediction,
                "detected_language": info.language,
                "wer": wer(reference, prediction),
                "cer": cer(reference, prediction),
                "inference_latency_seconds": round(latency, 4),
            }
            results.append(result)

            print(
                f"{sample_id} | {condition} | "
                f"WER={result['wer']:.3f} | "
                f"CER={result['cer']:.3f} | "
                f"Latency={latency:.2f}s"
            )

    details = pd.DataFrame(results)
    details.to_csv(
        DETAILS_FILE, index=False, encoding="utf-8-sig"
    )

    summary = (
        details.groupby("condition")
        .agg(
            sample_count=("sample_id", "count"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_latency_seconds=(
                "inference_latency_seconds", "mean"
            ),
        )
        .round(4)
    )

    summary.to_csv(SUMMARY_FILE, encoding="utf-8-sig")

    print("\nMean results by condition:")
    print(summary.to_string())
    print(f"\nDetailed results: {DETAILS_FILE}")
    print(f"Summary results:  {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

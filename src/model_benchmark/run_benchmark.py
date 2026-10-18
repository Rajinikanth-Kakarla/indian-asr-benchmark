
from pathlib import Path
import time
import warnings

import unicodedata
import librosa
import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from transformers import AutoModel
import jiwer

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.csv"
OUTPUT_DIR = ROOT / "results" / "model_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDIC_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000






def normalize_text(text):
    """Normalize Unicode text while preserving letters and combining marks."""
    text = unicodedata.normalize("NFC", str(text)).lower()

    cleaned = []
    for char in text:
        category = unicodedata.category(char)

        # Preserve letters, combining marks, numbers, and whitespace.
        if category[0] in {"L", "M", "N"} or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    return " ".join("".join(cleaned).split())



def calculate_metrics(reference, prediction):
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)

    if not ref:
        return None, None

    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    return wer, cer


def load_audio(audio_path):
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
    )
    return np.asarray(audio, dtype=np.float32)


def main():
    manifest = pd.read_csv(MANIFEST)
    required = {"sample_id", "audio_path", "language", "reference_text"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    print("Loading Faster-Whisper...")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    print("Loading IndicConformer...")
    indic = AutoModel.from_pretrained(
        INDIC_MODEL_ID,
        trust_remote_code=True,
    )
    indic.eval()

    results = []

    for row in manifest.itertuples(index=False):
        audio_path = ROOT / row.audio_path
        if not audio_path.exists():
            print(f"SKIP {row.sample_id}: audio not found: {audio_path}")
            continue

        print(f"\nSample: {row.sample_id} ({row.language})")
        audio = load_audio(audio_path)
        reference = str(row.reference_text)

        # Faster-Whisper
        whisper_language = {
            "Hindi": "hi",
            "Telugu": "te",
        }.get(row.language)

        start = time.perf_counter()
        segments, _ = whisper.transcribe(
            audio,
            language=whisper_language,
            beam_size=5,
        )
        whisper_prediction = " ".join(
            segment.text.strip() for segment in segments
        )
        whisper_latency = time.perf_counter() - start

        wer, cer = calculate_metrics(reference, whisper_prediction)
        results.append({
            "sample_id": row.sample_id,
            "language": row.language,
            "model": "Faster-Whisper base",
            "reference": reference,
            "prediction": whisper_prediction,
            "normalized_wer": wer,
            "normalized_cer": cer,
            "latency_seconds": whisper_latency,
            "status": "success",
        })
        print(f"Faster-Whisper: {whisper_prediction}")

        # IndicConformer language selection
        indic_language = {
            "Hindi": "hi",
            "Telugu": "te",
            "Telugu-Hindi": "te",
        }.get(row.language)

        if indic_language is None:
            print(f"SKIP IndicConformer: unsupported language {row.language}")
            continue

        waveform = torch.from_numpy(audio).unsqueeze(0)

        start = time.perf_counter()
        with torch.inference_mode():
            indic_prediction = indic(waveform, indic_language, "ctc")
        indic_latency = time.perf_counter() - start

        if not isinstance(indic_prediction, str):
            indic_prediction = str(indic_prediction)

        wer, cer = calculate_metrics(reference, indic_prediction)
        results.append({
            "sample_id": row.sample_id,
            "language": row.language,
            "model": "IndicConformer 600M",
            "reference": reference,
            "prediction": indic_prediction,
            "normalized_wer": wer,
            "normalized_cer": cer,
            "latency_seconds": indic_latency,
            "status": "success",
        })
        print(f"IndicConformer: {indic_prediction}")

    if not results:
        raise RuntimeError("No samples were successfully benchmarked.")

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_DIR / "predictions_and_metrics.csv", index=False)

    summary = (
        output.groupby("model", as_index=False)
        .agg(
            samples=("sample_id", "count"),
            mean_normalized_wer=("normalized_wer", "mean"),
            mean_normalized_cer=("normalized_cer", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
    )
    summary.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)

    print("\n=== Model Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

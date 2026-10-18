
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "telephony"
REPORTS = ROOT / "reports" / "telephony"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    resampling_path = RESULTS / "resampling_comparison.csv"
    noise_path = RESULTS / "noise_robustness_results.csv"
    snr_path = RESULTS / "verified_snr.csv"

    for path in (resampling_path, noise_path, snr_path):
        if not path.exists():
            raise FileNotFoundError(f"Required result file missing: {path}")

    resampling = pd.read_csv(resampling_path)
    noise = pd.read_csv(noise_path)
    snr = pd.read_csv(snr_path)

    resampling_summary = (
        resampling.groupby("condition")
        .agg(
            sample_count=("sample_id", "count"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_latency_seconds=("inference_latency_seconds", "mean"),
        )
        .reset_index()
    )

    noise_summary = (
        noise.groupby("condition")
        .agg(
            sample_count=("sample_id", "count"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_latency_seconds=("inference_latency_seconds", "mean"),
        )
        .reset_index()
    )

    resampling_summary.to_csv(
        REPORTS / "resampling_summary.csv", index=False
    )
    noise_summary.to_csv(
        REPORTS / "noise_summary.csv", index=False
    )

    # Plot resampling error rates.
    ax = resampling_summary.set_index("condition")[
        ["mean_wer", "mean_cer"]
    ].plot(kind="bar")
    ax.set_title("ASR Error Rates by Sampling Condition")
    ax.set_ylabel("Mean error rate")
    ax.set_xlabel("Audio condition")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS / "resampling_error_rates.png", dpi=150)
    plt.close()

    # Plot noise-condition error rates.
    ax = noise_summary.set_index("condition")[
        ["mean_wer", "mean_cer"]
    ].plot(kind="bar")
    ax.set_title("ASR Error Rates by Noise Condition")
    ax.set_ylabel("Mean error rate")
    ax.set_xlabel("Noise condition")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS / "noise_error_rates.png", dpi=150)
    plt.close()

    report = f"""# Telephony Audio Processing and ASR Robustness

## Objective

Evaluate how resampling and controlled synthetic white noise affect
Faster-Whisper transcription quality and inference latency.

## Experimental setup

- ASR model: Faster-Whisper base
- Device: CPU, int8 compute type
- Dataset: {resampling['sample_id'].nunique()} short recordings
- Original sample rate: 44.1 kHz
- Resampled rates: 16 kHz and 8 kHz
- Noise conditions: 20 dB and 10 dB SNR
- Metrics: WER, CER and inference latency

## Resampling results

{resampling_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")}

## Noise robustness results

{noise_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")}

## SNR validation

Verified SNR measurements:

{snr.to_string(index=False, float_format=lambda x: f"{x:.4f}")}

## Interpretation and limitations

- The dataset contains only three short recordings. Results are exploratory.
- Raw WER and CER can be affected by script differences between references
  and predictions, particularly for Indian-language speech.
- Synthetic white noise does not reproduce every characteristic of
  real telephony channels.
- Latency is machine- and runtime-dependent; repeat measurements are
  needed for reliable performance comparisons.
- The observed error rates do not establish that noise improves ASR.
- Reference transcripts should be independently verified before expanding
  the benchmark or drawing language-level conclusions.

## Reproducibility

Run the inspection, resampling, noise generation, ASR evaluation and SNR
verification scripts in `src/audio_processing/`. Preserve original audio
and record any changes to model settings or dataset references.
"""

    (REPORTS / "README.md").write_text(report, encoding="utf-8")

    print("Report generated:")
    print(REPORTS / "README.md")
    print(REPORTS / "resampling_summary.csv")
    print(REPORTS / "noise_summary.csv")
    print(REPORTS / "resampling_error_rates.png")
    print(REPORTS / "noise_error_rates.png")


if __name__ == "__main__":
    main()

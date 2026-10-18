# Indian-Language ASR Evaluation and Benchmarking

## Overview

A Python-based evaluation pipeline for benchmarking pretrained automatic speech recognition (ASR) on Hindi, Telugu, and Telugu–Hindi code-switched audio. The project validates audio metadata, generates transcripts using Faster-Whisper, calculates Word Error Rate (WER) and Character Error Rate (CER), analyzes recognition errors, and generates visual reports.

## Objectives

- Evaluate pretrained ASR predictions against reference transcripts.
- Measure word-level and character-level transcription errors.
- Compare raw metrics with text-normalized metrics.
- Inspect substitutions, deletions, and insertions.
- Produce reproducible reports and automated tests.

## Technology Stack

Python, Faster-Whisper, CTranslate2, PyAV, Pandas, JiWER, Matplotlib, Pytest.

## Project Structure

```text
indian-asr-benchmark/
├── data/
│   ├── audio/
│   └── manifest.csv
├── src/
│   ├── validate_manifest.py
│   ├── run_asr.py
│   ├── evaluate_asr.py
│   ├── evaluate_normalized.py
│   ├── compare_metrics.py
│   ├── error_analysis.py
│   └── visualize_results.py
├── tests/
│   └── test_evaluation.py
├── results/
└── reports/
```

## Environment Setup

Create and activate a Python virtual environment, then install the required packages:

```bash
python -m pip install faster-whisper jiwer pandas soundfile librosa matplotlib pytest
```

## Run the Pipeline

Run these commands from the project root, with the virtual environment activated:

```bash
python src/validate_manifest.py
python src/run_asr.py
python src/evaluate_asr.py
python src/evaluate_normalized.py
python src/compare_metrics.py
python src/error_analysis.py
python src/visualize_results.py
python -m pytest tests/test_evaluation.py -v
```

The first ASR run may download pretrained model weights and requires an internet connection.

## Evaluation Metrics

- **WER:** word-level substitutions, deletions, and insertions divided by the number of reference words.
- **CER:** character-level edit distance divided by the number of reference characters.
- **Normalized WER/CER:** the same metrics after Unicode normalization, case normalization where applicable, punctuation removal, and whitespace normalization.

Lower scores indicate fewer errors. WER and CER may exceed 100% when insertions are sufficiently numerous.

## Initial Experiment

The initial test set contains three recordings: one Hindi sample, one Telugu sample, and one Telugu–Hindi code-switched sample.

| Metric | Initial mean |
|---|---:|
| Raw WER | 115.87% |
| Normalized WER | 111.71% |
| Raw CER | 91.16% |
| Normalized CER | 90.11% |

These figures describe only the current three-sample test set. They are not representative estimates of general ASR performance.

## Limitations

- The dataset is too small to support reliable language-level performance claims.
- Recognition output may contain incorrect words or scripts.
- Unicode and punctuation normalization does not resolve transliteration or cross-script differences.
- Reference transcript accuracy must be verified against the recordings.
- Results depend on the model version, decoding configuration, audio quality, and transcript conventions.

## Future Improvements

- Expand the test set with verified, diverse recordings.
- Evaluate noise, accent, speaking rate, and code-switching separately.
- Compare multiple pretrained ASR models using the same dataset.
- Add confidence intervals and per-condition error analysis.
- Record model configuration and package versions for reproducibility.


## Project 2: Telephony Audio Processing and ASR Robustness

### Objective

Evaluate how audio sampling rate and controlled background noise affect automatic speech recognition (ASR) accuracy and inference latency. This project extends the original Indian-language ASR benchmark with reproducible audio preprocessing and robustness experiments.

### Features Implemented

- **Audio inspection:** Measure sample rate, channel count, duration, peak amplitude, and a near-full-scale clipping indicator.
- **Mono conversion and resampling:** Generate mono, 16-bit PCM WAV files at 16 kHz and 8 kHz using PyAV audio resampling.
- **Sampling-rate comparison:** Evaluate original 44.1 kHz audio against the resampled 16 kHz and 8 kHz versions using the same Faster-Whisper base model.
- **Controlled noise generation:** Create reproducible white-noise test samples at signal-to-noise ratios (SNR) of 20 dB and 10 dB.
- **Noise robustness evaluation:** Compare word error rate (WER), character error rate (CER), and inference latency across clean and noisy audio conditions.
- **SNR verification:** Measure the generated noise levels to verify that they match their intended SNR targets.
- **Automated reporting:** Generate CSV summaries, Markdown documentation, and plots for comparing experimental conditions.
- **Unit testing:** Test noise-generation properties, reproducibility, silence handling, and numerical validity.

### Technology Stack

Python, PyAV, NumPy, SoundFile, Pandas, Faster-Whisper, JiWER, Matplotlib, and Pytest.

### Experimental Results

#### Sampling-rate comparison

| Input condition | Mean WER | Mean CER | Mean latency (s) |
|---|---:|---:|---:|
| Original 44.1 kHz | 1.1587 | 0.9116 | 1.1902 |
| Resampled 16 kHz | 1.1587 | 0.8799 | 1.1586 |
| Resampled 8 kHz | 1.3122 | 0.8889 | 4.5918 |

#### Noise robustness comparison

| Input condition | Mean WER | Mean CER | Mean latency (s) |
|---|---:|---:|---:|
| Clean 16 kHz | 1.1587 | 0.8799 | 1.2833 |
| White noise, 20 dB SNR | 1.1647 | 0.8425 | 4.1700 |
| White noise, 10 dB SNR | 1.0741 | 0.8203 | 2.8609 |

*These are preliminary results from three audio samples. Lower error rates in some noisy conditions do not establish that noise improves recognition; the sample is too small for that conclusion. Latency is also sensitive to runtime conditions.*

### Generated Outputs

- `results/telephony/audio_inspection.csv` — audio metadata and amplitude inspection.
- `results/telephony/resampling_comparison.csv` — sampling-rate evaluation results.
- `results/telephony/noise_robustness_results.csv` — per-sample noise evaluation results.
- `results/telephony/noise_robustness_summary.csv` — aggregated noise evaluation metrics.
- `results/telephony/verified_snr.csv` — measured SNR verification.
- `reports/telephony/README.md` — experiment report.
- `reports/telephony/resampling_summary.csv` — resampling summary.
- `reports/telephony/noise_summary.csv` — noise summary.
- `reports/telephony/resampling_error_rates.png` — resampling comparison plot.
- `reports/telephony/noise_error_rates.png` — noise comparison plot.

Processed WAV files are generated locally under `data/processed/` and excluded from version control to avoid committing redundant audio artifacts.

### Reproducibility and Testing

The experiments use consistent reference transcripts and the same ASR model configuration across conditions. Noise generation uses a fixed random seed, and measured SNR is verified against the target values.

Run the test suite:

```bash
python -m pytest -v
```

**Current test status:** 10 tests passed, covering the existing evaluation pipeline and the new audio-processing tests.

### Limitations and Future Work

- The current dataset contains only three samples and is intended for pipeline validation, not generalizable
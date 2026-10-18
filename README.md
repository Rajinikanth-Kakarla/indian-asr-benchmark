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

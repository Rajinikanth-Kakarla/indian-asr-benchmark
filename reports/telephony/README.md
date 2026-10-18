# Telephony Audio Processing and ASR Robustness

## Objective

Evaluate how resampling and controlled synthetic white noise affect
Faster-Whisper transcription quality and inference latency.

## Experimental setup

- ASR model: Faster-Whisper base
- Device: CPU, int8 compute type
- Dataset: 3 short recordings
- Original sample rate: 44.1 kHz
- Resampled rates: 16 kHz and 8 kHz
- Noise conditions: 20 dB and 10 dB SNR
- Metrics: WER, CER and inference latency

## Resampling results

        condition  sample_count  mean_wer  mean_cer  mean_latency_seconds
 original_44100hz             3    1.1587    0.9116                1.1902
resampled_16000hz             3    1.1587    0.8799                1.1586
 resampled_8000hz             3    1.3122    0.8889                4.5918

## Noise robustness results

  condition  sample_count  mean_wer  mean_cer  mean_latency_seconds
clean_16khz             3    1.1587    0.8799                1.2833
 noise_10db             3    1.0741    0.8203                2.8609
 noise_20db             3    1.1647    0.8425                4.1700

## SNR validation

Verified SNR measurements:

   sample_id  condition  target_snr_db  measured_snr_db
 hindi_audio noise_20db             20          20.0000
 hindi_audio noise_10db             10          10.0000
  hyderabadi noise_20db             20          20.0000
  hyderabadi noise_10db             10          10.0000
telugu_audio noise_20db             20          20.0000
telugu_audio noise_10db             10          10.0000

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

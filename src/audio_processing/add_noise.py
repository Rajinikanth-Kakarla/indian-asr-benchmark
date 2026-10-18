
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed" / "16000hz"
OUTPUT_DIR = ROOT / "data" / "processed" / "noise"
SNR_LEVELS_DB = (20, 10)
SAMPLE_RATE = 16000


def add_white_noise(audio: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add reproducible white noise at the requested approximate SNR."""
    rng = np.random.default_rng(seed)

    signal_power = float(np.mean(audio ** 2))
    if signal_power <= 1e-12:
        raise ValueError("Audio signal power is too low")

    noise = rng.normal(0, 1, size=audio.shape)
    noise_power = float(np.mean(noise ** 2))

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_power / noise_power)

    noisy = audio + noise

    # Avoid clipping when saving as PCM_16.
    peak = float(np.max(np.abs(noisy)))
    if peak > 0.999:
        noisy *= 0.999 / peak

    return noisy.astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {INPUT_DIR}")

    for path in files:
        audio, sample_rate = sf.read(path, dtype="float32")

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{path.name}: expected {SAMPLE_RATE} Hz, got {sample_rate}"
            )

        for snr_db in SNR_LEVELS_DB:
            noisy = add_white_noise(
                audio,
                snr_db=snr_db,
                seed=16000 + snr_db + sum(path.stem.encode("utf-8")),
            )

            output_dir = OUTPUT_DIR / f"snr_{snr_db}db"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / path.name

            sf.write(
                output_path,
                noisy,
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            print(
                f"Created {output_path.relative_to(ROOT)} "
                f"| sample rate={sample_rate} Hz"
            )


if __name__ == "__main__":
    main()

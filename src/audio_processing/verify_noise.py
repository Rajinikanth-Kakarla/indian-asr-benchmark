
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "processed" / "16000hz"
NOISE_DIR = ROOT / "data" / "processed" / "noise"
OUTPUT = ROOT / "results" / "telephony" / "verified_snr.csv"


def calculate_snr(clean, noisy):
    """Estimate SNR from the clean and noisy waveforms."""
    if clean.shape != noisy.shape:
        raise ValueError("Clean and noisy audio shapes do not match")

    noise = noisy.astype(np.float64) - clean.astype(np.float64)
    signal_power = np.mean(clean.astype(np.float64) ** 2)
    noise_power = np.mean(noise ** 2)

    if signal_power <= 1e-12 or noise_power <= 1e-15:
        return float("inf") if noise_power <= 1e-15 else float("-inf")

    return 10 * np.log10(signal_power / noise_power)


def main():
    rows = []

    for clean_path in sorted(CLEAN_DIR.glob("*.wav")):
        clean, clean_rate = sf.read(clean_path, dtype="float32")

        for condition, folder in [
            ("noise_20db", "snr_20db"),
            ("noise_10db", "snr_10db"),
        ]:
            noisy_path = NOISE_DIR / folder / clean_path.name
            noisy, noisy_rate = sf.read(noisy_path, dtype="float32")

            if clean_rate != noisy_rate:
                raise ValueError(f"Sample rate mismatch: {clean_path.name}")

            measured_snr = calculate_snr(clean, noisy)

            rows.append({
                "sample_id": clean_path.stem,
                "condition": condition,
                "target_snr_db": 20 if condition == "noise_20db" else 10,
                "measured_snr_db": round(measured_snr, 2),
            })

    if not rows:
        raise FileNotFoundError("No clean recordings found")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(df.to_string(index=False))
    print(f"\nSNR verification saved to: {OUTPUT}")


if __name__ == "__main__":
    main()

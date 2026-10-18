
from pathlib import Path

import av
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "results" / "telephony"
OUTPUT_FILE = OUTPUT_DIR / "audio_inspection.csv"


def inspect_audio(audio_path: Path) -> dict:
    """Inspect decoded audio without modifying the original file."""
    with av.open(str(audio_path)) as container:
        audio_streams = [
            stream for stream in container.streams
            if stream.type == "audio"
        ]

        if not audio_streams:
            raise ValueError("No audio stream found")

        stream = audio_streams[0]
        sample_rate = stream.codec_context.sample_rate
        channels = len(stream.codec_context.layout.channels)

        if not sample_rate:
            raise ValueError("Sample rate is unavailable")

        total_samples = 0
        peak_amplitude = 0.0
        clipped_samples = 0
        total_values = 0

        for frame in container.decode(stream):
            audio = frame.to_ndarray()
            total_samples += frame.samples

            # Convert decoded samples to a normalized amplitude range.
            if np.issubdtype(audio.dtype, np.integer):
                info = np.iinfo(audio.dtype)
                audio = audio.astype(np.float64) / max(
                    abs(info.min), info.max
                )
            else:
                audio = audio.astype(np.float64)

            if audio.size:
                peak_amplitude = max(
                    peak_amplitude,
                    float(np.max(np.abs(audio)))
                )
                clipped_samples += int(
                    np.count_nonzero(np.abs(audio) >= 0.999)
                )
                total_values += audio.size

        duration = total_samples / sample_rate

        return {
            "file": audio_path.name,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "duration_seconds": round(duration, 3),
            "peak_amplitude": round(peak_amplitude, 6),
            "clipped_sample_count": clipped_samples,
            "clipped_sample_percentage": round(
                100 * clipped_samples / total_values, 6
            ) if total_values else 0.0,
        }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(AUDIO_DIR.glob("*"))
    audio_files = [
        path for path in audio_files
        if path.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}
    ]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {AUDIO_DIR}")

    results = []

    for audio_path in audio_files:
        try:
            result = inspect_audio(audio_path)
            results.append(result)
            print(f"Inspected: {audio_path.name}")
        except Exception as exc:
            print(f"FAILED: {audio_path.name}: {exc}")

    if not results:
        raise RuntimeError("No audio files could be inspected")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nAudio inspection results:")
    print(df.to_string(index=False))
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

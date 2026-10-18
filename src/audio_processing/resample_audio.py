
from pathlib import Path

import av
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "audio"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET_RATES = (16000, 8000)


def decode_and_resample(input_path: Path, target_rate: int) -> np.ndarray:
    """Decode an audio file and genuinely resample it to mono float audio."""
    chunks = []

    with av.open(str(input_path)) as container:
        streams = [
            stream for stream in container.streams
            if stream.type == "audio"
        ]
        if not streams:
            raise ValueError("No audio stream found")

        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=target_rate,
        )

        for frame in container.decode(streams[0]):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))

        # Flush any samples remaining in the resampler.
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError("No audio samples decoded")

    audio = np.concatenate(chunks).astype(np.float32)

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")

    return audio


def main():
    audio_files = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}
    )

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {INPUT_DIR}")

    for source in audio_files:
        for rate in TARGET_RATES:
            output_folder = OUTPUT_DIR / f"{rate}hz"
            output_folder.mkdir(parents=True, exist_ok=True)

            output_path = output_folder / f"{source.stem}.wav"
            audio = decode_and_resample(source, rate)

            sf.write(output_path, audio, rate, subtype="PCM_16")

            # Verify the saved file's sample rate and channel count.
            saved_info = sf.info(output_path)
            print(
                f"{source.name} -> {output_path.relative_to(ROOT)} | "
                f"rate={saved_info.samplerate} Hz | "
                f"channels={saved_info.channels} | "
                f"duration={saved_info.duration:.3f}s"
            )


if __name__ == "__main__":
    main()

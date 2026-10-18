
import numpy as np

from src.audio_processing.add_noise import add_white_noise


def test_noise_output_has_same_shape():
    audio = np.full(16000, 0.1, dtype=np.float32)

    noisy = add_white_noise(audio, snr_db=20, seed=42)

    assert noisy.shape == audio.shape


def test_noise_generation_is_reproducible():
    audio = np.full(16000, 0.1, dtype=np.float32)

    first = add_white_noise(audio, snr_db=20, seed=42)
    second = add_white_noise(audio, snr_db=20, seed=42)

    np.testing.assert_array_equal(first, second)


def test_noise_generation_rejects_silence():
    audio = np.zeros(16000, dtype=np.float32)

    try:
        add_white_noise(audio, snr_db=20, seed=42)
    except ValueError as exc:
        assert "power is too low" in str(exc)
    else:
        raise AssertionError("Expected silence to be rejected")


def test_noise_output_is_finite():
    audio = np.full(16000, 0.1, dtype=np.float32)

    noisy = add_white_noise(audio, snr_db=10, seed=42)

    assert np.isfinite(noisy).all()

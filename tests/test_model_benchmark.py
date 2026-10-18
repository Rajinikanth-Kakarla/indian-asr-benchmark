
from src.model_benchmark.run_benchmark import (
    normalize_text,
    calculate_metrics,
)


def test_normalize_removes_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_text("  hello   world  ") == "hello world"


def test_normalize_preserves_telugu_script():
    text = "రాము! నిన్న కొత్త సినిమా"
    assert normalize_text(text) == "రాము నిన్న కొత్త సినిమా"


def test_identical_text_has_zero_error():
    wer, cer = calculate_metrics(
        "రాము నిన్న సినిమా",
        "రాము నిన్న సినిమా",
    )
    assert wer == 0.0
    assert cer == 0.0


def test_punctuation_does_not_affect_normalized_metrics():
    wer, cer = calculate_metrics(
        "Hello, World!",
        "hello world",
    )
    assert wer == 0.0
    assert cer == 0.0


def test_empty_reference_returns_no_metrics():
    wer, cer = calculate_metrics("", "hello")
    assert wer is None
    assert cer is None

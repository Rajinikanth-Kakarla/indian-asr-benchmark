
import sys
from pathlib import Path

from jiwer import wer, cer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate_normalized import normalize_text


def test_normalize_removes_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_text("  hello   world  ") == "hello world"


def test_normalize_preserves_telugu_script():
    text = "రాము! ఎలా ఉంది?"
    assert normalize_text(text) == "రాము ఎలా ఉంది"


def test_identical_text_has_zero_error():
    reference = "hello world"
    assert wer(reference, reference) == 0.0
    assert cer(reference, reference) == 0.0


def test_substitution_is_counted():
    assert wer("hello world", "hello there") == 0.5


def test_empty_prediction_has_full_deletion_error():
    assert wer("hello world", "") == 1.0
    assert cer("hello world", "") == 1.0

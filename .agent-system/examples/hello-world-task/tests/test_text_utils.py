import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from text_utils import normalize_whitespace


def test_double_space_collapses():
    assert normalize_whitespace("a  b") == "a b"


def test_tabs_and_newlines_collapse():
    assert normalize_whitespace("a\t\tb\n\nc   d") == "a b c d"


def test_leading_and_trailing_whitespace_stripped():
    assert normalize_whitespace("  a  b  ") == "a b"


def test_empty_string():
    assert normalize_whitespace("") == ""

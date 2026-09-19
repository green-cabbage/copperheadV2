import re


def normalize_whitespace(text: str) -> str:
    """Collapse any run of whitespace (spaces, tabs, newlines) into a
    single space and strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()

"""Plain-text file extraction helpers."""

from .base import normalize_text, read_text_file


def extract_text(filepath: str) -> str:
    """Read plain text content from a local file."""
    return normalize_text(read_text_file(filepath))

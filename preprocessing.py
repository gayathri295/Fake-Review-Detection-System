"""Small, transparent text normalization helpers."""

from __future__ import annotations


def normalize_review(text: str) -> str:
    """Normalize case and whitespace without deleting punctuation or words."""
    return " ".join(str(text).split()).lower()

"""Lightweight PDF extraction based on RAGFlow PlainParser behavior."""

from __future__ import annotations

from .base import join_sections, normalize_text


def _extract_with_pypdf(filepath: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    sections = []
    for page in reader.pages:
        page_text = normalize_text(page.extract_text() or "")
        if page_text:
            sections.append(page_text)
    return join_sections(sections)


def extract_text(filepath: str) -> str:
    """Extract text from text-based PDF files using pypdf."""
    return _extract_with_pypdf(filepath)

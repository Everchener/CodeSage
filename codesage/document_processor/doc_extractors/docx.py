"""DOCX extraction helpers adapted to preserve paragraph and table order."""

from __future__ import annotations

import re

from .base import join_sections, prefix_heading, table_rows_to_text

HEADING_PATTERN = re.compile(r"heading\s*(\d+)", re.IGNORECASE)


def _get_heading_level(paragraph) -> int | None:
    style = getattr(paragraph, "style", None)
    style_name = getattr(style, "name", "") or ""
    match = HEADING_PATTERN.search(style_name)
    if not match:
        return None
    return int(match.group(1))


def _table_to_text(table) -> str:
    rows = [[cell.text for cell in row.cells] for row in table.rows]
    return table_rows_to_text(rows)


def extract_text(filepath: str) -> str:
    """Extract ordered paragraph and table content from a DOCX file."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(filepath)
    sections: list[str] = []

    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            heading_level = _get_heading_level(paragraph)
            sections.append(prefix_heading(text, heading_level) if heading_level else text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            table_text = _table_to_text(table)
            if table_text:
                sections.append(table_text)

    return join_sections(sections)

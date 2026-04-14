"""Lightweight document extractors adapted from RAGFlow ideas."""

from .docx import extract_text as extract_docx_text
from .excel import extract_text as extract_excel_text
from .html import extract_text as extract_html_text
from .markdown import extract_text as extract_markdown_text
from .pdf import extract_text as extract_pdf_text
from .pptx import extract_text as extract_pptx_text
from .text import extract_text as extract_plain_text

__all__ = [
    "extract_docx_text",
    "extract_excel_text",
    "extract_html_text",
    "extract_markdown_text",
    "extract_pdf_text",
    "extract_pptx_text",
    "extract_plain_text",
]

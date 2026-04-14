"""HTML extraction helpers inspired by RAGFlow's block-based parsing."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from .base import html_table_to_text, join_sections, prefix_heading, read_text_file

BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "div",
    "article",
    "section",
    "aside",
    "ul",
    "ol",
    "li",
    "table",
    "pre",
    "code",
    "blockquote",
    "figure",
    "figcaption",
}
TITLE_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _read_text_recursively(element, parser_result: list[dict], parent_name: str | None = None, block_id: str | None = None) -> None:
    from bs4 import Comment, NavigableString, Tag

    if isinstance(element, Comment):
        return

    if isinstance(element, NavigableString):
        content = str(element).strip()
        if not content:
            return
        parser_result.append(
            {
                "content": content,
                "tag_name": parent_name or "text",
                "metadata": {"block_id": block_id},
            }
        )
        return

    if not isinstance(element, Tag):
        return

    if element.name == "table":
        parser_result.append(
            {
                "content": html_table_to_text(str(element)),
                "tag_name": "table",
                "metadata": {"table_id": str(uuid.uuid4())},
            }
        )
        return

    next_block_id = str(uuid.uuid4()) if element.name in BLOCK_TAGS else block_id
    for child in element.children:
        _read_text_recursively(child, parser_result, element.name, next_block_id)


def _merge_blocks(parser_result: Iterable[dict]) -> tuple[list[str], list[str]]:
    blocks: list[str] = []
    tables: list[str] = []
    current_parts: list[str] = []
    last_block_id: str | None = None

    for item in parser_result:
        content = item.get("content", "").strip()
        if not content:
            continue

        tag_name = item.get("tag_name", "")
        block_id = item.get("metadata", {}).get("block_id")

        if tag_name == "table":
            tables.append(content)
            continue

        if tag_name in TITLE_LEVELS:
            content = prefix_heading(content, TITLE_LEVELS[tag_name])

        if block_id and block_id != last_block_id:
            if current_parts:
                blocks.append(" ".join(current_parts))
            current_parts = [content]
            last_block_id = block_id
        else:
            current_parts.append(content)

    if current_parts:
        blocks.append(" ".join(current_parts))

    return blocks, tables


def extract_text(filepath: str) -> str:
    """Extract plain text blocks from an HTML document."""
    from bs4 import BeautifulSoup, Comment

    raw_html = read_text_file(filepath)
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup.find_all(["style", "script"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    parser_result: list[dict] = []
    _read_text_recursively(soup.body or soup, parser_result)
    blocks, tables = _merge_blocks(parser_result)
    return join_sections([*blocks, *tables])

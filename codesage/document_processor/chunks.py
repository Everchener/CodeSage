from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .doc_extractors.base import normalize_text
from .text_splitter import split_text


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SLIDE_RE = re.compile(r"^第\s*(\d+)\s*页幻灯片$")


class DocumentChunkRecord(TypedDict):
    text: str
    search_text: str
    source: str
    chunk_index: int
    doc_type: str
    title: str
    section_path: str
    source_path: str
    file_ext: str
    page_or_slide: int
    tags: list[str]


def infer_doc_type(source_path: str = "", source: str = "") -> str:
    ext = Path(source_path or source).suffix.lower()
    mapping = {
        ".md": "markdown",
        ".html": "html",
        ".docx": "document",
        ".pdf": "pdf",
        ".txt": "text",
        ".pptx": "presentation",
        ".xlsx": "spreadsheet",
        ".xls": "spreadsheet",
        ".csv": "spreadsheet",
    }
    return mapping.get(ext, "document")


def serialize_tags(tags: list[str]) -> str:
    cleaned = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for tag in cleaned:
        if tag in seen:
            continue
        unique.append(tag)
        seen.add(tag)
    return f"|{'|'.join(unique)}|" if unique else ""


def deserialize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(tag).strip().lower() for tag in value if str(tag).strip()]
    text = str(value or "").strip("|")
    if not text:
        return []
    return [segment.strip().lower() for segment in text.split("|") if segment.strip()]


def build_document_filter_expr(filters: dict[str, Any] | None) -> str | None:
    if not filters:
        return None

    clauses: list[str] = []
    for key in ("doc_type", "source", "file_ext"):
        value = str(filters.get(key, "") or "").strip()
        if value:
            clauses.append(f'{key} == "{_escape_expr_value(value)}"')

    raw_tags = filters.get("tags")
    tag_values = raw_tags if isinstance(raw_tags, list) else [raw_tags] if raw_tags else []
    cleaned_tags = [str(tag).strip().lower() for tag in tag_values if str(tag).strip()]
    if cleaned_tags:
        tag_clauses = [f'tags like "%|{_escape_expr_value(tag)}|%"' for tag in cleaned_tags]
        clauses.append(f"({' or '.join(tag_clauses)})")

    return " and ".join(clauses) if clauses else None


def matches_document_filters(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True

    for key in ("doc_type", "source", "file_ext"):
        value = str(filters.get(key, "") or "").strip()
        if value and str(record.get(key, "") or "").strip() != value:
            return False

    raw_tags = filters.get("tags")
    tag_values = raw_tags if isinstance(raw_tags, list) else [raw_tags] if raw_tags else []
    cleaned_tags = [str(tag).strip().lower() for tag in tag_values if str(tag).strip()]
    if cleaned_tags:
        actual_tags = deserialize_tags(record.get("tags", []))
        if not any(tag in actual_tags for tag in cleaned_tags):
            return False

    return True


def build_document_chunk_records(
    text: str,
    *,
    source: str,
    source_path: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunkRecord]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    file_ext = Path(source_path or source).suffix.lower()
    doc_type = infer_doc_type(source_path=source_path, source=source)
    fallback_title = Path(source).stem or source or "untitled"
    section_stack: list[tuple[int, str]] = []
    title = fallback_title
    current_slide = 0
    blocks: list[dict[str, Any]] = []

    for paragraph in normalized.split("\n\n"):
        block = paragraph.strip()
        if not block:
            continue

        heading_match = HEADING_RE.match(block)
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            section_stack = [item for item in section_stack if item[0] < level]
            section_stack.append((level, heading))
            if level == 1 and title == fallback_title:
                title = heading
            slide_match = SLIDE_RE.match(heading)
            if slide_match:
                current_slide = int(slide_match.group(1))
            continue

        section_hint = _derive_section_hint(block)
        parts = [item[1] for item in section_stack]
        if section_hint:
            parts.append(section_hint)
        section_path = " > ".join(part for part in parts if part)
        block_chunks = split_text(block, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not block_chunks:
            block_chunks = [block]
        for chunk in block_chunks:
            tags = _derive_tags(doc_type=doc_type, file_ext=file_ext, section_path=section_path, page_or_slide=current_slide)
            blocks.append(
                {
                    "text": chunk,
                    "source": source,
                    "doc_type": doc_type,
                    "title": title,
                    "section_path": section_path,
                    "source_path": source_path,
                    "file_ext": file_ext,
                    "page_or_slide": current_slide,
                    "tags": tags,
                }
            )

    if not blocks:
        for chunk in split_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap) or [normalized]:
            blocks.append(
                {
                    "text": chunk,
                    "source": source,
                    "doc_type": doc_type,
                    "title": title,
                    "section_path": title if title != fallback_title else "",
                    "source_path": source_path,
                    "file_ext": file_ext,
                    "page_or_slide": current_slide,
                    "tags": _derive_tags(doc_type=doc_type, file_ext=file_ext, section_path="", page_or_slide=current_slide),
                }
            )

    records: list[DocumentChunkRecord] = []
    for index, block in enumerate(blocks):
        record: DocumentChunkRecord = {
            "text": str(block["text"]).strip(),
            "search_text": build_chunk_search_text(
                title=str(block["title"]).strip(),
                section_path=str(block["section_path"]).strip(),
                doc_type=str(block["doc_type"]).strip(),
                text=str(block["text"]).strip(),
            ),
            "source": str(block["source"]).strip(),
            "chunk_index": index,
            "doc_type": str(block["doc_type"]).strip(),
            "title": str(block["title"]).strip(),
            "section_path": str(block["section_path"]).strip(),
            "source_path": str(block["source_path"]).strip(),
            "file_ext": str(block["file_ext"]).strip(),
            "page_or_slide": int(block["page_or_slide"] or 0),
            "tags": list(block["tags"]),
        }
        records.append(record)
    return records


def build_chunk_search_text(*, title: str, section_path: str, doc_type: str, text: str) -> str:
    parts = []
    if title:
        parts.append(f"title: {title}")
    if section_path:
        parts.append(f"section: {section_path}")
    if doc_type:
        parts.append(f"doc_type: {doc_type}")
    parts.append(text)
    return "\n".join(part for part in parts if part.strip())


def _derive_section_hint(block: str) -> str:
    first_line = block.splitlines()[0].strip()
    if first_line.startswith(("表格：", "工作表：")):
        return first_line
    return ""


def _derive_tags(*, doc_type: str, file_ext: str, section_path: str, page_or_slide: int) -> list[str]:
    tags: list[str] = []
    if file_ext:
        tags.append(file_ext.lstrip(".").lower())
    if doc_type == "text":
        tags.append("plain-text")
    else:
        tags.append(doc_type)
    if section_path:
        tags.append("structured")
    if doc_type == "spreadsheet":
        tags.append("tabular")
    if doc_type == "presentation" or page_or_slide > 0:
        tags.append("slides")
    return deserialize_tags(serialize_tags(tags))


def _escape_expr_value(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')

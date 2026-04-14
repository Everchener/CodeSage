"""轻量级文档提取器的共享辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

TEXT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1")


def read_text_file(filepath: str, encodings: Sequence[str] = TEXT_ENCODINGS) -> str:
    """使用简短的回退编码链读取文本文件。"""
    path = Path(filepath)
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception:
            raise
    if last_error is not None:
        raise last_error
    return ""


def normalize_text(text: str) -> str:
    """统一换行符并压缩过多的空白行。"""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    normalized_lines: list[str] = []
    last_blank = False

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if line.strip():
            normalized_lines.append(line)
            last_blank = False
        elif normalized_lines and not last_blank:
            normalized_lines.append("")
            last_blank = True

    return "\n".join(normalized_lines).strip()


def join_sections(sections: Iterable[str]) -> str:
    """将非空分段拼接为规范化的纯文本内容。"""
    normalized_sections = [normalize_text(section) for section in sections if normalize_text(section)]
    return "\n\n".join(normalized_sections).strip()


def clean_cell_text(value: object) -> str:
    """将表格单元格内容转换为紧凑的文本形式。"""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)


def prefix_heading(text: str, level: int) -> str:
    """以接近 Markdown 的纯文本形式渲染标题。"""
    heading = clean_cell_text(text)
    if not heading:
        return ""
    safe_level = max(1, min(level, 6))
    return f"{'#' * safe_level} {heading}"


def table_rows_to_text(rows: Sequence[Sequence[object]], caption: str | None = None) -> str:
    """将表格行转换为便于阅读的文本表示。"""
    normalized_rows = [
        [clean_cell_text(cell) for cell in row]
        for row in rows
        if any(clean_cell_text(cell) for cell in row)
    ]

    if not normalized_rows:
        return ""

    lines: list[str] = []
    cleaned_caption = clean_cell_text(caption)
    if cleaned_caption:
        lines.append(f"表格：{cleaned_caption}")

    if len(normalized_rows) == 1:
        single_row = [cell for cell in normalized_rows[0] if cell]
        if single_row:
            lines.append(" | ".join(single_row))
        return "\n".join(lines).strip()

    header = normalized_rows[0]
    body = normalized_rows[1:]

    header_cells = [cell for cell in header if cell]
    if header_cells:
        lines.append("列：" + " | ".join(header_cells))

    for row in body:
        fields: list[str] = []
        for index, cell in enumerate(row):
            if not cell:
                continue
            title = header[index] if index < len(header) else ""
            fields.append(f"{title}: {cell}" if title else cell)
        if fields:
            lines.append("; ".join(fields))

    if len(lines) == (1 if cleaned_caption else 0) and header_cells:
        lines.append(" | ".join(header_cells))

    return "\n".join(lines).strip()


def html_table_to_text(html: str) -> str:
    """将 HTML 表格转换为便于阅读的文本行。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        return normalize_text(soup.get_text("\n"))

    caption = table.find("caption")
    rows = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    return table_rows_to_text(rows, caption.get_text(" ", strip=True) if caption else None)

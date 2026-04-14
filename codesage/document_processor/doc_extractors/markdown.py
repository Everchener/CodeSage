"""Markdown extraction helpers adapted from RAGFlow's element extraction flow."""

from __future__ import annotations

import re

from .base import html_table_to_text, join_sections, normalize_text, read_text_file


class MarkdownElementExtractor:
    """Extract semantically grouped markdown blocks."""

    def __init__(self, markdown_content: str):
        self.markdown_content = markdown_content
        self.lines = markdown_content.split("\n")

    @staticmethod
    def _extract_header(lines: list[str], start_pos: int) -> tuple[dict, int]:
        return (
            {
                "type": "header",
                "content": lines[start_pos],
                "start_line": start_pos,
                "end_line": start_pos,
            },
            start_pos + 1,
        )

    @staticmethod
    def _extract_code_block(lines: list[str], start_pos: int) -> tuple[dict, int]:
        end_pos = start_pos
        content_lines = [lines[start_pos]]
        for index in range(start_pos + 1, len(lines)):
            content_lines.append(lines[index])
            end_pos = index
            if lines[index].strip().startswith("```"):
                break
        return (
            {
                "type": "code_block",
                "content": "\n".join(content_lines),
                "start_line": start_pos,
                "end_line": end_pos,
            },
            end_pos + 1,
        )

    @staticmethod
    def _extract_list_block(lines: list[str], start_pos: int) -> tuple[dict, int]:
        end_pos = start_pos
        content_lines: list[str] = []
        index = start_pos

        while index < len(lines):
            line = lines[index]
            if (
                re.match(r"^\s*[-*+]\s+.*$", line)
                or re.match(r"^\s*\d+\.\s+.*$", line)
                or (index > start_pos and not line.strip())
                or (index > start_pos and re.match(r"^\s{2,}[-*+]\s+.*$", line))
                or (index > start_pos and re.match(r"^\s{2,}\d+\.\s+.*$", line))
                or (index > start_pos and re.match(r"^\s+\w+.*$", line))
            ):
                content_lines.append(line)
                end_pos = index
                index += 1
            else:
                break

        return (
            {
                "type": "list_block",
                "content": "\n".join(content_lines),
                "start_line": start_pos,
                "end_line": end_pos,
            },
            end_pos + 1,
        )

    @staticmethod
    def _extract_blockquote(lines: list[str], start_pos: int) -> tuple[dict, int]:
        end_pos = start_pos
        content_lines: list[str] = []
        index = start_pos

        while index < len(lines):
            line = lines[index]
            if line.strip().startswith(">") or (index > start_pos and not line.strip()):
                content_lines.append(line)
                end_pos = index
                index += 1
            else:
                break

        return (
            {
                "type": "blockquote",
                "content": "\n".join(content_lines),
                "start_line": start_pos,
                "end_line": end_pos,
            },
            end_pos + 1,
        )

    @staticmethod
    def _extract_text_block(lines: list[str], start_pos: int) -> tuple[dict, int]:
        end_pos = start_pos
        content_lines = [lines[start_pos]]
        index = start_pos + 1

        while index < len(lines):
            line = lines[index]
            if (
                re.match(r"^#{1,6}\s+.*$", line)
                or line.strip().startswith("```")
                or re.match(r"^\s*[-*+]\s+.*$", line)
                or re.match(r"^\s*\d+\.\s+.*$", line)
                or line.strip().startswith(">")
            ):
                break
            if not line.strip():
                if index + 1 < len(lines) and (
                    re.match(r"^#{1,6}\s+.*$", lines[index + 1])
                    or lines[index + 1].strip().startswith("```")
                    or re.match(r"^\s*[-*+]\s+.*$", lines[index + 1])
                    or re.match(r"^\s*\d+\.\s+.*$", lines[index + 1])
                    or lines[index + 1].strip().startswith(">")
                ):
                    break
            content_lines.append(line)
            end_pos = index
            index += 1

        return (
            {
                "type": "text_block",
                "content": "\n".join(content_lines),
                "start_line": start_pos,
                "end_line": end_pos,
            },
            end_pos + 1,
        )

    def extract_elements(self) -> list[str]:
        sections: list[str] = []
        index = 0

        while index < len(self.lines):
            line = self.lines[index]
            if re.match(r"^#{1,6}\s+.*$", line):
                block, index = self._extract_header(self.lines, index)
            elif line.strip().startswith("```"):
                block, index = self._extract_code_block(self.lines, index)
            elif re.match(r"^\s*[-*+]\s+.*$", line) or re.match(r"^\s*\d+\.\s+.*$", line):
                block, index = self._extract_list_block(self.lines, index)
            elif line.strip().startswith(">"):
                block, index = self._extract_blockquote(self.lines, index)
            elif line.strip():
                block, index = self._extract_text_block(self.lines, index)
            else:
                index += 1
                continue

            content = normalize_text(block["content"])
            if content:
                sections.append(content)

        return sections


MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")


def _markdown_table_to_text(block: str) -> str:
    try:
        from markdown import markdown

        html = markdown(block, extensions=["markdown.extensions.tables"])
        table_text = html_table_to_text(html)
        if table_text:
            return table_text
    except Exception:
        pass

    rows = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or MARKDOWN_TABLE_SEPARATOR.match(stripped):
            continue
        if "|" not in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if any(cells):
            rows.append(cells)

    from .base import table_rows_to_text

    return table_rows_to_text(rows)


def _convert_block(block: str) -> str:
    lowered = block.lower()
    if "<table" in lowered:
        return html_table_to_text(block)
    if "|" in block and len([line for line in block.splitlines() if "|" in line]) >= 2:
        return _markdown_table_to_text(block)
    return normalize_text(block)


def extract_text(filepath: str) -> str:
    """Extract structured plain text from markdown."""
    markdown_text = read_text_file(filepath)
    extractor = MarkdownElementExtractor(markdown_text)
    return join_sections(_convert_block(section) for section in extractor.extract_elements())

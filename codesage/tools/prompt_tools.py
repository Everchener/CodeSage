"""CodeSage Agent 提示词构造辅助工具。"""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence


def _clean_text(text: str) -> str:
    return dedent(text).strip()


def build_list(items: Sequence[str], *, ordered: bool = False) -> str:
    """将字符串序列格式化为项目列表。"""
    normalized = [item.strip() for item in items if item and item.strip()]
    if not normalized:
        return ""

    lines = []
    for index, item in enumerate(normalized, start=1):
        prefix = f"{index}. " if ordered else "- "
        lines.append(f"{prefix}{item}")
    return "\n".join(lines)


def build_examples(examples: Sequence[tuple[str, str]]) -> str:
    """将 few-shot 示例格式化为统一文本块。"""
    blocks = []
    for index, (input_text, output_text) in enumerate(examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"示例 {index}",
                    f"输入：{_clean_text(input_text)}",
                    f"输出：{_clean_text(output_text)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_section(title: str, body: str) -> str:
    """构造标准章节。"""
    cleaned = _clean_text(body)
    if not cleaned:
        return ""
    return f"## {title}\n{cleaned}"


def build_json_output(
    schema: str,
    *,
    empty_result: str | None = None,
    extra_rules: Sequence[str] | None = None,
) -> str:
    """构造 JSON 输出契约章节内容。"""
    parts = [
        "只返回合法 JSON，且不要输出额外解释、Markdown 或思考过程。",
        f"JSON 结构：\n```json\n{_clean_text(schema)}\n```",
    ]
    if empty_result:
        parts.append(f"空结果约定：{_clean_text(empty_result)}")
    if extra_rules:
        parts.append(f"补充要求：\n{build_list(list(extra_rules), ordered=False)}")
    return "\n\n".join(parts)


def build_prompt(
    *,
    task: str,
    context_sections: Sequence[tuple[str, str]] | None = None,
    rules: Sequence[str] | None = None,
    output_format: str | None = None,
    empty_result: str | None = None,
    examples: Sequence[tuple[str, str]] | None = None,
) -> str:
    """构造统一的任务提示词正文。"""
    sections = [build_section("任务目标", task)]

    for title, body in context_sections or []:
        sections.append(build_section(title, body))

    if rules:
        sections.append(build_section("约束规则", build_list(list(rules), ordered=True)))

    if examples:
        sections.append(build_section("示例", build_examples(list(examples))))

    output_parts = []
    if output_format:
        output_parts.append(_clean_text(output_format))
    if empty_result:
        output_parts.append(f"空结果约定：{_clean_text(empty_result)}")
    if output_parts:
        sections.append(build_section("输出格式", "\n\n".join(output_parts)))

    return "\n\n".join(section for section in sections if section)


def build_system_prompt(
    *,
    role: str,
    responsibilities: Sequence[str],
    rules: Sequence[str] | None = None,
    output_instruction: str | None = None,
) -> str:
    """构造统一的 system prompt。"""
    sections = [
        build_section("角色定义", role),
        build_section("任务目标", build_list(list(responsibilities), ordered=True)),
    ]
    if rules:
        sections.append(build_section("约束规则", build_list(list(rules), ordered=True)))
    if output_instruction:
        sections.append(build_section("输出要求", output_instruction))
    return "\n\n".join(section for section in sections if section)

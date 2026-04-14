from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import Any, TypedDict

from codesage.tools.llm_tools import call_llm
from codesage.tools.milvus_tools import search_codebase
from codesage.tools.prompt_tools import build_prompt, build_system_prompt

logger = logging.getLogger(__name__)

MAX_REVIEW_FILES = 5
MAX_QUERIES_PER_FILE = 3
MAX_HITS_PER_FILE = 4
MAX_GLOBAL_HITS = 8
MAX_SNIPPET_CHARS = 200
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IDENTIFIER_STOPWORDS = {
    "and",
    "as",
    "async",
    "await",
    "class",
    "def",
    "elif",
    "else",
    "except",
    "false",
    "for",
    "from",
    "if",
    "import",
    "in",
    "is",
    "none",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "self",
    "true",
    "try",
    "while",
    "with",
}

FILE_SUMMARY_SYSTEM = build_system_prompt(
    role="你是 PR 审查历史代码上下文助手，负责把单个变更文件的历史实现压缩成审查可用摘要。",
    responsibilities=[
        "说明历史代码的作用。",
        "指出它和当前变更的直接关系。",
        "给出审查时应关注的兼容性和行为风险。",
    ],
    rules=[
        "如果历史代码和当前变更关系有限，必须明确写出“关系有限”。",
        "不要复述大段代码，重点提炼审查价值。",
    ],
    output_instruction="直接输出 Markdown，不要输出额外解释。",
)

GLOBAL_SUMMARY_SYSTEM = build_system_prompt(
    role="你是 PR 审查 RAG 汇总助手，负责将多个文件级历史上下文压缩成简版全局参考。",
    responsibilities=[
        "整合多文件的历史语义线索。",
        "突出跨文件兼容性与行为约束。",
    ],
    rules=[
        "只保留对 PR 审查最有价值的信息。",
        "不要重复文件级细节。",
    ],
    output_instruction="直接输出简洁 Markdown，不要输出额外解释。",
)


class ReviewRAGResult(TypedDict):
    summary: str
    by_file: dict[str, str]
    hit_count: int
    hits_by_file: dict[str, list[dict[str, Any]]]
    fallback_used: bool
    file_fallback_count: int
    global_fallback_used: bool


class RAGAgent:
    """负责检索历史代码并生成结构化审查上下文。"""

    def __init__(self):
        self.name = "RAGAgent"

    def _added_lines(self, chunk: dict[str, Any]) -> list[str]:
        return [
            line[1:].strip()
            for line in str(chunk.get("lines", "")).splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]

    def _path_tokens(self, file_path: str) -> list[str]:
        path = PurePosixPath(file_path or "")
        tokens: list[str] = []
        for part in path.parts:
            lowered = part.strip().lower()
            if not lowered:
                continue
            tokens.append(lowered)
            stem = PurePosixPath(lowered).stem
            if stem and stem not in tokens:
                tokens.append(stem)

        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def _extract_identifiers(self, added_lines: list[str]) -> list[str]:
        identifiers: list[str] = []
        seen: set[str] = set()
        for line in added_lines:
            for match in IDENTIFIER_PATTERN.finditer(line):
                token = match.group(0).lower()
                if token in IDENTIFIER_STOPWORDS or token in seen:
                    continue
                seen.add(token)
                identifiers.append(token)
                if len(identifiers) >= 8:
                    return identifiers
        return identifiers

    def _semantic_query(self, file_path: str, added_lines: list[str], identifiers: list[str]) -> str:
        compact_lines: list[str] = []
        for line in added_lines:
            normalized = " ".join(line.split())
            if not normalized:
                continue
            compact_lines.append(normalized)
            if len(compact_lines) >= 3:
                break

        pieces = [PurePosixPath(file_path).name]
        if identifiers:
            pieces.append(" ".join(identifiers[:4]))
        if compact_lines:
            pieces.append(" | ".join(compact_lines))
        query = " ".join(piece for piece in pieces if piece).strip()
        return query[:220]

    def _build_queries(self, chunk: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
        file_path = str(chunk.get("file", "") or "")
        added_lines = self._added_lines(chunk)
        identifiers = self._extract_identifiers(added_lines)
        path_tokens = self._path_tokens(file_path)

        queries: list[str] = []
        path_query = " ".join(path_tokens[:4]).strip()
        if path_query:
            queries.append(path_query)
        if identifiers:
            queries.append(" ".join(identifiers[:6]))
        semantic_query = self._semantic_query(file_path, added_lines, identifiers)
        if semantic_query:
            queries.append(semantic_query)

        deduped_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = query.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped_queries.append(normalized)
            if len(deduped_queries) >= MAX_QUERIES_PER_FILE:
                break

        return deduped_queries, identifiers, path_tokens

    def _hit_key(self, hit: dict[str, Any]) -> tuple[str, str]:
        return (
            str(hit.get("file_path", "") or ""),
            str(hit.get("func_name", "") or ""),
        )

    def _compact_code(self, code: str, limit: int = MAX_SNIPPET_CHARS) -> str:
        compact = " ".join((code or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _rerank_hit(
        self,
        hit: dict[str, Any],
        *,
        identifiers: list[str],
        path_tokens: list[str],
        target_file: str,
    ) -> float:
        score = float(hit.get("score") or 0.0)
        file_path = str(hit.get("file_path", "") or "").lower()
        func_name = str(hit.get("func_name", "") or "").lower()
        code = str(hit.get("code", "") or "").lower()
        target_name = PurePosixPath(target_file).name.lower()

        bonus = 0.0
        if target_name and target_name in file_path:
            bonus += 1.0
        if func_name and func_name in identifiers:
            bonus += 1.6

        path_bonus = sum(0.18 for token in path_tokens if token and token in file_path)
        code_bonus = sum(0.12 for identifier in identifiers if identifier and identifier in code)
        bonus += min(path_bonus, 0.9)
        bonus += min(code_bonus, 0.6)
        return score + bonus

    def _search_file(self, chunk: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        file_path = str(chunk.get("file", "") or "")
        queries, identifiers, path_tokens = self._build_queries(chunk)
        merged_hits: dict[tuple[str, str], dict[str, Any]] = {}

        for query in queries:
            for hit in search_codebase(query, top_k=MAX_HITS_PER_FILE):
                key = self._hit_key(hit)
                payload = dict(hit)
                matched_queries = list(payload.get("matched_queries") or [])
                matched_queries.append(query)
                payload["matched_queries"] = matched_queries
                payload["rerank_score"] = self._rerank_hit(
                    payload,
                    identifiers=identifiers,
                    path_tokens=path_tokens,
                    target_file=file_path,
                )
                existing = merged_hits.get(key)
                if existing is None or float(payload["rerank_score"]) > float(existing.get("rerank_score") or 0.0):
                    merged_hits[key] = payload

        ranked = list(merged_hits.values())
        ranked.sort(
            key=lambda item: (
                float(item.get("rerank_score") or 0.0),
                float(item.get("score") or 0.0),
                str(item.get("file_path", "")),
                str(item.get("func_name", "")),
            ),
            reverse=True,
        )
        return file_path, ranked[:MAX_HITS_PER_FILE]

    def _fallback_file_summary(self, file_path: str, hits: list[dict[str, Any]]) -> str:
        lines = [
            "### 历史代码作用",
            "- 下面列出与当前文件最相关的历史实现命中。",
            "",
            "### 与当前变更关系",
            f"- 重点参考 `{file_path}` 相关实现，判断是否存在兼容性或行为约束。",
            "",
            "### 审查关注点",
        ]
        for hit in hits[:MAX_HITS_PER_FILE]:
            source = f"{hit.get('file_path', '')}::{hit.get('func_name', '')}".strip(":")
            snippet = self._compact_code(str(hit.get("code", "") or ""))
            lines.append(f"- `{source}`: {snippet}")
        return "\n".join(lines)

    def _summarize_file(self, file_path: str, hits: list[dict[str, Any]]) -> str:
        context_parts = []
        for hit in hits[:MAX_HITS_PER_FILE]:
            source = f"{hit.get('file_path', '')}::{hit.get('func_name', '')}".strip(":")
            context_parts.append(
                f"--- {source} ---\n{self._compact_code(str(hit.get('code', '') or ''), limit=400)}"
            )

        prompt = build_prompt(
            task="基于当前变更文件及命中的历史实现，生成适合 PR 审查使用的文件级上下文摘要。",
            context_sections=[
                ("当前变更文件", file_path),
                ("检索命中的历史实现", "\n\n".join(context_parts)),
            ],
            rules=[
                "只保留对审查最有价值的信息。",
                "不要逐段重复代码。",
                "如果关系有限，要明确说明。",
            ],
            output_format="""使用固定三段 Markdown 输出：
### 历史代码作用
- ...

### 与当前变更关系
- ...

### 审查关注点
- ...""",
        )
        summary = call_llm(prompt, system=FILE_SUMMARY_SYSTEM, max_tokens=500)
        return summary.strip() if summary else ""

    def _fallback_global_summary(self, by_file: dict[str, str]) -> str:
        lines = ["### 相关历史代码参考"]
        for file_path, summary in by_file.items():
            compact = " ".join(summary.split())
            if len(compact) > 180:
                compact = compact[:177].rstrip() + "..."
            lines.append(f"- `{file_path}`: {compact}")
        return "\n".join(lines)

    def _summarize_global(self, by_file: dict[str, str]) -> str:
        prompt = build_prompt(
            task="将多个文件级历史代码摘要压缩成一段 PR 审查可引用的全局简版参考。",
            context_sections=[
                ("文件级摘要", "\n\n".join(f"## {file}\n{summary}" for file, summary in by_file.items())),
            ],
            rules=[
                "聚焦跨文件兼容性、共享约束和行为风险。",
                "不要重复文件级细节。",
            ],
            output_format="""输出简短 Markdown：
### 相关历史代码参考
- ...
- ...""",
        )
        summary = call_llm(prompt, system=GLOBAL_SUMMARY_SYSTEM, max_tokens=400)
        return summary.strip() if summary else ""

    def run(self, diff_chunks: list[dict[str, Any]]) -> ReviewRAGResult:
        """执行检索并输出结构化上下文结果。"""
        py_chunks = [chunk for chunk in diff_chunks if str(chunk.get("file", "")).endswith(".py")]
        selected_chunks = py_chunks[:MAX_REVIEW_FILES]

        hits_by_file: dict[str, list[dict[str, Any]]] = {}
        by_file: dict[str, str] = {}
        global_hits: list[dict[str, Any]] = []
        file_fallback_count = 0

        for chunk in selected_chunks:
            file_path, hits = self._search_file(chunk)
            if not hits:
                continue
            hits_by_file[file_path] = hits
            global_hits.extend(hits)

            summary = self._summarize_file(file_path, hits)
            if not summary:
                logger.warning("PR RAG 文件级摘要生成失败，回退到紧凑证据列表: %s", file_path)
                summary = self._fallback_file_summary(file_path, hits)
                file_fallback_count += 1
            by_file[file_path] = summary

        if not hits_by_file:
            return {
                "summary": "",
                "by_file": {},
                "hit_count": 0,
                "hits_by_file": {},
                "fallback_used": False,
                "file_fallback_count": 0,
                "global_fallback_used": False,
            }

        unique_global_hits: dict[tuple[str, str], dict[str, Any]] = {}
        for hit in global_hits:
            key = self._hit_key(hit)
            existing = unique_global_hits.get(key)
            if existing is None or float(hit.get("rerank_score") or 0.0) > float(existing.get("rerank_score") or 0.0):
                unique_global_hits[key] = hit

        ranked_global_hits = sorted(
            unique_global_hits.values(),
            key=lambda item: float(item.get("rerank_score") or 0.0),
            reverse=True,
        )[:MAX_GLOBAL_HITS]

        global_summary = self._summarize_global(by_file)
        global_fallback_used = False
        if not global_summary:
            logger.warning("PR RAG 全局摘要生成失败，回退到文件级摘要压缩结果")
            global_summary = self._fallback_global_summary(by_file)
            global_fallback_used = True

        per_file_counts = {file_path: len(hits) for file_path, hits in hits_by_file.items()}
        logger.info(
            "PR RAG 完成: files=%s, hit_count=%s, per_file=%s, fallback=%s",
            len(hits_by_file),
            len(ranked_global_hits),
            per_file_counts,
            bool(file_fallback_count or global_fallback_used),
        )

        return {
            "summary": global_summary,
            "by_file": by_file,
            "hit_count": len(ranked_global_hits),
            "hits_by_file": hits_by_file,
            "fallback_used": bool(file_fallback_count or global_fallback_used),
            "file_fallback_count": file_fallback_count,
            "global_fallback_used": global_fallback_used,
        }

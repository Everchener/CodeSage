from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import threading
from typing import Any

from codesage.tools.milvus_tools import (
    hybrid_search_apidocs,
    search_codebase,
    search_documents,
)

DEFAULT_SOURCES = ("codebase", "documents", "apidocs")
DEFAULT_PER_SOURCE_TOP_K = 3
DEFAULT_TOTAL_TOP_K = 8
MAX_EXCERPT_CHARS = 400

_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="modify-rag")
_REGISTRY_LOCK = threading.Lock()
_PREFETCH_FUTURES: dict[str, Future["RetrievalBundle"]] = {}


@dataclass(frozen=True)
class RetrievalBundle:
    hits: list[dict[str, Any]]
    rendered_context: str
    sources_used: list[str]
    query_bundle: dict[str, Any]
    is_partial: bool


def _normalize_queries(query_bundle: dict[str, Any]) -> list[str]:
    values: list[str] = []

    raw_queries = query_bundle.get("queries")
    if isinstance(raw_queries, list):
        values.extend(str(item or "").strip() for item in raw_queries)

    for field_name in (
        "instruction",
        "analysis",
        "plan",
        "execution_feedback",
    ):
        value = str(query_bundle.get(field_name, "") or "").strip()
        if value:
            values.append(value)

    raw_hints = query_bundle.get("query_hints")
    if isinstance(raw_hints, list):
        values.extend(str(item or "").strip() for item in raw_hints)

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = " ".join(value.split())
        if not compact or compact in seen:
            continue
        seen.add(compact)
        normalized.append(compact)
    return normalized


def _normalize_sources(query_bundle: dict[str, Any]) -> list[str]:
    raw_sources = query_bundle.get("sources")
    if not isinstance(raw_sources, list):
        return list(DEFAULT_SOURCES)

    sources: list[str] = []
    for item in raw_sources:
        source = str(item or "").strip().lower()
        if source in DEFAULT_SOURCES and source not in sources:
            sources.append(source)
    return sources or list(DEFAULT_SOURCES)


def _build_source_label(hit: dict[str, Any], source_type: str) -> str:
    existing = str(hit.get("source_label", "") or "").strip()
    if existing:
        return existing

    if source_type == "codebase":
        file_path = str(hit.get("file_path", "")).strip()
        func_name = str(hit.get("func_name", "")).strip()
        if file_path and func_name:
            return f"{file_path}::{func_name}"
        return file_path or func_name or "unknown"

    if source_type == "documents":
        source = str(hit.get("source", "")).strip()
        chunk_index = hit.get("chunk_index", hit.get("chunk_idx", ""))
        if source and chunk_index not in ("", None):
            return f"{source}#{chunk_index}"
        return source or "unknown"

    title = str(hit.get("title", "")).strip()
    source = str(hit.get("source", "")).strip()
    if source and title:
        return f"{source}::{title}"
    return title or source or "unknown"


def _build_excerpt(hit: dict[str, Any], source_type: str) -> str:
    if source_type == "codebase":
        text = str(hit.get("code", "") or "")
    elif source_type == "documents":
        text = str(hit.get("text", "") or "")
    else:
        text = str(hit.get("content", "") or "")

    compact = " ".join(text.split())
    if len(compact) <= MAX_EXCERPT_CHARS:
        return compact
    return compact[: MAX_EXCERPT_CHARS - 3].rstrip() + "..."


def _annotate_hit(
    hit: dict[str, Any],
    *,
    source_type: str,
    query: str,
) -> dict[str, Any]:
    payload = dict(hit)
    payload["source_type"] = source_type
    payload["source_label"] = _build_source_label(payload, source_type)
    payload["excerpt"] = _build_excerpt(payload, source_type)
    payload["query"] = query
    payload["score"] = float(payload.get("score") or 0.0)
    return payload


def _source_handler(source_type: str):
    if source_type == "codebase":
        return search_codebase
    if source_type == "documents":
        return search_documents
    return hybrid_search_apidocs


def _retrieve_source(
    *,
    source_type: str,
    queries: list[str],
    per_source_top_k: int,
) -> list[dict[str, Any]]:
    handler = _source_handler(source_type)
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for query in queries:
        try:
            hits = handler(query, top_k=max(1, per_source_top_k))
        except Exception:
            continue

        for raw_hit in hits:
            annotated = _annotate_hit(raw_hit, source_type=source_type, query=query)
            key = (source_type, annotated["source_label"])
            existing = merged.get(key)
            if existing is None or annotated["score"] > float(existing.get("score") or 0.0):
                merged[key] = annotated

    ranked = list(merged.values())
    ranked.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            item.get("source_label", ""),
        ),
        reverse=True,
    )
    return ranked[: max(1, per_source_top_k)]


def _render_context(hits: list[dict[str, Any]], query_bundle: dict[str, Any], is_partial: bool) -> str:
    lines = ["RAG context package for modify", ""]

    queries = _normalize_queries(query_bundle)
    if queries:
        lines.append("Queries:")
        lines.extend(f"- {query}" for query in queries[:4])
        lines.append("")

    if is_partial:
        lines.append("Status:")
        lines.append("- Partial retrieval results were returned.")
        lines.append("")

    if not hits:
        lines.append("Evidence:")
        lines.append("- No retrieval hits.")
        return "\n".join(lines)

    lines.append("Evidence:")
    for hit in hits:
        lines.append(
            f"- [{hit['source_type']}] {hit['source_label']} (score={float(hit.get('score') or 0.0):.3f})"
        )
        excerpt = str(hit.get("excerpt", "") or "").strip()
        if excerpt:
            lines.append(f"  {excerpt}")
    return "\n".join(lines)


def parallel_retrieve(query_bundle: dict[str, Any]) -> RetrievalBundle:
    queries = _normalize_queries(query_bundle)
    sources = _normalize_sources(query_bundle)
    per_source_top_k = int(query_bundle.get("per_source_top_k") or DEFAULT_PER_SOURCE_TOP_K)
    total_top_k = int(query_bundle.get("total_top_k") or DEFAULT_TOTAL_TOP_K)

    if not queries:
        empty_context = _render_context([], query_bundle, False)
        return RetrievalBundle(
            hits=[],
            rendered_context=empty_context,
            sources_used=[],
            query_bundle=dict(query_bundle),
            is_partial=False,
        )

    futures = {
        source: _EXECUTOR.submit(
            _retrieve_source,
            source_type=source,
            queries=queries,
            per_source_top_k=per_source_top_k,
        )
        for source in sources
    }

    aggregated: list[dict[str, Any]] = []
    sources_used: list[str] = []
    is_partial = False

    for source, future in futures.items():
        try:
            hits = future.result()
        except Exception:
            is_partial = True
            continue
        if hits:
            aggregated.extend(hits)
            sources_used.append(source)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    aggregated.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            item.get("source_type", ""),
            item.get("source_label", ""),
        ),
        reverse=True,
    )
    for hit in aggregated:
        key = (str(hit.get("source_type", "")), str(hit.get("source_label", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
        if len(deduped) >= max(1, total_top_k):
            break

    context = _render_context(deduped, query_bundle, is_partial)
    return RetrievalBundle(
        hits=deduped,
        rendered_context=context,
        sources_used=sources_used,
        query_bundle=dict(query_bundle),
        is_partial=is_partial,
    )


def start_prefetch(run_id: str, query_bundle: dict[str, Any]) -> None:
    if not run_id:
        return

    with _REGISTRY_LOCK:
        existing = _PREFETCH_FUTURES.pop(run_id, None)
        if existing is not None:
            existing.cancel()
        _PREFETCH_FUTURES[run_id] = _EXECUTOR.submit(parallel_retrieve, dict(query_bundle))


def join_prefetch(run_id: str, timeout_s: float = 0) -> RetrievalBundle | None:
    if not run_id:
        return None

    with _REGISTRY_LOCK:
        future = _PREFETCH_FUTURES.get(run_id)

    if future is None:
        return None

    try:
        bundle = future.result(timeout=max(0.0, float(timeout_s or 0.0)))
    except TimeoutError:
        return None
    except Exception:
        with _REGISTRY_LOCK:
            _PREFETCH_FUTURES.pop(run_id, None)
        return None

    with _REGISTRY_LOCK:
        _PREFETCH_FUTURES.pop(run_id, None)
    return bundle


def cancel_prefetch(run_id: str) -> None:
    if not run_id:
        return

    with _REGISTRY_LOCK:
        future = _PREFETCH_FUTURES.pop(run_id, None)

    if future is not None:
        future.cancel()

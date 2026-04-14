from __future__ import annotations

import logging
import math
from typing import Any

from codesage.memory.fact_store import FactStore


logger = logging.getLogger(__name__)


def _tokenize(text: str) -> dict[str, float]:
    tokens: dict[str, float] = {}
    for token in (text or "").lower().split():
        normalized = token.strip()
        if not normalized:
            continue
        tokens[normalized] = tokens.get(normalized, 0.0) + 1.0
    return tokens


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, value in a.items():
        dot += value * b.get(key, 0.0)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LongTermRetriever:
    def __init__(self, *, fact_store: FactStore, long_store: Any):
        self.fact_store = fact_store
        self.long_store = long_store

    def retrieve(
        self,
        *,
        thread_id: str,
        project_id: str,
        query: str,
        policy: str,
        top_k: int,
    ) -> list[Any]:
        query_tokens = _tokenize(query)
        rows = self.fact_store.read_active_facts(thread_id=thread_id, project_id=project_id)
        local_items: list[Any] = []
        from codesage.memory.service import MemoryItem

        for row in rows:
            content = str(row.get("content", "") or "")
            lexical = _cosine_similarity(query_tokens, _tokenize(content))
            scope = str(row.get("scope", "") or "")
            scope_bonus = 0.3 if scope == "thread" else 0.15 if scope == "project" else 0.0
            score = lexical + scope_bonus + float(row.get("importance", 0.0) or 0.0) * 0.2
            local_items.append(
                MemoryItem(
                    memory_type=str(row.get("memory_type", "") or ""),
                    content=content,
                    summary=str(row.get("summary", "") or ""),
                    confidence=float(row.get("confidence", 0.0) or 0.0),
                    importance=float(row.get("importance", 0.0) or 0.0),
                    created_at=int(row.get("created_at", 0) or 0),
                    updated_at=int(row.get("updated_at", 0) or 0),
                    score=score,
                    item_id=str(row.get("memory_id", "") or row.get("id", "")),
                    scope=scope or "thread",
                    thread_id=str(row.get("thread_id", "") or ""),
                    project_id=str(row.get("project_id", "") or ""),
                    status=str(row.get("status", "active") or "active"),
                )
            )

        milvus_items: list[Any] = []
        try:
            milvus_items = list(
                self.long_store.retrieve(
                    thread_id=thread_id,
                    project_id=project_id,
                    query=query,
                    top_k=top_k,
                    policy=policy,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Long-term retriever fallback to file facts only: %s", exc)

        merged: dict[str, Any] = {}
        for item in [*local_items, *milvus_items]:
            if item.status and item.status != "active":
                continue
            key = str(item.item_id or f"{item.scope}:{item.memory_type}:{item.content}")
            existing = merged.get(key)
            if existing is None or float(item.score or 0.0) > float(existing.score or 0.0):
                merged[key] = item

        ranked = list(merged.values())
        ranked.sort(
            key=lambda item: (
                float(item.score or 0.0),
                float(item.importance or 0.0),
                float(item.confidence or 0.0),
                int(item.updated_at or 0),
            ),
            reverse=True,
        )
        return ranked[: max(1, int(top_k))]

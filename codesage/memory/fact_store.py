from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from codesage.core.error_handling import safe_json_loads, safe_read_text, safe_write_text
from codesage.core.runtime import REPO_ROOT


logger = logging.getLogger(__name__)

MEMORY_BASE_DIR = REPO_ROOT / ".codesage" / "memory"
FACTS_PATH = MEMORY_BASE_DIR / "facts.jsonl"


def _truncate_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_subject(content: str) -> str:
    tokens = re.sub(r"[^a-z0-9]+", " ", str(content or "").lower()).split()
    return " ".join(tokens[:12])


def _logical_memory_id(scope: str, memory_type: str, subject: str, project_id: str, thread_id: str) -> str:
    key = "|".join([scope, memory_type, subject, project_id, thread_id if scope == "thread" else ""])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


class FactStore:
    def __init__(self, *, path: Path | None = None):
        self.path = Path(path) if path is not None else FACTS_PATH

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        text = safe_read_text(
            self.path,
            fallback="",
            logger=logger,
            module=__name__,
            operation="read fact store",
        )
        if not text:
            return []
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = safe_json_loads(
                line,
                fallback=None,
                logger=logger,
                module=__name__,
                operation="parse fact row",
                target=str(self.path),
            )
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def read_active_facts(self, *, thread_id: str = "", project_id: str = "") -> list[dict[str, Any]]:
        rows = []
        for row in self.read_all():
            if str(row.get("status", "active") or "active") != "active":
                continue
            scope = str(row.get("scope", "") or "").strip()
            if scope == "thread" and thread_id and str(row.get("thread_id", "") or "") != thread_id:
                continue
            if project_id and str(row.get("project_id", "") or "") not in {"", project_id}:
                continue
            rows.append(row)
        return rows

    def upsert_facts(
        self,
        *,
        thread_id: str,
        project_id: str,
        items: list[dict[str, Any]],
        source_turn_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        rows = self.read_all()
        now = int(time.time())
        new_active: list[dict[str, Any]] = []

        for item in items:
            scope = str(item.get("scope", "thread") or "thread").strip().lower()
            if scope not in {"thread", "project", "user"}:
                scope = "thread"
            memory_type = str(item.get("memory_type", "")).strip().lower()
            content = _truncate_text(item.get("content", ""), 4096)
            if not memory_type or not content:
                continue
            subject = _normalize_subject(item.get("summary") or content)
            memory_id = str(item.get("memory_id", "") or _logical_memory_id(scope, memory_type, subject, project_id, thread_id))
            confidence = float(item.get("confidence", 0.0) or 0.0)
            importance = float(item.get("importance", 0.0) or 0.0)
            summary = _truncate_text(item.get("summary") or content, 220)
            matched_active = [
                row
                for row in rows
                if str(row.get("memory_id", "") or "") == memory_id
                and str(row.get("status", "active") or "active") == "active"
            ]
            for row in matched_active:
                row["status"] = "superseded"
                row["superseded_at"] = now

            new_row = {
                "id": uuid.uuid4().hex,
                "memory_id": memory_id,
                "scope": scope,
                "project_id": project_id,
                "thread_id": thread_id if scope == "thread" else str(item.get("thread_id", "") or ""),
                "memory_type": memory_type,
                "content": content,
                "summary": summary,
                "confidence": confidence,
                "importance": importance,
                "status": "active",
                "source_turn_id": int(source_turn_id or item.get("source_turn_id", 0) or 0),
                "updated_at": int(item.get("updated_at", now) or now),
                "created_at": int(item.get("created_at", now) or now),
                "logical_key": subject,
            }
            rows.append(new_row)
            new_active.append(new_row)

        self._write_all(rows)
        return new_active

    def _write_all(self, rows: list[dict[str, Any]]) -> bool:
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        if content:
            content += "\n"
        return safe_write_text(
            self.path,
            content,
            logger=logger,
            module=__name__,
            operation="write fact store",
        )

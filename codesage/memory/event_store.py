from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from codesage.core.error_handling import safe_append_text, safe_json_loads, safe_read_text
from codesage.core.runtime import REPO_ROOT


logger = logging.getLogger(__name__)

THREADS_BASE_DIR = REPO_ROOT / ".codesage" / "threads"
DEFAULT_RECENT_TURN_WINDOWS = (6, 4, 3, 2)


def _sanitize_thread_id(thread_id: str) -> str:
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(thread_id or "").strip())
    normalized = normalized.rstrip(" .")
    return normalized or "thread"


def _truncate_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


class EventStore:
    def __init__(self, *, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else THREADS_BASE_DIR

    def thread_dir(self, thread_id: str) -> Path:
        return self.base_dir / _sanitize_thread_id(thread_id)

    def path_for_thread(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "events.jsonl"

    def next_turn_id(self, thread_id: str) -> int:
        turn_id = 0
        for event in self.read_events(thread_id):
            try:
                turn_id = max(turn_id, int(event.get("turn_id", 0) or 0))
            except (TypeError, ValueError):
                continue
        return turn_id + 1

    def append_turn_events(
        self,
        *,
        thread_id: str,
        turn_id: int,
        events: list[dict[str, Any]],
        observed_at: float | int | None = None,
    ) -> list[dict[str, Any]]:
        if not str(thread_id or "").strip() or not events:
            return []

        path = self.path_for_thread(thread_id)
        timestamp = float(observed_at or time.time())
        persisted: list[dict[str, Any]] = []
        for index, raw in enumerate(events, start=1):
            payload = dict(raw)
            payload["thread_id"] = thread_id
            payload["turn_id"] = int(turn_id)
            payload.setdefault("event_id", f"{int(turn_id):06d}-{index:03d}")
            payload.setdefault("timestamp", timestamp)
            persisted.append(payload)
            safe_append_text(
                path,
                json.dumps(payload, ensure_ascii=False) + "\n",
                logger=logger,
                module=__name__,
                operation="append thread event",
            )
        return persisted

    def read_events(self, thread_id: str) -> list[dict[str, Any]]:
        path = self.path_for_thread(thread_id)
        if not path.exists():
            return []
        text = safe_read_text(
            path,
            fallback="",
            logger=logger,
            module=__name__,
            operation="read thread events",
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
                operation="parse thread event",
                target=str(path),
            )
            if isinstance(payload, dict):
                rows.append(payload)
        rows.sort(
            key=lambda item: (
                int(item.get("turn_id", 0) or 0),
                float(item.get("timestamp", 0.0) or 0.0),
                str(item.get("event_id", "")),
            )
        )
        return rows

    def read_recent_events(self, thread_id: str, *, turns: int = 6) -> list[dict[str, Any]]:
        events = self.read_events(thread_id)
        if not events:
            return []
        distinct_turns: list[int] = []
        seen: set[int] = set()
        for event in reversed(events):
            try:
                turn_id = int(event.get("turn_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if turn_id <= 0 or turn_id in seen:
                continue
            seen.add(turn_id)
            distinct_turns.append(turn_id)
            if len(distinct_turns) >= max(1, int(turns)):
                break
        kept_turns = set(distinct_turns)
        recent = [event for event in events if int(event.get("turn_id", 0) or 0) in kept_turns]
        recent.sort(
            key=lambda item: (
                int(item.get("turn_id", 0) or 0),
                float(item.get("timestamp", 0.0) or 0.0),
                str(item.get("event_id", "")),
            )
        )
        return recent

    def render_recent_context(
        self,
        thread_id: str,
        *,
        char_budget: int,
        turn_windows: tuple[int, ...] = DEFAULT_RECENT_TURN_WINDOWS,
    ) -> tuple[str, int]:
        if char_budget <= 0:
            return "", 0

        windows = tuple(int(item) for item in turn_windows if int(item) > 0) or (6,)
        best_content = ""
        best_from_turn = 0
        for turns in windows:
            events = self.read_recent_events(thread_id, turns=turns)
            if not events:
                continue
            rendered = self._render_events(events)
            if len(rendered) <= char_budget:
                return rendered, min(int(event.get("turn_id", 0) or 0) for event in events)
            best_content = rendered
            best_from_turn = min(int(event.get("turn_id", 0) or 0) for event in events)

        if not best_content:
            return "", 0
        return best_content[: max(0, char_budget - 3)].rstrip() + "...", best_from_turn

    def _render_events(self, events: list[dict[str, Any]]) -> str:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            grouped.setdefault(int(event.get("turn_id", 0) or 0), []).append(event)

        lines = ["# Recent Thread Events", ""]
        for turn_id in sorted(grouped):
            lines.append(f"## Turn {turn_id:03d}")
            for event in grouped[turn_id]:
                event_type = str(event.get("event_type", event.get("type", "")) or "").strip()
                content = _truncate_text(event.get("content", ""), 280)
                summary = _truncate_text(event.get("summary", ""), 200)
                status = str(event.get("status", "") or "").strip()
                tool = str(event.get("tool", "") or "").strip()
                route = str(event.get("route", "") or "").strip()
                if event_type == "user_message":
                    lines.append(f"- User: {content or summary or 'N/A'}")
                elif event_type == "assistant_final":
                    lines.append(f"- Assistant: {content or summary or 'N/A'}")
                elif event_type == "route_decision":
                    lines.append(f"- Route: {route or 'unknown'}")
                elif event_type == "tool_call":
                    lines.append(f"- Tool call: {tool or 'unknown'} | {summary or content or 'N/A'}")
                elif event_type == "tool_result":
                    lines.append(f"- Tool result: {tool or 'unknown'} | {summary or content or 'N/A'}")
                elif event_type == "status_change":
                    lines.append(f"- Status: {status or 'unknown'}")
                else:
                    lines.append(f"- {event_type or 'event'}: {summary or content or 'N/A'}")
            lines.append("")
        return "\n".join(lines).strip()

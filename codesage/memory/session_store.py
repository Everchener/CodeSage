from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from codesage.core.error_handling import safe_json_loads, safe_read_text, safe_write_text
from codesage.core.runtime import REPO_ROOT


logger = logging.getLogger(__name__)

THREADS_BASE_DIR = REPO_ROOT / ".codesage" / "threads"
SESSION_MARKER = "codesage:session_memory_v1"


def _sanitize_thread_id(thread_id: str) -> str:
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(thread_id or "").strip())
    normalized = normalized.rstrip(" .")
    return normalized or "thread"


def _truncate_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_list(values: Any, *, max_items: int = 8, max_chars: int = 220) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _truncate_text(item, max_chars)
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
        if len(normalized) >= max_items:
            break
    return normalized


def build_empty_session_state(thread_id: str = "") -> dict[str, Any]:
    return {
        "schema": "session_memory_v1",
        "thread_id": thread_id,
        "goal": "",
        "current_state": "",
        "constraints": [],
        "decisions": [],
        "files_modules": [],
        "open_loops": [],
        "errors_fixes": [],
        "worklog": [],
        "covers_until_turn": 0,
        "covers_until_event_id": "",
        "updated_at": "",
    }


class SessionStore:
    def __init__(self, *, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else THREADS_BASE_DIR

    def thread_dir(self, thread_id: str) -> Path:
        return self.base_dir / _sanitize_thread_id(thread_id)

    def markdown_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "session.md"

    def json_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "session.json"

    def read(self, thread_id: str) -> dict[str, Any]:
        if not str(thread_id or "").strip():
            return build_empty_session_state("")
        json_path = self.json_path(thread_id)
        if json_path.exists():
            text = safe_read_text(
                json_path,
                fallback="",
                logger=logger,
                module=__name__,
                operation="read session memory json",
            )
            payload = safe_json_loads(
                text,
                fallback=build_empty_session_state(thread_id),
                logger=logger,
                module=__name__,
                operation="parse session memory json",
                target=str(json_path),
            )
            if isinstance(payload, dict):
                return self._sanitize_state(thread_id, payload)

        md_path = self.markdown_path(thread_id)
        if not md_path.exists():
            return build_empty_session_state(thread_id)
        text = safe_read_text(
            md_path,
            fallback="",
            logger=logger,
            module=__name__,
            operation="read session memory markdown",
        )
        match = re.search(r"<!--\s*codesage:session_memory_v1\s*(.*?)\s*-->", text, flags=re.DOTALL)
        if not match:
            return build_empty_session_state(thread_id)
        payload = safe_json_loads(
            match.group(1),
            fallback=build_empty_session_state(thread_id),
            logger=logger,
            module=__name__,
            operation="parse session memory marker",
            target=str(md_path),
        )
        if isinstance(payload, dict):
            return self._sanitize_state(thread_id, payload)
        return build_empty_session_state(thread_id)

    def update_from_events(
        self,
        thread_id: str,
        recent_events: list[dict[str, Any]],
        previous_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._sanitize_state(thread_id, previous_state or build_empty_session_state(thread_id))
        if not recent_events:
            return state

        last_user = next(
            (
                _truncate_text(event.get("content", "") or event.get("summary", ""), 280)
                for event in reversed(recent_events)
                if str(event.get("event_type", event.get("type", "")) or "") == "user_message"
            ),
            state.get("goal", ""),
        )
        last_assistant = next(
            (
                _truncate_text(event.get("content", "") or event.get("summary", ""), 320)
                for event in reversed(recent_events)
                if str(event.get("event_type", event.get("type", "")) or "") == "assistant_final"
            ),
            state.get("current_state", ""),
        )
        last_status = next(
            (
                str(event.get("status", "") or "").strip()
                for event in reversed(recent_events)
                if str(event.get("event_type", event.get("type", "")) or "") == "status_change"
            ),
            "",
        )
        last_route = next(
            (
                str(event.get("route", "") or "").strip()
                for event in reversed(recent_events)
                if str(event.get("event_type", event.get("type", "")) or "") == "route_decision"
            ),
            "",
        )
        max_turn = max(int(event.get("turn_id", 0) or 0) for event in recent_events)
        max_event_id = str(recent_events[-1].get("event_id", "") or "")
        updated_at = str(recent_events[-1].get("timestamp", "") or "")

        worklog = list(state.get("worklog", []))
        worklog.append(
            _truncate_text(
                f"Turn {max_turn:03d}: route={last_route or 'unknown'} status={last_status or 'completed'} user={last_user or 'N/A'}",
                220,
            )
        )
        worklog = worklog[-12:]

        open_loops = list(state.get("open_loops", []))
        errors_fixes = list(state.get("errors_fixes", []))
        if last_status == "awaiting_confirmation":
            open_loops = ["Waiting for user confirmation on the latest change preview."]
        elif last_status == "error":
            errors_fixes.append(_truncate_text(last_assistant or "The latest turn ended with an error.", 220))
            errors_fixes = errors_fixes[-8:]
            open_loops = ["Investigate the latest error before continuing."]
        elif last_status == "timeout":
            open_loops = ["Retry the latest turn with a smaller scope."]
        elif last_status == "cancelled":
            open_loops = ["Wait for a new user request before continuing."]
        else:
            open_loops = []

        decisions = list(state.get("decisions", []))
        if last_route:
            decisions.append(_truncate_text(f"Current route is `{last_route}` for this thread.", 220))
        decisions = decisions[-8:]

        state.update(
            {
                "goal": last_user or state.get("goal", ""),
                "current_state": last_assistant or state.get("current_state", ""),
                "decisions": _normalize_list(decisions),
                "open_loops": _normalize_list(open_loops),
                "errors_fixes": _normalize_list(errors_fixes),
                "worklog": _normalize_list(worklog, max_items=12),
                "covers_until_turn": max(int(state.get("covers_until_turn", 0) or 0), max_turn),
                "covers_until_event_id": max_event_id or state.get("covers_until_event_id", ""),
                "updated_at": updated_at,
            }
        )
        self.write(thread_id, state)
        return state

    def write(self, thread_id: str, state: dict[str, Any]) -> bool:
        sanitized = self._sanitize_state(thread_id, state)
        json_ok = safe_write_text(
            self.json_path(thread_id),
            json.dumps(sanitized, ensure_ascii=False, indent=2),
            logger=logger,
            module=__name__,
            operation="write session memory json",
        )
        md_ok = safe_write_text(
            self.markdown_path(thread_id),
            self._render_document(sanitized),
            logger=logger,
            module=__name__,
            operation="write session memory markdown",
        )
        return bool(json_ok and md_ok)

    def render_prompt_context(self, thread_id: str, *, char_budget: int) -> tuple[str, int]:
        state = self.read(thread_id)
        if char_budget <= 0 or not any(
            [
                state.get("goal"),
                state.get("current_state"),
                state.get("constraints"),
                state.get("decisions"),
                state.get("open_loops"),
                state.get("worklog"),
            ]
        ):
            return "", int(state.get("covers_until_turn", 0) or 0)
        rendered = self._render_markdown(state)
        if len(rendered) > char_budget:
            rendered = rendered[: max(0, char_budget - 3)].rstrip() + "..."
        return rendered, int(state.get("covers_until_turn", 0) or 0)

    def _sanitize_state(self, thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = build_empty_session_state(thread_id)
        state["goal"] = _truncate_text(payload.get("goal", ""), 220)
        state["current_state"] = _truncate_text(payload.get("current_state", ""), 320)
        state["constraints"] = _normalize_list(payload.get("constraints", []))
        state["decisions"] = _normalize_list(payload.get("decisions", []))
        state["files_modules"] = _normalize_list(payload.get("files_modules", []))
        state["open_loops"] = _normalize_list(payload.get("open_loops", []))
        state["errors_fixes"] = _normalize_list(payload.get("errors_fixes", []))
        state["worklog"] = _normalize_list(payload.get("worklog", []), max_items=12)
        state["updated_at"] = _truncate_text(payload.get("updated_at", ""), 80)
        try:
            state["covers_until_turn"] = max(0, int(payload.get("covers_until_turn", 0) or 0))
        except (TypeError, ValueError):
            state["covers_until_turn"] = 0
        state["covers_until_event_id"] = _truncate_text(payload.get("covers_until_event_id", ""), 80)
        return state

    def _render_document(self, state: dict[str, Any]) -> str:
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        return f"<!-- {SESSION_MARKER}\n{state_json}\n-->\n\n{self._render_markdown(state)}\n"

    def _render_markdown(self, state: dict[str, Any]) -> str:
        sections = [
            "# Session Memory",
            "",
            f"- covers_until_turn: {state.get('covers_until_turn', 0)}",
            f"- covers_until_event_id: {state.get('covers_until_event_id', '') or 'N/A'}",
            "",
            "## Goal",
            state.get("goal", "") or "N/A",
            "",
            "## Current State",
            state.get("current_state", "") or "N/A",
            "",
            "## Constraints",
            *self._render_list(state.get("constraints", [])),
            "",
            "## Decisions",
            *self._render_list(state.get("decisions", [])),
            "",
            "## Files/Modules",
            *self._render_list(state.get("files_modules", [])),
            "",
            "## Open Loops",
            *self._render_list(state.get("open_loops", [])),
            "",
            "## Errors & Fixes",
            *self._render_list(state.get("errors_fixes", [])),
            "",
            "## Worklog",
            *self._render_list(state.get("worklog", [])),
        ]
        return "\n".join(sections).strip()

    @staticmethod
    def _render_list(values: list[str]) -> list[str]:
        if not values:
            return ["- N/A"]
        return [f"- {item}" for item in values]

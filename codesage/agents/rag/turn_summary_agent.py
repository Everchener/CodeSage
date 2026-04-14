from __future__ import annotations

import json
import time
from typing import Any


SNAPSHOT_SCHEMA = "thread_snapshot_v1"
MAX_PROMPT_TEXT_CHARS = 1800
MAX_TURN_FIELD_CHARS = 220
MAX_STATUS_LIST_ITEMS = 6
MAX_STATUS_ITEM_CHARS = 160

SUMMARY_SYSTEM_PROMPT = (
    "You are CodeSage's thread snapshot summarizer.\n"
    "Update the full thread snapshot JSON using the previous snapshot and the current turn.\n"
    "Keep only stable facts, current state, and recent turns.\n"
    "Return valid JSON only."
)


def _normalize_status(value: str | None) -> str:
    status = str(value or "").strip().lower()
    if not status:
        return "completed"
    if status in {"success", "completed"}:
        return "completed"
    if status == "timeout":
        return "timeout"
    return status


def _truncate_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _truncate_text(item, MAX_STATUS_ITEM_CHARS)
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
        if len(normalized) >= MAX_STATUS_LIST_ITEMS:
            break
    return normalized


def _format_timestamp(observed_at: float | int | None) -> str:
    timestamp = float(observed_at or time.time())
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


class TurnSummaryAgent:
    def summarize_turn(
        self,
        *,
        previous_state: dict[str, Any] | None,
        thread_id: str,
        user_input: str,
        assistant_output: str,
        route: str,
        agent: str,
        status: str,
        agent_payload: dict[str, Any] | None = None,
        observed_at: float | int | None = None,
    ) -> dict[str, Any]:
        fallback = self.build_fallback_state(
            previous_state=previous_state or {},
            thread_id=thread_id,
            user_input=user_input,
            assistant_output=assistant_output,
            route=route,
            agent=agent,
            status=status,
            observed_at=observed_at,
        )
        prompt = self._build_prompt(
            previous_state=previous_state or {},
            thread_id=thread_id,
            user_input=user_input,
            assistant_output=assistant_output,
            route=route,
            agent=agent,
            status=status,
            agent_payload=agent_payload or {},
            observed_at=observed_at,
        )
        try:
            from codesage.tools.llm_tools import call_llm_json
        except ModuleNotFoundError:
            return fallback

        payload = call_llm_json(prompt, system=SUMMARY_SYSTEM_PROMPT, max_tokens=1200)
        if not isinstance(payload, dict):
            return fallback
        return self._merge_with_fallback(payload, fallback)

    def build_fallback_state(
        self,
        *,
        previous_state: dict[str, Any],
        thread_id: str,
        user_input: str,
        assistant_output: str,
        route: str,
        agent: str,
        status: str,
        observed_at: float | int | None = None,
    ) -> dict[str, Any]:
        normalized_status = _normalize_status(status)
        timestamp = _format_timestamp(observed_at)
        recent_turns = list(previous_state.get("recent_turns", []) or [])
        next_turn_no = 1
        if recent_turns:
            try:
                next_turn_no = max(int(item.get("turn_no", 0) or 0) for item in recent_turns) + 1
            except (TypeError, ValueError):
                next_turn_no = len(recent_turns) + 1

        recent_turns.append(
            {
                "turn_no": next_turn_no,
                "time": timestamp,
                "route": str(route or previous_state.get("current_route", "") or "").strip(),
                "agent": str(agent or previous_state.get("current_agent", "") or "").strip(),
                "status": normalized_status,
                "user_intent": _truncate_text(user_input, MAX_STATUS_ITEM_CHARS),
                "summary": _truncate_text(assistant_output, MAX_TURN_FIELD_CHARS),
                "outcome": self._build_outcome(normalized_status, assistant_output),
                "next_action": self._build_next_action(normalized_status),
            }
        )
        return {
            "schema": SNAPSHOT_SCHEMA,
            "thread_id": thread_id,
            "topic": _truncate_text(previous_state.get("topic") or user_input, MAX_STATUS_ITEM_CHARS),
            "user_goal": _truncate_text(
                previous_state.get("user_goal") or user_input,
                MAX_STATUS_ITEM_CHARS,
            ),
            "current_route": str(route or previous_state.get("current_route", "") or "").strip(),
            "current_agent": str(agent or previous_state.get("current_agent", "") or "").strip(),
            "current_status": normalized_status,
            "updated_at": timestamp,
            "now": self._build_now(
                user_input=user_input,
                route=route,
                agent=agent,
                status=normalized_status,
            ),
            "stable_memory": _normalize_list(previous_state.get("stable_memory", [])),
            "open_loops": self._build_open_loops(
                status=normalized_status,
                previous_state=previous_state,
            ),
            "history_digest": _truncate_text(previous_state.get("history_digest", ""), 1200),
            "recent_turns": recent_turns,
        }

    def _build_prompt(
        self,
        *,
        previous_state: dict[str, Any],
        thread_id: str,
        user_input: str,
        assistant_output: str,
        route: str,
        agent: str,
        status: str,
        agent_payload: dict[str, Any],
        observed_at: float | int | None,
    ) -> str:
        turn_payload = {
            "thread_id": thread_id,
            "time": _format_timestamp(observed_at),
            "route": str(route or "").strip(),
            "agent": str(agent or "").strip(),
            "status": _normalize_status(status),
            "user_input": _truncate_text(user_input, MAX_PROMPT_TEXT_CHARS),
            "assistant_output": _truncate_text(assistant_output, MAX_PROMPT_TEXT_CHARS),
            "agent_payload": agent_payload,
        }
        return (
            "Return the next complete thread snapshot JSON.\n"
            "Required top-level fields: schema, thread_id, topic, user_goal, current_route, "
            "current_agent, current_status, updated_at, now, stable_memory, open_loops, "
            "history_digest, recent_turns.\n"
            "Each recent_turn item must include turn_no, time, route, agent, status, user_intent, "
            "summary, outcome, next_action.\n"
            "Keep it concise and factual.\n\n"
            f"Previous snapshot:\n{json.dumps(previous_state, ensure_ascii=False, indent=2)}\n\n"
            f"Current turn:\n{json.dumps(turn_payload, ensure_ascii=False, indent=2)}"
        )

    def _merge_with_fallback(
        self,
        payload: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(fallback)
        for key in (
            "topic",
            "user_goal",
            "current_route",
            "current_agent",
            "current_status",
            "updated_at",
            "history_digest",
        ):
            if payload.get(key) not in ("", None):
                merged[key] = payload[key]

        for key in ("now", "stable_memory", "open_loops", "recent_turns"):
            if isinstance(payload.get(key), list):
                merged[key] = payload[key]

        merged["schema"] = SNAPSHOT_SCHEMA
        merged["thread_id"] = fallback["thread_id"]
        return merged

    def _build_now(
        self,
        *,
        user_input: str,
        route: str,
        agent: str,
        status: str,
    ) -> list[str]:
        return _normalize_list(
            [
                f"Current user intent: {_truncate_text(user_input, MAX_STATUS_ITEM_CHARS)}",
                f"Route: {route or 'unknown'}; agent: {agent or 'unknown'}",
                self._build_status_summary(status),
            ]
        )

    def _build_open_loops(
        self,
        *,
        status: str,
        previous_state: dict[str, Any],
    ) -> list[str]:
        if status == "awaiting_confirmation":
            return ["Wait for the user to approve or reject the current preview."]
        if status == "error":
            return ["The latest turn ended with an error and may need a retry."]
        if status == "timeout":
            return ["The latest turn timed out and may need a retry with a smaller scope."]
        if status == "cancelled":
            return ["The latest turn was cancelled and needs a new user request to continue."]
        return _normalize_list(previous_state.get("open_loops", []))

    @staticmethod
    def _build_outcome(status: str, assistant_output: str) -> str:
        if status == "awaiting_confirmation":
            return "A preview was produced and is waiting for user confirmation."
        if status == "error":
            return "The turn failed."
        if status == "timeout":
            return "The turn timed out."
        if status == "cancelled":
            return "The turn was cancelled."
        return _truncate_text(assistant_output, MAX_TURN_FIELD_CHARS)

    @staticmethod
    def _build_next_action(status: str) -> str:
        if status == "awaiting_confirmation":
            return "Wait for user confirmation or rejection."
        if status == "error":
            return "Retry after inspecting the error or after receiving more context."
        if status == "timeout":
            return "Retry with a smaller scope or after reducing context."
        if status == "cancelled":
            return "Wait for a new user request."
        return "Continue from the current thread goal."

    @staticmethod
    def _build_status_summary(status: str) -> str:
        if status == "awaiting_confirmation":
            return "The thread is currently waiting for confirmation."
        if status == "error":
            return "The latest turn ended with an error."
        if status == "timeout":
            return "The latest turn ended because of a timeout."
        if status == "cancelled":
            return "The latest turn was cancelled."
        return "The latest turn completed."

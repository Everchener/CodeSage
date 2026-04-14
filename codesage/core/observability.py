from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from codesage.core.error_handling import safe_append_text, safe_coerce_float, safe_json_loads, safe_read_text
from codesage.core.runtime import (
    REPO_ROOT,
    RUNS_BASE_DIR,
    append_run_event,
    list_run_artifacts,
    read_run_state,
    run_events_path,
    run_state_path,
    run_tool_calls_path,
    write_run_state,
)


ACTIVE_RUN_STATUSES = {"running", "streaming", "cancelling", "awaiting_confirmation"}
WORKFLOW_LOG_PATH = REPO_ROOT / ".codesage" / "workflow.log"
MAX_WORKFLOW_LOG_LINE_CHARS = 1000

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    text = safe_read_text(
        path,
        fallback="",
        logger=logger,
        module=__name__,
        operation="read observed jsonl",
    )
    if not text:
        return rows
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = safe_json_loads(
            line,
            fallback=None,
            logger=logger,
            module=__name__,
            operation="parse observed jsonl line",
            target=str(path),
        )
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
    return rows


def _normalize_status(value: str | None) -> str:
    status = str(value or "").strip().lower()
    if not status:
        return "unknown"
    if status == "success":
        return "completed"
    if status == "timeout":
        return "timed_out"
    return status


def _sanitize_log_line(value: Any, *, max_chars: int = MAX_WORKFLOW_LOG_LINE_CHARS) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _append_workflow_log(run_id: str, line: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    text = f"[{timestamp}] {_sanitize_log_line(line)}\n"

    safe_append_text(
        WORKFLOW_LOG_PATH,
        text,
        logger=logger,
        module=__name__,
        operation="append global workflow log",
    )

    run_log_path = RUNS_BASE_DIR / str(run_id) / "workflow.log"
    safe_append_text(
        run_log_path,
        text,
        logger=logger,
        module=__name__,
        operation="append run workflow log",
    )


def write_workflow_log(run_id: str, line: str) -> None:
    if not str(run_id or "").strip():
        return
    _append_workflow_log(str(run_id).strip(), line)


def _run_log_path(run_id: str) -> str:
    return str(RUNS_BASE_DIR / str(run_id) / "workflow.log")


def _state_status_from_event(event: dict[str, Any], current_status: str) -> str:
    event_type = str(event.get("type", "")).strip().lower()
    raw_status = _normalize_status(str(event.get("status", "")).strip())

    if event_type == "confirmation_required":
        return "awaiting_confirmation"
    if event_type == "done":
        if raw_status in {"completed", "awaiting_confirmation", "cancelled", "timed_out", "error"}:
            return raw_status
        return "completed"
    if event_type == "error":
        if raw_status in {"cancelled", "timed_out"}:
            return raw_status
        return "error"
    if raw_status in {"running", "streaming", "skipped"}:
        return "running"
    if raw_status in {"completed", "awaiting_confirmation", "cancelled", "timed_out", "error"}:
        return raw_status
    return current_status or "running"


def _build_run_summary(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "thread_id": str(state.get("thread_id", "") or ""),
        "route": str(state.get("route", "") or ""),
        "mode": str(state.get("mode", "") or ""),
        "agent": str(state.get("agent", "") or ""),
        "task_id": str(state.get("task_id", "") or ""),
        "task_type": str(state.get("task_type", "") or ""),
        "child_agent_mode": str(state.get("child_agent_mode", "") or ""),
        "fork_reason": str(state.get("fork_reason", "") or ""),
        "skill_name": str(state.get("skill_name", "") or ""),
        "skill_source": str(state.get("skill_source", "") or ""),
        "selection_mode": str(state.get("selection_mode", "") or ""),
        "status": _normalize_status(
            str(state.get("status", "") or state.get("final_status", "") or "")
        ),
        "current_stage": str(state.get("current_stage", "") or ""),
        "last_summary": str(state.get("last_summary", "") or state.get("summary", "") or ""),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "terminal_status": str(state.get("terminal_status", "") or ""),
        "source": str(state.get("source", "") or ""),
        "event_count": int(state.get("event_count", 0) or 0),
        "related_run_ids": list(state.get("related_run_ids", []) or []),
        "parent_run_id": str(state.get("parent_run_id", "") or ""),
        "child_run_ids": list(state.get("child_run_ids", []) or []),
        "trigger_stage": str(state.get("trigger_stage", "") or ""),
        "workflow_log_path": _run_log_path(run_id),
    }


def _load_run_state(run_id: str) -> dict[str, Any]:
    try:
        return read_run_state(run_id)
    except FileNotFoundError:
        return {}


def _ensure_run_exists(run_id: str) -> None:
    state = _load_run_state(run_id)
    if state:
        return
    if run_state_path(run_id).exists():
        return
    if run_events_path(run_id).exists() or run_tool_calls_path(run_id).exists():
        return
    raise FileNotFoundError(f"Run `{run_id}` does not exist.")


def start_observed_run(
    run_id: str,
    *,
    source: str,
    thread_id: str = "",
    route: str = "",
    mode: str = "",
    agent: str = "",
    summary: str = "",
    parent_run_id: str = "",
    trigger_stage: str = "",
) -> dict[str, Any]:
    now = time.time()
    state = _load_run_state(run_id)
    state.update(
        {
            "run_id": run_id,
            "source": source,
            "thread_id": thread_id or str(state.get("thread_id", "") or ""),
            "route": route or str(state.get("route", "") or ""),
            "mode": mode or str(state.get("mode", "") or ""),
            "agent": agent or str(state.get("agent", "") or ""),
            "status": _normalize_status(str(state.get("status", "") or "running")),
            "current_stage": str(state.get("current_stage", "") or "accepted"),
            "last_summary": summary or str(state.get("last_summary", "") or ""),
            "created_at": state.get("created_at", now),
            "updated_at": now,
            "event_count": int(state.get("event_count", 0) or 0),
            "related_run_ids": list(state.get("related_run_ids", []) or []),
            "parent_run_id": parent_run_id or str(state.get("parent_run_id", "") or ""),
            "child_run_ids": list(state.get("child_run_ids", []) or []),
            "trigger_stage": trigger_stage or str(state.get("trigger_stage", "") or ""),
            "workflow_log_path": _run_log_path(run_id),
        }
    )
    write_run_state(run_id, state)
    _append_workflow_log(
        run_id,
        f"run_started run_id={run_id} source={source or '-'} route={route or '-'} "
        f"agent={agent or '-'} parent={parent_run_id or '-'} summary={summary or '-'}",
    )
    return state


def link_observed_runs(
    parent_run_id: str,
    child_run_id: str,
    *,
    trigger_stage: str = "",
    summary: str = "",
) -> None:
    parent_id = str(parent_run_id or "").strip()
    child_id = str(child_run_id or "").strip()
    if not parent_id or not child_id or parent_id == child_id:
        return

    now = time.time()
    parent_state = _load_run_state(parent_id)
    child_state = _load_run_state(child_id)

    parent_children = {
        str(item).strip()
        for item in list(parent_state.get("child_run_ids", []) or [])
        if str(item).strip()
    }
    parent_related = {
        str(item).strip()
        for item in list(parent_state.get("related_run_ids", []) or [])
        if str(item).strip()
    }
    child_related = {
        str(item).strip()
        for item in list(child_state.get("related_run_ids", []) or [])
        if str(item).strip()
    }

    parent_children.add(child_id)
    parent_related.add(child_id)
    child_related.add(parent_id)

    if parent_state:
        parent_state.update(
            {
                "run_id": parent_id,
                "updated_at": now,
                "child_run_ids": sorted(parent_children),
                "related_run_ids": sorted(parent_related),
            }
        )
        write_run_state(parent_id, parent_state)

    child_state.update(
        {
            "run_id": child_id,
            "parent_run_id": parent_id,
            "trigger_stage": trigger_stage or str(child_state.get("trigger_stage", "") or ""),
            "updated_at": now,
            "related_run_ids": sorted(child_related),
            "workflow_log_path": _run_log_path(child_id),
        }
    )
    write_run_state(child_id, child_state)
    _append_workflow_log(
        parent_id,
        f"child_linked parent_run_id={parent_id} child_run_id={child_id} "
        f"trigger_stage={trigger_stage or '-'} summary={summary or '-'}",
    )
    _append_workflow_log(
        child_id,
        f"parent_linked child_run_id={child_id} parent_run_id={parent_id} "
        f"trigger_stage={trigger_stage or '-'} summary={summary or '-'}",
    )


def record_observed_event(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = _load_run_state(run_id)
    now = time.time()
    event = dict(payload)
    event["run_id"] = run_id
    event.setdefault("observed_at", now)
    append_run_event(run_id, event)

    current_status = _normalize_status(str(state.get("status", "") or "running"))
    next_status = _state_status_from_event(event, current_status)
    related_run_ids = {
        str(item).strip()
        for item in list(state.get("related_run_ids", []) or [])
        if str(item).strip()
    }
    source_run_id = str(event.get("source_run_id", "") or "").strip()
    child_run_id = str(event.get("child_run_id", "") or "").strip()
    if source_run_id and source_run_id != run_id:
        related_run_ids.add(source_run_id)
        link_observed_runs(
            run_id,
            source_run_id,
            trigger_stage=str(event.get("stage", "") or ""),
            summary=str(event.get("summary", "") or ""),
        )
    if child_run_id and child_run_id != run_id:
        related_run_ids.add(child_run_id)

    state.update(
        {
            "run_id": run_id,
            "thread_id": str(event.get("thread_id", "") or state.get("thread_id", "") or ""),
            "route": str(event.get("route", "") or state.get("route", "") or ""),
            "mode": str(event.get("mode", "") or state.get("mode", "") or ""),
            "agent": str(event.get("agent", "") or state.get("agent", "") or ""),
            "status": next_status,
            "current_stage": str(event.get("stage", "") or state.get("current_stage", "") or ""),
            "last_summary": str(event.get("summary", "") or state.get("last_summary", "") or ""),
            "updated_at": event["observed_at"],
            "event_count": int(state.get("event_count", 0) or 0) + 1,
            "related_run_ids": sorted(related_run_ids),
        }
    )

    if "schema" in event:
        state["schema"] = event["schema"]
    if "content" in event and str(event.get("content", "")).strip():
        state["last_content"] = str(event["content"])
    if "context_layers" in event and event["context_layers"] is not None:
        state["context_layers"] = event["context_layers"]
    if "context_chars" in event and event["context_chars"] is not None:
        state["context_chars"] = event["context_chars"]
    if "skill_enabled" in event:
        state["skill_enabled"] = bool(event["skill_enabled"])
    if "skill_name" in event:
        state["skill_name"] = str(event.get("skill_name", "") or "")
    if "skill_source" in event:
        state["skill_source"] = str(event.get("skill_source", "") or "")
    if "skill_mode" in event:
        state["skill_mode"] = str(event.get("skill_mode", "") or "")
    if "source_run_id" in event and source_run_id:
        state["last_source_run_id"] = source_run_id
    if "parent_run_id" in event:
        state["parent_run_id"] = str(event.get("parent_run_id", "") or "")
    if "task_id" in event:
        state["task_id"] = str(event.get("task_id", "") or "")
    if "task_type" in event:
        state["task_type"] = str(event.get("task_type", "") or "")
    if "child_agent_mode" in event:
        state["child_agent_mode"] = str(event.get("child_agent_mode", "") or "")
    if "fork_reason" in event:
        state["fork_reason"] = str(event.get("fork_reason", "") or "")
    if "skill_name" in event:
        state["skill_name"] = str(event.get("skill_name", "") or "")
    if "skill_source" in event:
        state["skill_source"] = str(event.get("skill_source", "") or "")
    if "selection_mode" in event:
        state["selection_mode"] = str(event.get("selection_mode", "") or "")
    if "child_run_ids" in event:
        state["child_run_ids"] = list(event.get("child_run_ids", []) or [])
    elif child_run_id:
        child_run_ids = {
            str(item).strip()
            for item in list(state.get("child_run_ids", []) or [])
            if str(item).strip()
        }
        child_run_ids.add(child_run_id)
        state["child_run_ids"] = sorted(child_run_ids)
    if "preview_id" in event:
        state["preview_id"] = str(event.get("preview_id", "") or "")
    if "pending_changes" in event:
        state["pending_changes"] = list(event.get("pending_changes", []) or [])
    if "risk_reasons" in event:
        state["risk_reasons"] = list(event.get("risk_reasons", []) or [])
    if "diff_summary" in event:
        state["diff_summary"] = str(event.get("diff_summary", "") or "")

    state.setdefault("source", "chat")
    state.setdefault("created_at", now)
    state.setdefault("workflow_log_path", _run_log_path(run_id))
    write_run_state(run_id, state)
    _append_workflow_log(
        run_id,
        f"event type={event.get('type', '-') or '-'} stage={event.get('stage', '-') or '-'} "
        f"status={_normalize_status(str(event.get('status', '') or '')) or '-'} "
        f"route={event.get('route', '-') or '-'} agent={event.get('agent', '-') or '-'} "
        f"tool={event.get('tool', '-') or '-'} source_run_id={source_run_id or '-'} "
        f"child_run_id={child_run_id or '-'} "
        f"summary={str(event.get('summary', '') or '-').replace(chr(10), ' ')}",
    )
    return event


def read_observed_events(run_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
    _ensure_run_exists(run_id)
    rows = _read_jsonl(run_events_path(run_id))
    if limit > 0:
        return rows[-limit:]
    return rows


def read_observed_tool_calls(run_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
    _ensure_run_exists(run_id)
    rows = _read_jsonl(run_tool_calls_path(run_id))
    if limit > 0:
        return rows[-limit:]
    return rows


def list_observed_runs(
    *,
    limit: int = 20,
    status: str | None = None,
    route: str | None = None,
) -> list[dict[str, Any]]:
    if not RUNS_BASE_DIR.exists():
        return []

    target_status = _normalize_status(status) if status else ""
    target_route = str(route or "").strip().lower()
    summaries: list[dict[str, Any]] = []

    for run_root in RUNS_BASE_DIR.iterdir():
        if not run_root.is_dir():
            continue
        run_id = run_root.name
        state = _load_run_state(run_id)
        if not state:
            continue
        summary = _build_run_summary(state, run_id)
        if target_status and summary["status"] != target_status:
            continue
        if target_route and summary["route"].strip().lower() != target_route:
            continue
        if not summary["event_count"]:
            summary["event_count"] = len(_read_jsonl(run_events_path(run_id)))
        summaries.append(summary)

    summaries.sort(
        key=lambda item: (
            safe_coerce_float(
                item.get("updated_at"),
                default=0.0,
                logger=logger,
                module=__name__,
                operation="sort observed runs by updated_at",
                target=str(item.get("run_id", "") or ""),
                kind="timestamp",
            ),
            safe_coerce_float(
                item.get("created_at"),
                default=0.0,
                logger=logger,
                module=__name__,
                operation="sort observed runs by created_at",
                target=str(item.get("run_id", "") or ""),
                kind="timestamp",
            ),
        ),
        reverse=True,
    )
    return summaries[: max(limit, 0)]


def get_observed_run(run_id: str, *, limit: int = 200) -> dict[str, Any]:
    _ensure_run_exists(run_id)
    state = _load_run_state(run_id)

    events = read_observed_events(run_id, limit=limit)
    tool_calls = read_observed_tool_calls(run_id, limit=limit)
    summary = _build_run_summary(state, run_id)
    if not summary["event_count"]:
        summary["event_count"] = len(read_observed_events(run_id, limit=0))

    related_run_ids = {
        str(item).strip()
        for item in list(state.get("related_run_ids", []) or [])
        if str(item).strip()
    }
    for event in events:
        source_run_id = str(event.get("source_run_id", "") or "").strip()
        if source_run_id and source_run_id != run_id:
            related_run_ids.add(source_run_id)

    child_runs = [
        _build_run_summary(_load_run_state(child_id), child_id)
        for child_id in list(state.get("child_run_ids", []) or [])
        if _load_run_state(child_id)
    ]
    workflow_children = [item for item in child_runs if item.get("child_agent_mode") == "workflow"]
    fork_worker_children = [item for item in child_runs if item.get("child_agent_mode") == "fork_worker"]
    fork_tasks = [
        event
        for event in events
        if str(event.get("type", "")).strip().lower().startswith("fork_task_")
    ]

    return {
        "summary": summary,
        "state": state,
        "events": events,
        "tool_calls": tool_calls,
        "artifacts": list_run_artifacts(run_id),
        "related_run_ids": sorted(related_run_ids),
        "parent_run": _build_run_summary(_load_run_state(str(state.get("parent_run_id", "") or "")), str(state.get("parent_run_id", "") or ""))
        if str(state.get("parent_run_id", "") or "").strip()
        else None,
        "child_runs": child_runs,
        "fork_children": child_runs,
        "workflow_children": workflow_children,
        "fork_worker_children": fork_worker_children,
        "fork_tasks": fork_tasks,
        "workflow_log_path": _run_log_path(run_id),
        "global_workflow_log_path": str(WORKFLOW_LOG_PATH),
    }


def list_active_observed_runs(active_chat_runs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    summaries = list_observed_runs(limit=200)
    active: list[dict[str, Any]] = []
    seen: set[str] = set()

    for summary in summaries:
        if summary["status"] not in ACTIVE_RUN_STATUSES:
            continue
        active.append(summary)
        seen.add(summary["run_id"])

    for run_id, payload in (active_chat_runs or {}).items():
        if run_id in seen:
            continue
        active.append(
            {
                "run_id": run_id,
                "thread_id": str(payload.get("thread_id", "") or ""),
                "route": str(payload.get("route", "") or ""),
                "mode": str(payload.get("mode", "") or ""),
                "agent": str(payload.get("agent", "") or ""),
                "status": _normalize_status(str(payload.get("status", "") or "running")),
                "current_stage": str(payload.get("current_stage", "") or ""),
                "last_summary": str(payload.get("last_summary", "") or ""),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at", payload.get("created_at")),
                "terminal_status": "",
                "source": "chat_runtime",
                "event_count": int(payload.get("event_count", 0) or 0),
                "related_run_ids": [],
            }
        )

    active.sort(
        key=lambda item: (
            safe_coerce_float(
                item.get("updated_at"),
                default=0.0,
                logger=logger,
                module=__name__,
                operation="sort active observed runs by updated_at",
                target=str(item.get("run_id", "") or ""),
                kind="timestamp",
            ),
            safe_coerce_float(
                item.get("created_at"),
                default=0.0,
                logger=logger,
                module=__name__,
                operation="sort active observed runs by created_at",
                target=str(item.get("run_id", "") or ""),
                kind="timestamp",
            ),
        ),
        reverse=True,
    )
    return active

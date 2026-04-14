from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ForkChildAgentMode = Literal["workflow", "fork_worker"]
ForkTaskResultStatus = Literal["completed", "failed", "cancelled", "awaiting_confirmation"]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _normalize_text(value)
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return normalized


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    return f"<{value.__class__.__module__}.{value.__class__.__name__}>"


@dataclass(frozen=True)
class ForkTaskSpec:
    parent_run_id: str
    parent_agent: str
    child_agent_name: str
    child_agent_mode: ForkChildAgentMode
    task_type: str
    goal: str
    inputs: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    expected_output: dict[str, Any] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()
    timeout_seconds: int = 60
    priority: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    fork_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_run_id", _normalize_text(self.parent_run_id))
        object.__setattr__(self, "parent_agent", _normalize_text(self.parent_agent))
        object.__setattr__(self, "child_agent_name", _normalize_text(self.child_agent_name))
        object.__setattr__(self, "task_type", _normalize_text(self.task_type))
        object.__setattr__(self, "goal", _normalize_text(self.goal))
        object.__setattr__(self, "fork_reason", _normalize_text(self.fork_reason))
        object.__setattr__(self, "task_id", _normalize_text(self.task_id) or uuid.uuid4().hex)

        normalized_mode = _normalize_text(self.child_agent_mode).lower()
        if normalized_mode not in {"workflow", "fork_worker"}:
            raise ValueError(f"Unsupported child_agent_mode: {self.child_agent_mode!r}")
        object.__setattr__(self, "child_agent_mode", normalized_mode)

        timeout_seconds = int(self.timeout_seconds or 0)
        if timeout_seconds <= 0:
            timeout_seconds = 60
        object.__setattr__(self, "timeout_seconds", timeout_seconds)

        priority = int(self.priority or 0)
        object.__setattr__(self, "priority", max(0, priority))

        object.__setattr__(self, "inputs", dict(self.inputs or {}))
        object.__setattr__(self, "constraints", dict(self.constraints or {}))
        object.__setattr__(self, "expected_output", dict(self.expected_output or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "allowed_tools", tuple(_normalize_text_list(list(self.allowed_tools or ()))))
        object.__setattr__(self, "allowed_actions", tuple(_normalize_text_list(list(self.allowed_actions or ()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "parent_agent": self.parent_agent,
            "child_agent_name": self.child_agent_name,
            "child_agent_mode": self.child_agent_mode,
            "task_type": self.task_type,
            "goal": self.goal,
            "inputs": _serialize_value(self.inputs),
            "constraints": _serialize_value(self.constraints),
            "expected_output": _serialize_value(self.expected_output),
            "allowed_tools": list(self.allowed_tools),
            "allowed_actions": list(self.allowed_actions),
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
            "metadata": _serialize_value(self.metadata),
            "task_id": self.task_id,
            "fork_reason": self.fork_reason,
        }


@dataclass(frozen=True)
class ForkResult:
    task_id: str
    child_run_id: str
    child_agent_name: str
    status: ForkTaskResultStatus
    summary: str
    result_type: str = "structured"
    result_payload: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = field(default_factory=time.time)
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _normalize_text(self.task_id))
        object.__setattr__(self, "child_run_id", _normalize_text(self.child_run_id))
        object.__setattr__(self, "child_agent_name", _normalize_text(self.child_agent_name))
        object.__setattr__(self, "summary", _normalize_text(self.summary))
        object.__setattr__(self, "result_type", _normalize_text(self.result_type) or "structured")
        object.__setattr__(self, "result_payload", dict(self.result_payload or {}))
        object.__setattr__(self, "artifacts", list(self.artifacts or []))
        object.__setattr__(self, "error", _normalize_text(self.error))
        normalized_status = _normalize_text(self.status).lower()
        if normalized_status not in {"completed", "failed", "cancelled", "awaiting_confirmation"}:
            raise ValueError(f"Unsupported fork result status: {self.status!r}")
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(self, "started_at", float(self.started_at or time.time()))
        object.__setattr__(self, "finished_at", float(self.finished_at or self.started_at))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

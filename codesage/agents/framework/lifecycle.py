from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from codesage.agents.framework.base import AgentInstance
from codesage.agents.framework.specs import AgentSpec
from codesage.core.runtime import HookContext, emit_hook

AgentLifecycleStatus = Literal[
    "created",
    "ready",
    "running",
    "awaiting_confirmation",
    "completed",
    "failed",
    "cancelled",
    "disposed",
]

_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "disposed"}
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "created": {"ready", "failed", "cancelled", "disposed"},
    "ready": {"running", "failed", "cancelled", "disposed"},
    "running": {"awaiting_confirmation", "completed", "failed", "cancelled", "disposed"},
    "awaiting_confirmation": {"running", "completed", "failed", "cancelled", "disposed"},
    "completed": {"disposed"},
    "failed": {"disposed"},
    "cancelled": {"disposed"},
    "disposed": set(),
}


@dataclass
class AgentHandle:
    run_id: str
    spec: AgentSpec
    instance: AgentInstance
    status: AgentLifecycleStatus = "created"
    session_id: str = ""
    parent_run_id: str = ""
    task_id: str = ""
    task_type: str = ""
    child_agent_mode: str = ""
    fork_reason: str = ""
    capability_scope: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_error: str = ""
    last_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_name(self) -> str:
        return self.spec.name

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "factory_name": self.spec.factory_name,
            "scope": self.spec.scope,
            "status": self.status,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "child_agent_mode": self.child_agent_mode,
            "fork_reason": self.fork_reason,
            "capability_scope": dict(self.capability_scope),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
            "metadata": dict(self.metadata),
            "instance": self.instance.snapshot(),
        }


class AgentLifecycleManager:
    def __init__(self) -> None:
        self._handles: dict[str, AgentHandle] = {}

    def add(self, handle: AgentHandle) -> AgentHandle:
        if handle.run_id in self._handles:
            raise ValueError(f"Agent run {handle.run_id!r} already exists.")
        self._handles[handle.run_id] = handle
        return handle

    def get(self, run_id: str) -> AgentHandle:
        handle = self._handles.get(str(run_id or "").strip())
        if handle is None:
            raise KeyError(f"Unknown agent run {run_id!r}.")
        return handle

    def list(self, *, agent_name: str | None = None) -> list[AgentHandle]:
        handles = list(self._handles.values())
        if agent_name is None:
            return handles
        normalized_agent_name = str(agent_name or "").strip()
        return [handle for handle in handles if handle.agent_name == normalized_agent_name]

    def transition(
        self,
        run_id: str,
        status: AgentLifecycleStatus,
        *,
        error: str = "",
        result: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentHandle:
        handle = self.get(run_id)
        previous_status = handle.status
        target_status = str(status or "").strip().lower()
        if target_status != handle.status:
            allowed = _ALLOWED_TRANSITIONS.get(handle.status, set())
            if target_status not in allowed:
                raise ValueError(
                    f"Invalid lifecycle transition for {handle.agent_name!r}: "
                    f"{handle.status!r} -> {target_status!r}"
                )
            handle.status = target_status  # type: ignore[assignment]
        handle.updated_at = time.time()
        if error:
            handle.last_error = str(error)
        if result is not None:
            handle.last_result = result
        if metadata:
            handle.metadata.update(metadata)

        transition_response = emit_hook(
            HookContext(
                hook_name="agent_transition",
                route=str(handle.task_type or "").strip(),
                agent=handle.agent_name,
                run_id=handle.run_id,
                action="transition",
                payload={
                    "previous_status": previous_status,
                    "status": target_status,
                    "metadata": dict(metadata or {}),
                },
                result=result,
                error=handle.last_error,
                metadata={
                    "status": target_status,
                    "previous_status": previous_status,
                    "summary": str(handle.metadata.get("summary", "") or ""),
                },
            )
        )
        if transition_response.metadata:
            handle.metadata.update(transition_response.metadata)

        if handle.is_terminal:
            terminal_response = emit_hook(
                HookContext(
                    hook_name="on_terminal",
                    route=str(handle.task_type or "").strip(),
                    agent=handle.agent_name,
                    run_id=handle.run_id,
                    action="terminal",
                    payload={"status": target_status},
                    result=handle.last_result,
                    error=handle.last_error,
                    metadata={
                        "status": target_status,
                        "terminal_status": target_status,
                        "summary": str(handle.metadata.get("summary", "") or handle.last_error or ""),
                    },
                )
            )
            if terminal_response.metadata:
                handle.metadata.update(terminal_response.metadata)
        return handle

    def snapshot(self, run_id: str) -> dict[str, Any]:
        return self.get(run_id).snapshot()


def normalize_result_status(result: Any) -> AgentLifecycleStatus:
    if hasattr(result, "status"):
        raw_status = str(getattr(result, "status", "") or "").strip().lower()
        if raw_status in {"awaiting_confirmation"}:
            return "awaiting_confirmation"
        if raw_status in {"cancelled", "canceled"}:
            return "cancelled"
        if raw_status in {"error", "failed", "timeout", "timed_out", "blocked_low_confidence"}:
            return "failed"
        if raw_status in {"completed", "success"}:
            return "completed"
    if isinstance(result, dict):
        raw_status = str(result.get("status") or result.get("final_status") or "").strip().lower()
        if raw_status in {"awaiting_confirmation"}:
            return "awaiting_confirmation"
        if raw_status in {"cancelled", "canceled"}:
            return "cancelled"
        if raw_status in {"error", "failed", "timeout", "timed_out", "blocked_low_confidence"}:
            return "failed"
        if raw_status in {"completed", "success"}:
            return "completed"
    return "completed"


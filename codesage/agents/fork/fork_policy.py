from __future__ import annotations

from typing import Any

from codesage.agents.fork.fork_models import ForkTaskSpec
from codesage.agents.framework.lifecycle import AgentHandle


ALLOWED_FORK_PARENT_AGENTS = {"supervisor_agent"}
ALLOWED_FORK_CHILD_AGENTS = {
    "pr_review_agent",
    "enhanced_rag_agent",
    "code_modify_agent",
    "fork_worker_agent",
}
MAX_CHILD_RUNS_PER_PARENT = 4


def _normalize_name(value: Any) -> str:
    return str(value or "").strip()


def validate_fork_request(
    parent_handle: AgentHandle | None,
    task: ForkTaskSpec,
    *,
    existing_child_count: int,
) -> None:
    if parent_handle is None:
        raise ValueError("Fork requires a live parent handle.")
    if parent_handle.agent_name not in ALLOWED_FORK_PARENT_AGENTS:
        raise PermissionError(f"Agent {parent_handle.agent_name!r} is not allowed to fork.")
    if parent_handle.parent_run_id:
        raise PermissionError("Fork child runs are not allowed to recursively fork in v1.")
    if task.child_agent_name not in ALLOWED_FORK_CHILD_AGENTS:
        raise PermissionError(f"Child agent {task.child_agent_name!r} is not allowed in fork.")
    if task.parent_agent != parent_handle.agent_name:
        raise ValueError("Fork task parent_agent does not match the live parent handle.")
    if task.parent_run_id != parent_handle.run_id:
        raise ValueError("Fork task parent_run_id does not match the live parent handle.")
    if existing_child_count >= MAX_CHILD_RUNS_PER_PARENT:
        raise PermissionError("Fork child run quota exceeded for the parent run.")
    if task.child_agent_mode == "fork_worker" and task.child_agent_name != "fork_worker_agent":
        raise ValueError("fork_worker mode requires child_agent_name='fork_worker_agent'.")
    if task.child_agent_mode == "workflow" and task.child_agent_name == "fork_worker_agent":
        raise ValueError("fork_worker_agent must run in fork_worker mode.")
    if not task.goal:
        raise ValueError("Fork task requires a non-empty goal.")


def capability_scope_for_task(task: ForkTaskSpec) -> dict[str, Any]:
    return {
        "allowed_tools": list(task.allowed_tools),
        "allowed_actions": list(task.allowed_actions),
        "child_agent_mode": task.child_agent_mode,
        "task_type": task.task_type,
    }


def build_fork_payload(task: ForkTaskSpec, *, thread_id: str) -> dict[str, Any]:
    return {
        "fork_task": task.to_dict(),
        "thread_id": _normalize_name(thread_id),
        "fork_reason": task.fork_reason,
        "capability_scope": capability_scope_for_task(task),
    }


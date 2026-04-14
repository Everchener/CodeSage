from __future__ import annotations

import threading
from typing import Any

from codesage.agents.fork.fork_models import ForkResult, ForkTaskSpec
from codesage.agents.framework.specs import AgentSpec


class CodeModifyAgentRuntime:
    def __init__(
        self,
        *,
        spec: AgentSpec,
        tools: list[Any],
        context: dict[str, Any],
    ) -> None:
        self.spec = spec
        self.tools = list(tools)
        self.context = dict(context)
        self._cancel_event: threading.Event = threading.Event()
        self._disposed = False
        self._last_payload: dict[str, Any] = {}
        self._last_result: dict[str, Any] = {}

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        from codesage.agents.modify.code_modify_agent import invoke_code_modify_agent

        request = dict(self.context)
        request.update(dict(payload or {}))
        instruction = str(request.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("code_modify_agent requires a non-empty 'instruction'.")

        cancel_event = self._resolve_cancel_event(request.get("cancel_event"))
        working_dir = str(request.get("working_dir") or ".")
        result = invoke_code_modify_agent(
            instruction=instruction,
            working_dir=working_dir,
            progress_callback=request.get("progress_callback"),
            approval_mode=str(request.get("approval_mode") or "high_risk"),
            memory_context=self._normalize_optional_text(request.get("memory_context")),
            skill_context=request.get("skill_context"),
            cancel_event=cancel_event,
            run_id=str(request.get("run_id") or "").strip(),
            parent_run_id=str(request.get("parent_run_id") or "").strip(),
        )
        self._last_payload = {
            "instruction": instruction,
            "working_dir": working_dir,
            "approval_mode": str(request.get("approval_mode") or "high_risk"),
        }
        self._last_result = dict(result or {})
        return self._last_result

    def invoke_task(self, task: ForkTaskSpec) -> ForkResult:
        inputs = dict(task.inputs or {})
        instruction = str(inputs.get("instruction", "") or "").strip()
        working_dir = str(inputs.get("working_dir", ".") or ".")
        if not instruction:
            raise ValueError("code_modify_agent fork task requires inputs.instruction.")
        result = self.invoke(
            {
                "instruction": instruction,
                "working_dir": working_dir,
                "approval_mode": inputs.get("approval_mode", "high_risk"),
                "progress_callback": inputs.get("progress_callback"),
                "memory_context": inputs.get("memory_context"),
                "skill_context": inputs.get("skill_context"),
                "cancel_event": inputs.get("cancel_event"),
                "parent_run_id": task.parent_run_id,
            }
        )
        status = str(result.get("status", "completed") or "completed").lower()
        summary = str(result.get("output_result", "") or task.goal)
        payload = {
            "changes_made": list(result.get("changes_made", []) or []),
            "applied_changes": list(result.get("applied_changes", []) or []),
            "preview_id": str(result.get("preview_id", "") or ""),
            "pending_changes": list(result.get("pending_changes", []) or []),
            "risk_reasons": list(result.get("risk_reasons", []) or []),
            "verification_result": str(result.get("verification_result", "") or ""),
        }
        return ForkResult(
            task_id=task.task_id,
            child_run_id="",
            child_agent_name=task.child_agent_name,
            status=status if status in {"completed", "failed", "cancelled", "awaiting_confirmation"} else "completed",
            summary=summary,
            result_type="code_modify",
            result_payload=payload,
            error=str(result.get("error", "") or ""),
        )

    def cancel(self) -> None:
        self._cancel_event.set()

    def dispose(self) -> None:
        self._disposed = True
        self._cancel_event.set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_count": len(self.tools),
            "disposed": self._disposed,
            "cancelled": self._cancel_event.is_set(),
            "last_payload": dict(self._last_payload),
            "last_result": {
                "run_id": self._last_result.get("run_id", ""),
                "status": self._last_result.get("status", ""),
                "final_status": self._last_result.get("final_status", ""),
                "requires_confirmation": bool(self._last_result.get("requires_confirmation", False)),
                "preview_id": self._last_result.get("preview_id", ""),
            },
        }

    def _resolve_cancel_event(self, provided_event: Any) -> Any:
        if provided_event is not None and hasattr(provided_event, "is_set") and hasattr(provided_event, "set"):
            return provided_event
        if self._cancel_event.is_set():
            self._cancel_event = threading.Event()
        return self._cancel_event

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None


def build_code_modify_runtime(
    *,
    spec: AgentSpec,
    tools: list[Any],
    context: dict[str, Any],
) -> CodeModifyAgentRuntime:
    return CodeModifyAgentRuntime(spec=spec, tools=tools, context=context)


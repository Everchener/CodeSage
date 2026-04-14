from __future__ import annotations

import time
from typing import Any

from codesage.agents.fork.fork_models import ForkResult, ForkTaskSpec
from codesage.agents.fork.fork_worker_prompting import (
    FORK_WORKER_SYSTEM_PROMPT,
    ForkWorkerPromptRenderer,
    ForkWorkerTaskSerializer,
)
from codesage.agents.framework.specs import AgentSpec
from codesage.skills.registry import SkillExecutionSpec
from codesage.tools.llm_tools import call_llm_json


class ForkWorkerAgentRuntime:
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
        self._last_task_id = ""
        self._last_prefix = ""
        self._last_result: dict[str, Any] = {}

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = self._coerce_task(payload)
        return self.invoke_task(task).to_dict()

    def invoke_task(self, task: ForkTaskSpec) -> ForkResult:
        if task.task_type == "skill_execution":
            return self.invoke_skill_task(task)

        started_at = time.time()
        self._last_task_id = task.task_id
        self._last_prefix = ForkWorkerTaskSerializer.render_prefix(task)
        prompt = ForkWorkerPromptRenderer.render(task)
        payload = call_llm_json(
            prompt,
            system=FORK_WORKER_SYSTEM_PROMPT,
            max_tokens=min(max(task.timeout_seconds * 10, 300), 1200),
        )
        finished_at = time.time()

        if not isinstance(payload, dict):
            result = ForkResult(
                task_id=task.task_id,
                child_run_id="",
                child_agent_name=task.child_agent_name,
                status="failed",
                summary="ForkWorker did not return a valid JSON payload.",
                result_type="structured",
                result_payload={},
                artifacts=[],
                started_at=started_at,
                finished_at=finished_at,
                error="invalid_json",
            )
            self._last_result = result.to_dict()
            return result

        status = str(payload.get("status", "completed") or "completed").strip().lower()
        if status not in {"completed", "failed", "cancelled", "awaiting_confirmation"}:
            status = "completed"

        result = ForkResult(
            task_id=task.task_id,
            child_run_id="",
            child_agent_name=task.child_agent_name,
            status=status,
            summary=str(payload.get("summary", "") or task.goal).strip(),
            result_type=str(payload.get("result_type", "structured") or "structured"),
            result_payload=dict(payload.get("result_payload", {}) or {}),
            artifacts=list(payload.get("artifacts", []) or []),
            started_at=started_at,
            finished_at=finished_at,
            error=str(payload.get("error", "") or "").strip(),
        )
        self._last_result = result.to_dict()
        return result

    def invoke_skill_task(self, task: ForkTaskSpec) -> ForkResult:
        started_at = time.time()
        self._last_task_id = task.task_id
        self._last_prefix = ForkWorkerTaskSerializer.render_prefix(task)
        spec = self._extract_skill_execution_spec(task)
        self._validate_skill_execution(task, spec)
        prompt = ForkWorkerPromptRenderer.render(task)
        payload = call_llm_json(
            prompt,
            system=FORK_WORKER_SYSTEM_PROMPT,
            max_tokens=min(max(task.timeout_seconds * 10, 300), 1200),
        )
        finished_at = time.time()

        if not isinstance(payload, dict):
            result = ForkResult(
                task_id=task.task_id,
                child_run_id="",
                child_agent_name=task.child_agent_name,
                status="failed",
                summary=f"Skill worker `{spec.skill_name}` did not return a valid JSON payload.",
                result_type="skill_execution",
                result_payload={},
                artifacts=[],
                started_at=started_at,
                finished_at=finished_at,
                error="invalid_json",
            )
            self._last_result = result.to_dict()
            return result

        status = str(payload.get("status", "completed") or "completed").strip().lower()
        if status not in {"completed", "failed", "cancelled", "awaiting_confirmation"}:
            status = "completed"

        result = ForkResult(
            task_id=task.task_id,
            child_run_id="",
            child_agent_name=task.child_agent_name,
            status=status,
            summary=str(payload.get("summary", "") or spec.user_request or task.goal).strip(),
            result_type="skill_execution",
            result_payload={
                "skill_name": spec.skill_name,
                "skill_source": spec.skill_source,
                "selection_mode": spec.selection_mode,
                **dict(payload.get("result_payload", {}) or {}),
            },
            artifacts=list(payload.get("artifacts", []) or []),
            started_at=started_at,
            finished_at=finished_at,
            error=str(payload.get("error", "") or "").strip(),
        )
        self._last_result = result.to_dict()
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_count": len(self.tools),
            "last_task_id": self._last_task_id,
            "last_prefix": self._last_prefix,
            "last_result_keys": sorted(self._last_result),
        }

    @staticmethod
    def _coerce_task(payload: dict[str, Any]) -> ForkTaskSpec:
        raw_task = payload.get("fork_task", payload)
        if isinstance(raw_task, ForkTaskSpec):
            return raw_task
        if not isinstance(raw_task, dict):
            raise ValueError("fork_worker_agent requires a fork_task payload.")
        return ForkTaskSpec(**raw_task)

    @staticmethod
    def _extract_skill_execution_spec(task: ForkTaskSpec) -> SkillExecutionSpec:
        inputs = dict(task.inputs or {})
        raw = inputs.get("skill_execution")
        if isinstance(raw, SkillExecutionSpec):
            return raw
        if not isinstance(raw, dict):
            raise ValueError("skill_execution task requires a skill_execution payload.")
        return SkillExecutionSpec(**raw)

    @staticmethod
    def _validate_skill_execution(task: ForkTaskSpec, spec: SkillExecutionSpec) -> None:
        if not spec.skill_name:
            raise ValueError("skill_execution task requires a non-empty skill_name.")
        unauthorized_tools = [tool for tool in task.allowed_tools if tool not in spec.allowed_tools]
        if unauthorized_tools:
            raise PermissionError(
                f"Skill `{spec.skill_name}` does not allow tools: {', '.join(unauthorized_tools)}"
            )


def build_fork_worker_runtime(
    *,
    spec: AgentSpec,
    tools: list[Any],
    context: dict[str, Any],
) -> ForkWorkerAgentRuntime:
    return ForkWorkerAgentRuntime(spec=spec, tools=tools, context=context)


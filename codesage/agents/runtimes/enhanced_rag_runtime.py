from __future__ import annotations

from typing import Any

from codesage.agents.fork.fork_models import ForkResult, ForkTaskSpec
from codesage.agents.framework.specs import AgentSpec


class EnhancedRAGAgentRuntime:
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
        self._agent: Any | None = None
        self._last_result: dict[str, Any] = {}

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(self.context)
        request.update(dict(payload or {}))
        result = self._get_agent().invoke(request)
        self._last_result = dict(result or {})
        self._last_result.setdefault("status", "completed")
        return self._last_result

    def stream(self, payload: dict[str, Any], config: Any | None = None, stream_mode: str = "values") -> Any:
        del stream_mode
        request = dict(self.context)
        request.update(dict(payload or {}))
        result = self._get_agent().stream_invoke(request, progress_callback=config)
        self._last_result = dict(result or {})
        self._last_result.setdefault("status", "completed")
        return self._last_result

    def invoke_task(self, task: ForkTaskSpec) -> ForkResult:
        inputs = dict(task.inputs or {})
        question = str(inputs.get("question", "") or "").strip()
        if not question:
            raise ValueError("enhanced_rag_agent fork task requires inputs.question.")
        request = {
            "query": question,
            "messages": [],
            "search_scope": inputs.get("search_scope"),
            "expected_output": inputs.get("expected_output"),
        }
        result = self.invoke(request)
        return ForkResult(
            task_id=task.task_id,
            child_run_id="",
            child_agent_name=task.child_agent_name,
            status=str(result.get("status", "completed") or "completed").lower(),
            summary=str(result.get("final_answer", "") or question),
            result_type="rag_answer",
            result_payload={
                "final_answer": str(result.get("final_answer", "") or ""),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_count": len(self.tools),
            "last_result_keys": sorted(self._last_result),
        }

    def _get_agent(self) -> Any:
        if self._agent is None:
            from codesage.agents.rag.enhanced_rag_agent import get_enhanced_rag_agent

            self._agent = get_enhanced_rag_agent()
        return self._agent


def build_enhanced_rag_runtime(
    *,
    spec: AgentSpec,
    tools: list[Any],
    context: dict[str, Any],
) -> EnhancedRAGAgentRuntime:
    return EnhancedRAGAgentRuntime(spec=spec, tools=tools, context=context)


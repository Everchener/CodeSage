from __future__ import annotations

from typing import Any

from codesage.agents.fork.fork_models import ForkResult, ForkTaskSpec
from codesage.agents.framework.specs import AgentSpec


class PRReviewAgentRuntime:
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
        self._last_mode = ""
        self._last_result: dict[str, Any] = {}

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(self.context)
        request.update(dict(payload or {}))

        if self._is_structured_review_request(request):
            self._last_mode = "run"
            result = self._get_agent().run(
                str(request.get("repo") or ""),
                int(request.get("pr_number") or 0),
                str(request.get("diff_text") or ""),
                cancel_event=request.get("cancel_event"),
            )
            self._last_result = {"status": "completed", **dict(result or {})}
            return self._last_result

        self._last_mode = "invoke"
        result = self._get_agent().invoke(request)
        if isinstance(result, dict):
            self._last_result = dict(result)
            self._last_result.setdefault("status", "completed")
            return self._last_result

        self._last_result = {"status": "completed", "result": result}
        return self._last_result

    def invoke_task(self, task: ForkTaskSpec) -> ForkResult:
        inputs = dict(task.inputs or {})
        repo = str(inputs.get("repo", "") or "").strip()
        pr_number = int(inputs.get("pr_number") or 0)
        diff_text = str(inputs.get("diff_text", "") or "")
        if not repo or pr_number <= 0 or not diff_text.strip():
            raise ValueError("pr_review_agent fork task requires repo, pr_number and diff_text.")
        result = self.invoke(
            {
                "repo": repo,
                "pr_number": pr_number,
                "diff_text": diff_text,
                "cancel_event": inputs.get("cancel_event"),
            }
        )
        return ForkResult(
            task_id=task.task_id,
            child_run_id="",
            child_agent_name=task.child_agent_name,
            status=str(result.get("status", "completed") or "completed").lower(),
            summary=str(result.get("final_comment", "") or task.goal),
            result_type="review_report",
            result_payload=dict(result or {}),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_count": len(self.tools),
            "last_mode": self._last_mode,
            "last_result_keys": sorted(self._last_result),
        }

    def _get_agent(self) -> Any:
        if self._agent is None:
            from codesage.agents.review.pr_agent import PRReviewAgent

            self._agent = PRReviewAgent()
        return self._agent

    @staticmethod
    def _is_structured_review_request(payload: dict[str, Any]) -> bool:
        return all(key in payload for key in ("repo", "pr_number", "diff_text"))


def build_pr_review_runtime(
    *,
    spec: AgentSpec,
    tools: list[Any],
    context: dict[str, Any],
) -> PRReviewAgentRuntime:
    return PRReviewAgentRuntime(spec=spec, tools=tools, context=context)


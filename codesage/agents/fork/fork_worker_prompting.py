from __future__ import annotations

import json
from typing import Any

from codesage.agents.fork.fork_models import ForkTaskSpec
from codesage.skills.registry import SkillExecutionSpec
from codesage.tools.prompt_tools import build_system_prompt


FORK_WORKER_SYSTEM_PROMPT = build_system_prompt(
    role="You are CodeSage ForkWorker, a delegated operator for skill execution and structured sub-tasks.",
    responsibilities=[
        "Complete the assigned sub-task using only the provided task contract.",
        "Return a concise structured result that the parent agent can consume.",
        "Stay within the allowed capabilities and do not invent extra scope.",
    ],
    rules=[
        "Do not ask for missing parent context; only use the task fields provided.",
        "Do not reveal chain-of-thought.",
        "Return valid JSON only.",
    ],
    output_instruction="Return a single JSON object and nothing else.",
)


class ForkWorkerPromptScaffold:
    @staticmethod
    def render_skill_prefix(spec: SkillExecutionSpec, *, task_type: str) -> str:
        payload = {
            "role": "fork_worker_agent",
            "task_type": task_type,
            "skill_identity": {
                "skill_name": spec.skill_name,
                "skill_source": spec.skill_source,
                "selection_mode": spec.selection_mode,
            },
            "skill_description": spec.description,
            "skill_compatibility": spec.compatibility,
            "allowed_tools": list(spec.allowed_tools),
            "allowed_actions": list(spec.allowed_actions),
            "expected_output": ForkWorkerTaskSerializer._stable(spec.expected_output),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ForkWorkerTaskSerializer:
    @staticmethod
    def render(task: ForkTaskSpec) -> str:
        payload = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "goal": task.goal,
            "inputs": ForkWorkerTaskSerializer._stable(task.inputs),
            "constraints": ForkWorkerTaskSerializer._stable(task.constraints),
            "expected_output": ForkWorkerTaskSerializer._stable(task.expected_output),
            "allowed_tools": list(task.allowed_tools),
            "allowed_actions": list(task.allowed_actions),
            "metadata": ForkWorkerTaskSerializer._stable(task.metadata),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def render_prefix(task: ForkTaskSpec) -> str:
        if task.task_type == "skill_execution":
            spec = ForkWorkerTaskSerializer.extract_skill_execution_spec(task)
            return ForkWorkerPromptScaffold.render_skill_prefix(spec, task_type=task.task_type)

        payload = {
            "task_type": task.task_type,
            "expected_output": ForkWorkerTaskSerializer._stable(task.expected_output),
            "constraints": ForkWorkerTaskSerializer._stable(task.constraints),
            "normalized_inputs": {
                "output_contract": ForkWorkerTaskSerializer._stable(dict(task.inputs or {}).get("output_contract", {})),
            },
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def extract_skill_execution_spec(task: ForkTaskSpec) -> SkillExecutionSpec:
        inputs = dict(task.inputs or {})
        raw_spec = inputs.get("skill_execution")
        if isinstance(raw_spec, SkillExecutionSpec):
            return raw_spec
        if not isinstance(raw_spec, dict):
            raise ValueError("skill_execution task requires inputs.skill_execution.")
        return SkillExecutionSpec(**raw_spec)

    @staticmethod
    def _stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): ForkWorkerTaskSerializer._stable(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [ForkWorkerTaskSerializer._stable(item) for item in value]
        return value


class ForkWorkerPromptRenderer:
    @staticmethod
    def render(task: ForkTaskSpec) -> str:
        if task.task_type == "skill_execution":
            return ForkWorkerPromptRenderer.render_skill_execution(task)

        inputs = dict(task.inputs or {})
        task_brief = str(inputs.get("task_brief", "") or "").strip()
        structured_inputs = ForkWorkerTaskSerializer._stable(inputs.get("structured_inputs", {}))
        output_contract = ForkWorkerTaskSerializer._stable(inputs.get("output_contract", {}))
        constraints = ForkWorkerTaskSerializer._stable(task.constraints)
        expected_output = ForkWorkerTaskSerializer._stable(task.expected_output)
        return "\n".join(
            [
                "[TASK_BRIEF]",
                task_brief or task.goal,
                "",
                "[STRUCTURED_INPUTS]",
                json.dumps(structured_inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "",
                "[OUTPUT_CONTRACT]",
                json.dumps(output_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "",
                "[CONSTRAINTS]",
                json.dumps(constraints, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "",
                "[EXPECTED_OUTPUT]",
                json.dumps(expected_output, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ]
        ).strip()

    @staticmethod
    def render_skill_execution(task: ForkTaskSpec) -> str:
        inputs = dict(task.inputs or {})
        spec = ForkWorkerTaskSerializer.extract_skill_execution_spec(task)
        task_brief = str(inputs.get("task_brief", "") or spec.user_request or task.goal).strip()
        structured_inputs = ForkWorkerTaskSerializer._stable(inputs.get("structured_inputs", {}))
        output_contract = ForkWorkerTaskSerializer._stable(inputs.get("output_contract", {}))
        return "\n".join(
            [
                "[ROLE]",
                "fork_worker_agent",
                "",
                "[SKILL_IDENTITY]",
                json.dumps(
                    {
                        "skill_name": spec.skill_name,
                        "skill_source": spec.skill_source,
                        "selection_mode": spec.selection_mode,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "",
                "[SKILL_DESCRIPTION]",
                spec.description or "N/A",
                "",
                "[SKILL_COMPATIBILITY]",
                spec.compatibility or "N/A",
                "",
                "[ALLOWED_TOOLS]",
                json.dumps(list(spec.allowed_tools), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "",
                "[TASK_BRIEF]",
                task_brief,
                "",
                "[STRUCTURED_INPUTS]",
                json.dumps(structured_inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "",
                "[OUTPUT_CONTRACT]",
                json.dumps(output_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ]
        ).strip()


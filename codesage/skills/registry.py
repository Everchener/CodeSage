from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from codesage.agents.fork.fork_models import ForkTaskSpec
from codesage.skills.discovery import ResolvedSkill


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def normalize_skill_args(skill_args: dict[str, Any] | None) -> dict[str, Any]:
    return dict(_stable(dict(skill_args or {})))


@dataclass(frozen=True)
class SkillExecutionSpec:
    skill_name: str
    skill_source: str
    selection_mode: str
    user_request: str
    skill_args: dict[str, Any] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()
    expected_output: dict[str, Any] = field(default_factory=dict)
    compatibility: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_name", _normalize_text(self.skill_name))
        object.__setattr__(self, "skill_source", _normalize_text(self.skill_source))
        object.__setattr__(self, "selection_mode", _normalize_text(self.selection_mode))
        object.__setattr__(self, "user_request", _normalize_text(self.user_request))
        object.__setattr__(self, "description", _normalize_text(self.description))
        object.__setattr__(self, "compatibility", _normalize_text(self.compatibility))
        object.__setattr__(self, "skill_args", normalize_skill_args(self.skill_args))
        object.__setattr__(self, "expected_output", dict(_stable(self.expected_output)))
        object.__setattr__(self, "metadata", dict(_stable(self.metadata)))
        object.__setattr__(
            self,
            "allowed_tools",
            tuple(item for item in (_normalize_text(v) for v in self.allowed_tools) if item),
        )
        object.__setattr__(
            self,
            "allowed_actions",
            tuple(item for item in (_normalize_text(v) for v in self.allowed_actions) if item),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_skill_execution(
    resolved_skill: ResolvedSkill | dict[str, Any],
    *,
    user_request: str | None = None,
    skill_args: dict[str, Any] | None = None,
    allowed_actions: Sequence[str] | None = None,
    expected_output: dict[str, Any] | None = None,
) -> SkillExecutionSpec:
    normalized = (
        resolved_skill.to_context_dict()
        if isinstance(resolved_skill, ResolvedSkill)
        else dict(resolved_skill or {})
    )
    return SkillExecutionSpec(
        skill_name=str(normalized.get("name", "") or "").strip(),
        skill_source=str(normalized.get("source", "") or "").strip(),
        selection_mode=str(normalized.get("selection_mode", "") or "").strip(),
        user_request=str(user_request or normalized.get("user_request", "") or "").strip(),
        skill_args=normalize_skill_args(skill_args),
        allowed_tools=tuple(normalized.get("allowed_tools", []) or ()),
        allowed_actions=tuple(allowed_actions or ("skill",)),
        expected_output=dict(expected_output or {"summary": "str", "result_payload": "object"}),
        compatibility=str(normalized.get("compatibility", "") or "").strip(),
        metadata=dict(normalized.get("metadata", {}) or {}),
        description=str(normalized.get("description", "") or "").strip(),
    )


def collect_skill_cache_metadata(spec: SkillExecutionSpec) -> dict[str, str]:
    scaffold_basis = {
        "skill_name": spec.skill_name,
        "skill_source": spec.skill_source,
        "selection_mode": spec.selection_mode,
        "allowed_tools": list(spec.allowed_tools),
        "allowed_actions": list(spec.allowed_actions),
        "compatibility": spec.compatibility,
        "description": spec.description,
        "expected_output": _stable(spec.expected_output),
    }
    digest = hashlib.sha256(str(_stable(scaffold_basis)).encode("utf-8")).hexdigest()[:16]
    return {
        "scaffold_cache_key": f"{spec.skill_name}:{digest}",
        "skill_name": spec.skill_name,
        "skill_source": spec.skill_source,
        "selection_mode": spec.selection_mode,
    }


def build_skill_execution_task(
    spec: SkillExecutionSpec,
    *,
    parent_run_id: str,
    parent_agent: str,
    fork_reason: str = "supervisor_skill_route",
    allowed_tools: Sequence[str] | None = None,
) -> ForkTaskSpec:
    task_allowed_tools = tuple(
        tool_name
        for tool_name in spec.allowed_tools
        if not allowed_tools or tool_name in set(allowed_tools)
    )
    cache_metadata = collect_skill_cache_metadata(spec)
    return ForkTaskSpec(
        parent_run_id=parent_run_id,
        parent_agent=parent_agent,
        child_agent_name="fork_worker_agent",
        child_agent_mode="fork_worker",
        task_type="skill_execution",
        goal=spec.user_request or f"Execute skill `{spec.skill_name}`.",
        inputs={
            "skill_execution": spec.to_dict(),
            "task_brief": spec.user_request or f"Execute skill `{spec.skill_name}`.",
            "structured_inputs": {"skill_args": normalize_skill_args(spec.skill_args)},
            "output_contract": dict(spec.expected_output),
        },
        constraints={
            "compatibility": spec.compatibility,
            "no_parent_context_inheritance": True,
        },
        expected_output=dict(spec.expected_output),
        allowed_tools=task_allowed_tools,
        allowed_actions=spec.allowed_actions,
        metadata={**dict(spec.metadata), **cache_metadata},
        fork_reason=fork_reason,
    )

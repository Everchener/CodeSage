from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentScope = Literal["singleton", "session", "request"]


def normalize_agent_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Agent name cannot be empty.")
    return normalized


@dataclass(frozen=True)
class AgentSpec:
    name: str
    factory_name: str
    scope: AgentScope
    default_tools: tuple[str, ...] = ()
    supports_stream: bool = False
    supports_cancel: bool = False
    supports_resume: bool = False
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_agent_name(self.name))
        object.__setattr__(self, "factory_name", normalize_agent_name(self.factory_name))
        object.__setattr__(self, "scope", self._normalize_scope(self.scope))
        object.__setattr__(
            self,
            "default_tools",
            tuple(tool_name for tool_name in self.default_tools if str(tool_name or "").strip()),
        )
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @staticmethod
    def _normalize_scope(scope: str) -> AgentScope:
        normalized = str(scope or "").strip().lower()
        if normalized not in {"singleton", "session", "request"}:
            raise ValueError(f"Unsupported agent scope: {scope!r}")
        return normalized  # type: ignore[return-value]

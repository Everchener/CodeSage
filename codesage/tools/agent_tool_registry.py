from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def _normalize_name(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _resolve_tool_name(tool: Any, explicit_name: str | None = None) -> str:
    candidate = explicit_name or getattr(tool, "name", None) or getattr(tool, "__name__", None)
    return _normalize_name(str(candidate or ""), field_name="tool name")


def _unique_ordered_names(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_name(value, field_name="agent name")
        if normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    tool: Any
    description: str = ""
    agents: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentToolRegistry:
    """Manage tool registration and agent-to-tool bindings in one place."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}
        self._agent_tools: dict[str, list[str]] = {}

    def register(
        self,
        tool: Any,
        *,
        name: str | None = None,
        agent: str | None = None,
        agents: Iterable[str] | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ToolRegistration:
        tool_name = _resolve_tool_name(tool, name)
        agent_names = self._merge_agent_names(agent=agent, agents=agents)
        existing = self._tools.get(tool_name)

        if existing is not None and existing.tool is not tool and not overwrite:
            raise ValueError(f"Tool {tool_name!r} is already registered.")

        merged_agent_names = list(existing.agents) if existing is not None else []
        merged_agent_names.extend(agent_names)
        normalized_agents = tuple(_unique_ordered_names(merged_agent_names))

        registration = ToolRegistration(
            name=tool_name,
            tool=tool,
            description=str(description if description is not None else getattr(tool, "description", "") or ""),
            agents=normalized_agents,
            metadata=dict(metadata or {}),
        )
        self._tools[tool_name] = registration

        for agent_name in normalized_agents:
            self._bind(agent_name, tool_name)

        return registration

    def register_agent_tools(
        self,
        agent_name: str,
        tools: Iterable[Any],
        *,
        overwrite: bool = False,
    ) -> list[ToolRegistration]:
        normalized_agent_name = _normalize_name(agent_name, field_name="agent name")
        registrations: list[ToolRegistration] = []
        for tool in tools:
            registrations.append(
                self.register(tool, agent=normalized_agent_name, overwrite=overwrite)
            )
        return registrations

    def bind_tool(self, agent_name: str, tool_name: str) -> None:
        normalized_agent_name = _normalize_name(agent_name, field_name="agent name")
        normalized_tool_name = _normalize_name(tool_name, field_name="tool name")
        if normalized_tool_name not in self._tools:
            raise KeyError(f"Tool {normalized_tool_name!r} is not registered.")

        registration = self._tools[normalized_tool_name]
        if normalized_agent_name not in registration.agents:
            self._tools[normalized_tool_name] = ToolRegistration(
                name=registration.name,
                tool=registration.tool,
                description=registration.description,
                agents=tuple(_unique_ordered_names([*registration.agents, normalized_agent_name])),
                metadata=dict(registration.metadata),
            )

        self._bind(normalized_agent_name, normalized_tool_name)

    def get_registration(self, tool_name: str) -> ToolRegistration:
        normalized_tool_name = _normalize_name(tool_name, field_name="tool name")
        registration = self._tools.get(normalized_tool_name)
        if registration is None:
            raise KeyError(f"Tool {normalized_tool_name!r} is not registered.")
        return registration

    def get_tool(self, tool_name: str) -> Any:
        return self.get_registration(tool_name).tool

    def get_tools_for_agent(self, agent_name: str) -> list[Any]:
        return [
            self._tools[tool_name].tool
            for tool_name in self.get_tool_names_for_agent(agent_name)
        ]

    def get_tool_names_for_agent(self, agent_name: str) -> list[str]:
        normalized_agent_name = _normalize_name(agent_name, field_name="agent name")
        return list(self._agent_tools.get(normalized_agent_name, []))

    def list_agents(self) -> list[str]:
        return sorted(self._agent_tools)

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def snapshot(self) -> dict[str, Any]:
        return {
            "tools": {
                name: {
                    "description": registration.description,
                    "agents": list(registration.agents),
                    "metadata": dict(registration.metadata),
                }
                for name, registration in sorted(self._tools.items())
            },
            "agents": {
                agent_name: list(tool_names)
                for agent_name, tool_names in sorted(self._agent_tools.items())
            },
        }

    @staticmethod
    def _merge_agent_names(
        *,
        agent: str | None,
        agents: Iterable[str] | None,
    ) -> list[str]:
        merged: list[str] = []
        if agent is not None:
            merged.append(agent)
        if agents is not None:
            merged.extend(agents)
        return _unique_ordered_names(merged)

    def _bind(self, agent_name: str, tool_name: str) -> None:
        tool_names = self._agent_tools.setdefault(agent_name, [])
        if tool_name not in tool_names:
            tool_names.append(tool_name)


DEFAULT_AGENT_TOOL_REGISTRY = AgentToolRegistry()


def register_agent_tool(
    tool: Any,
    *,
    name: str | None = None,
    agent: str | None = None,
    agents: Iterable[str] | None = None,
    description: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> ToolRegistration:
    return DEFAULT_AGENT_TOOL_REGISTRY.register(
        tool,
        name=name,
        agent=agent,
        agents=agents,
        description=description,
        metadata=metadata,
        overwrite=overwrite,
    )


def register_agent_tools(
    agent_name: str,
    tools: Iterable[Any],
    *,
    overwrite: bool = False,
) -> list[ToolRegistration]:
    return DEFAULT_AGENT_TOOL_REGISTRY.register_agent_tools(
        agent_name,
        tools,
        overwrite=overwrite,
    )

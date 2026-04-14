from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

from codesage.agents.framework.bootstrap import (
    get_default_agent_manager,
    register_builtin_agent_tools,
    reset_default_agent_manager,
)
from codesage.agents.framework.manager import AgentManager
from codesage.core.builtin_hooks import register_builtin_hooks
from codesage.core.runtime import list_registered_hooks
from codesage.tools.agent_tool_registry import AgentToolRegistry, DEFAULT_AGENT_TOOL_REGISTRY


@dataclass(frozen=True)
class ApiRuntimeServices:
    agent_manager: AgentManager
    tool_registry: AgentToolRegistry
    bootstrap_report: dict[str, Any]


def initialize_api_runtime(
    app: Any,
    *,
    tool_registry: AgentToolRegistry | None = None,
    reset_manager: bool = False,
) -> ApiRuntimeServices:
    if not hasattr(app, "state"):
        app.state = SimpleNamespace()
    resolved_tool_registry = tool_registry or DEFAULT_AGENT_TOOL_REGISTRY
    if reset_manager:
        reset_default_agent_manager()

    register_builtin_agent_tools(resolved_tool_registry)
    register_builtin_hooks(overwrite=True)
    agent_manager = get_default_agent_manager()
    bootstrap_report = {
        "status": "ok",
        "agent_count": len(agent_manager.list_specs()),
        "agents": [spec.name for spec in agent_manager.list_specs()],
        "tool_agents": resolved_tool_registry.list_agents(),
        "hook_count": len(list_registered_hooks()),
        "hook_names": [hook.handler_name for hook in list_registered_hooks()],
    }
    services = ApiRuntimeServices(
        agent_manager=agent_manager,
        tool_registry=resolved_tool_registry,
        bootstrap_report=bootstrap_report,
    )
    app.state.runtime_services = services
    app.state.agent_manager = agent_manager
    app.state.agent_bootstrap_error = ""
    app.state.agent_bootstrap_report = bootstrap_report
    return services


def get_api_runtime_services(app: FastAPI) -> ApiRuntimeServices:
    if not hasattr(app, "state"):
        app.state = SimpleNamespace()
    services = getattr(app.state, "runtime_services", None)
    if services is None:
        return initialize_api_runtime(app)
    return services


def get_app_agent_manager(app: Any) -> AgentManager:
    services = get_api_runtime_services(app)
    return services.agent_manager

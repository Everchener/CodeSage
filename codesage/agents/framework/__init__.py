"""Lazy exports for agent framework primitives."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesage.agents.framework.base import AgentInstance, ManagedAgentInstance
    from codesage.agents.framework.bootstrap import (
        get_default_agent_manager,
        register_builtin_agent_tools,
        register_builtin_agents,
        reset_default_agent_manager,
    )
    from codesage.agents.framework.factory import AgentFactory, CallableAgentFactory, LazyImportAgentFactory
    from codesage.agents.framework.lifecycle import AgentHandle, AgentLifecycleManager
    from codesage.agents.framework.manager import AgentManager
    from codesage.agents.framework.specs import AgentSpec

__all__ = [
    "AgentFactory",
    "AgentHandle",
    "AgentInstance",
    "AgentLifecycleManager",
    "AgentManager",
    "AgentSpec",
    "CallableAgentFactory",
    "LazyImportAgentFactory",
    "ManagedAgentInstance",
    "get_default_agent_manager",
    "register_builtin_agent_tools",
    "register_builtin_agents",
    "reset_default_agent_manager",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "AgentFactory": ("codesage.agents.framework.factory", "AgentFactory"),
    "AgentHandle": ("codesage.agents.framework.lifecycle", "AgentHandle"),
    "AgentInstance": ("codesage.agents.framework.base", "AgentInstance"),
    "AgentLifecycleManager": ("codesage.agents.framework.lifecycle", "AgentLifecycleManager"),
    "AgentManager": ("codesage.agents.framework.manager", "AgentManager"),
    "AgentSpec": ("codesage.agents.framework.specs", "AgentSpec"),
    "CallableAgentFactory": ("codesage.agents.framework.factory", "CallableAgentFactory"),
    "LazyImportAgentFactory": ("codesage.agents.framework.factory", "LazyImportAgentFactory"),
    "ManagedAgentInstance": ("codesage.agents.framework.base", "ManagedAgentInstance"),
    "get_default_agent_manager": ("codesage.agents.framework.bootstrap", "get_default_agent_manager"),
    "register_builtin_agent_tools": ("codesage.agents.framework.bootstrap", "register_builtin_agent_tools"),
    "register_builtin_agents": ("codesage.agents.framework.bootstrap", "register_builtin_agents"),
    "reset_default_agent_manager": ("codesage.agents.framework.bootstrap", "reset_default_agent_manager"),
}


def __getattr__(name: str) -> Any:
    export = _EXPORT_MAP.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = export
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

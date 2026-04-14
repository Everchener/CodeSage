from __future__ import annotations

from importlib import import_module
from typing import Any

from codesage.agents.runtimes.code_modify_runtime import build_code_modify_runtime
from codesage.agents.runtimes.enhanced_rag_runtime import build_enhanced_rag_runtime
from codesage.agents.framework.factory import CallableAgentFactory
from codesage.agents.framework.factory import LazyImportAgentFactory
from codesage.agents.runtimes.fork_worker_runtime import build_fork_worker_runtime
from codesage.agents.framework.manager import AgentManager
from codesage.agents.runtimes.pr_review_runtime import build_pr_review_runtime
from codesage.agents.framework.specs import AgentSpec
from codesage.tools.agent_tool_registry import AgentToolRegistry, DEFAULT_AGENT_TOOL_REGISTRY

_default_agent_manager: AgentManager | None = None
_BUILTIN_TOOL_REGISTRARS: tuple[str, ...] = (
    "codesage.agents.routing.supervisor_agent:register_supervisor_tools",
)


def _build_builtin_specs() -> list[AgentSpec]:
    return [
        AgentSpec(
            name="supervisor_agent",
            factory_name="supervisor_factory",
            scope="singleton",
            supports_stream=True,
            description="Supervisor router agent.",
        ),
        AgentSpec(
            name="enhanced_rag_agent",
            factory_name="enhanced_rag_factory",
            scope="singleton",
            supports_stream=True,
            description="Repository question answering agent.",
        ),
        AgentSpec(
            name="pr_review_agent",
            factory_name="pr_review_factory",
            scope="request",
            description="Pull request review agent.",
        ),
        AgentSpec(
            name="code_modify_agent",
            factory_name="code_modify_factory",
            scope="request",
            supports_cancel=True,
            description="Code modification workflow agent.",
        ),
        AgentSpec(
            name="fork_worker_agent",
            factory_name="fork_worker_factory",
            scope="request",
            description="Prompt-driven fork worker agent.",
        ),
    ]


def _build_builtin_factories() -> dict[str, Any]:
    return {
        "supervisor_factory": LazyImportAgentFactory(
            "codesage.agents.routing.supervisor_agent:build_supervisor_runtime"
        ),
        "enhanced_rag_factory": CallableAgentFactory(build_enhanced_rag_runtime),
        "pr_review_factory": CallableAgentFactory(build_pr_review_runtime),
        "code_modify_factory": CallableAgentFactory(build_code_modify_runtime),
        "fork_worker_factory": CallableAgentFactory(build_fork_worker_runtime),
    }


def _load_callable(import_path: str) -> Any:
    module_name, attr_name = import_path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def register_builtin_agents(
    manager: AgentManager | None = None,
    *,
    tool_registry: AgentToolRegistry | None = None,
) -> AgentManager:
    resolved_tool_registry = tool_registry or DEFAULT_AGENT_TOOL_REGISTRY
    resolved_manager = manager or AgentManager(tool_registry=resolved_tool_registry)

    for spec in _build_builtin_specs():
        resolved_manager.register_spec(spec, overwrite=True)
    for factory_name, factory in _build_builtin_factories().items():
        resolved_manager.register_factory(factory_name, factory, overwrite=True)

    return resolved_manager


def register_builtin_agent_tools(
    tool_registry: AgentToolRegistry | None = None,
) -> AgentToolRegistry:
    resolved_tool_registry = tool_registry or DEFAULT_AGENT_TOOL_REGISTRY
    for import_path in _BUILTIN_TOOL_REGISTRARS:
        registrar = _load_callable(import_path)
        registrar(resolved_tool_registry)
    return resolved_tool_registry


def get_default_agent_manager() -> AgentManager:
    global _default_agent_manager
    if _default_agent_manager is None:
        _default_agent_manager = register_builtin_agents()
    return _default_agent_manager


def reset_default_agent_manager() -> None:
    global _default_agent_manager
    _default_agent_manager = None


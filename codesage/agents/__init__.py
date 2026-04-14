"""为 CodeSage Agent 入口提供惰性导出。"""

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
    from codesage.agents.modify.code_modify_agent import get_code_modify_agent, invoke_code_modify_agent
    from codesage.agents.rag.enhanced_rag_agent import EnhancedRAGAgent, get_enhanced_rag_agent
    from codesage.agents.framework.factory import CallableAgentFactory, LazyImportAgentFactory
    from codesage.agents.fork.fork_models import ForkResult, ForkTaskSpec
    from codesage.agents.framework.lifecycle import AgentHandle, AgentLifecycleManager
    from codesage.agents.framework.manager import AgentManager
    from codesage.agents.review.pr_agent import PRReviewAgent, ReviewState
    from codesage.agents.framework.specs import AgentSpec
    from codesage.agents.routing.supervisor_agent import get_supervisor, invoke_supervisor
    from codesage.skills.registry import SkillExecutionSpec

__all__ = [
    "AgentInstance",
    "ManagedAgentInstance",
    "AgentSpec",
    "ForkTaskSpec",
    "ForkResult",
    "SkillExecutionSpec",
    "CallableAgentFactory",
    "LazyImportAgentFactory",
    "AgentHandle",
    "AgentLifecycleManager",
    "AgentManager",
    "register_builtin_agents",
    "register_builtin_agent_tools",
    "get_default_agent_manager",
    "reset_default_agent_manager",
    "get_supervisor",
    "invoke_supervisor",
    "PRReviewAgent",
    "ReviewState",
    "EnhancedRAGAgent",
    "get_enhanced_rag_agent",
    "get_code_modify_agent",
    "invoke_code_modify_agent",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "AgentInstance": ("codesage.agents.framework.base", "AgentInstance"),
    "ManagedAgentInstance": ("codesage.agents.framework.base", "ManagedAgentInstance"),
    "AgentSpec": ("codesage.agents.framework.specs", "AgentSpec"),
    "ForkTaskSpec": ("codesage.agents.fork.fork_models", "ForkTaskSpec"),
    "ForkResult": ("codesage.agents.fork.fork_models", "ForkResult"),
    "SkillExecutionSpec": ("codesage.skills.registry", "SkillExecutionSpec"),
    "CallableAgentFactory": ("codesage.agents.framework.factory", "CallableAgentFactory"),
    "LazyImportAgentFactory": ("codesage.agents.framework.factory", "LazyImportAgentFactory"),
    "AgentHandle": ("codesage.agents.framework.lifecycle", "AgentHandle"),
    "AgentLifecycleManager": ("codesage.agents.framework.lifecycle", "AgentLifecycleManager"),
    "AgentManager": ("codesage.agents.framework.manager", "AgentManager"),
    "register_builtin_agents": ("codesage.agents.framework.bootstrap", "register_builtin_agents"),
    "register_builtin_agent_tools": ("codesage.agents.framework.bootstrap", "register_builtin_agent_tools"),
    "get_default_agent_manager": ("codesage.agents.framework.bootstrap", "get_default_agent_manager"),
    "reset_default_agent_manager": ("codesage.agents.framework.bootstrap", "reset_default_agent_manager"),
    "get_supervisor": ("codesage.agents.routing.supervisor_agent", "get_supervisor"),
    "invoke_supervisor": ("codesage.agents.routing.supervisor_agent", "invoke_supervisor"),
    "PRReviewAgent": ("codesage.agents.review.pr_agent", "PRReviewAgent"),
    "ReviewState": ("codesage.agents.review.pr_agent", "ReviewState"),
    "EnhancedRAGAgent": ("codesage.agents.rag.enhanced_rag_agent", "EnhancedRAGAgent"),
    "get_enhanced_rag_agent": ("codesage.agents.rag.enhanced_rag_agent", "get_enhanced_rag_agent"),
    "get_code_modify_agent": ("codesage.agents.modify.code_modify_agent", "get_code_modify_agent"),
    "invoke_code_modify_agent": ("codesage.agents.modify.code_modify_agent", "invoke_code_modify_agent"),
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

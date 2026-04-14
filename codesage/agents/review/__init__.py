"""Lazy exports for review-related agents."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesage.agents.review.logic_agent import LogicAgent
    from codesage.agents.review.review_rag_agent import RAGAgent, ReviewRAGResult
    from codesage.agents.review.security_agent import SecurityAgent
    from codesage.agents.review.supervisor import SupervisorAgent

__all__ = [
    "LogicAgent",
    "RAGAgent",
    "ReviewRAGResult",
    "SecurityAgent",
    "SupervisorAgent",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "LogicAgent": ("codesage.agents.review.logic_agent", "LogicAgent"),
    "RAGAgent": ("codesage.agents.review.review_rag_agent", "RAGAgent"),
    "ReviewRAGResult": ("codesage.agents.review.review_rag_agent", "ReviewRAGResult"),
    "SecurityAgent": ("codesage.agents.review.security_agent", "SecurityAgent"),
    "SupervisorAgent": ("codesage.agents.review.supervisor", "SupervisorAgent"),
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

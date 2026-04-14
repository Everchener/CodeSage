"""Lazy exports for RAG-oriented agents."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesage.agents.rag.enhanced_rag_agent import EnhancedRAGAgent, get_enhanced_rag_agent
    from codesage.agents.rag.turn_summary_agent import SNAPSHOT_SCHEMA, TurnSummaryAgent

__all__ = [
    "EnhancedRAGAgent",
    "SNAPSHOT_SCHEMA",
    "TurnSummaryAgent",
    "get_enhanced_rag_agent",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "EnhancedRAGAgent": ("codesage.agents.rag.enhanced_rag_agent", "EnhancedRAGAgent"),
    "SNAPSHOT_SCHEMA": ("codesage.agents.rag.turn_summary_agent", "SNAPSHOT_SCHEMA"),
    "TurnSummaryAgent": ("codesage.agents.rag.turn_summary_agent", "TurnSummaryAgent"),
    "get_enhanced_rag_agent": ("codesage.agents.rag.enhanced_rag_agent", "get_enhanced_rag_agent"),
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

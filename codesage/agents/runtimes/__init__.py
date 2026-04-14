"""Lazy exports for agent runtime adapters."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesage.agents.runtimes.code_modify_runtime import build_code_modify_runtime
    from codesage.agents.runtimes.enhanced_rag_runtime import build_enhanced_rag_runtime
    from codesage.agents.runtimes.fork_worker_runtime import build_fork_worker_runtime
    from codesage.agents.runtimes.pr_review_runtime import build_pr_review_runtime

__all__ = [
    "build_code_modify_runtime",
    "build_enhanced_rag_runtime",
    "build_fork_worker_runtime",
    "build_pr_review_runtime",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "build_code_modify_runtime": ("codesage.agents.runtimes.code_modify_runtime", "build_code_modify_runtime"),
    "build_enhanced_rag_runtime": ("codesage.agents.runtimes.enhanced_rag_runtime", "build_enhanced_rag_runtime"),
    "build_fork_worker_runtime": ("codesage.agents.runtimes.fork_worker_runtime", "build_fork_worker_runtime"),
    "build_pr_review_runtime": ("codesage.agents.runtimes.pr_review_runtime", "build_pr_review_runtime"),
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

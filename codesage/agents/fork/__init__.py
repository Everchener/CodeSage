"""Lazy exports for fork task helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesage.agents.fork.fork_models import ForkResult, ForkTaskSpec
    from codesage.agents.fork.fork_worker_prompting import ForkWorkerPromptRenderer, ForkWorkerTaskSerializer

__all__ = [
    "ForkResult",
    "ForkTaskSpec",
    "ForkWorkerPromptRenderer",
    "ForkWorkerTaskSerializer",
    "build_fork_payload",
    "capability_scope_for_task",
    "validate_fork_request",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "ForkResult": ("codesage.agents.fork.fork_models", "ForkResult"),
    "ForkTaskSpec": ("codesage.agents.fork.fork_models", "ForkTaskSpec"),
    "ForkWorkerPromptRenderer": ("codesage.agents.fork.fork_worker_prompting", "ForkWorkerPromptRenderer"),
    "ForkWorkerTaskSerializer": ("codesage.agents.fork.fork_worker_prompting", "ForkWorkerTaskSerializer"),
    "build_fork_payload": ("codesage.agents.fork.fork_policy", "build_fork_payload"),
    "capability_scope_for_task": ("codesage.agents.fork.fork_policy", "capability_scope_for_task"),
    "validate_fork_request": ("codesage.agents.fork.fork_policy", "validate_fork_request"),
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

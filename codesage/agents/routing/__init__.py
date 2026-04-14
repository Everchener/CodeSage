"""Lazy exports for routing agents."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesage.agents.routing.supervisor_agent import (
        build_supervisor_runtime,
        get_supervisor,
        invoke_supervisor,
        register_supervisor_tools,
    )

__all__ = [
    "build_supervisor_runtime",
    "get_supervisor",
    "invoke_supervisor",
    "register_supervisor_tools",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "build_supervisor_runtime": ("codesage.agents.routing.supervisor_agent", "build_supervisor_runtime"),
    "get_supervisor": ("codesage.agents.routing.supervisor_agent", "get_supervisor"),
    "invoke_supervisor": ("codesage.agents.routing.supervisor_agent", "invoke_supervisor"),
    "register_supervisor_tools": ("codesage.agents.routing.supervisor_agent", "register_supervisor_tools"),
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

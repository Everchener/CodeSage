"""Lazy exports for code modification agents."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesage.agents.modify.code_modify_agent import get_code_modify_agent, invoke_code_modify_agent

__all__ = [
    "get_code_modify_agent",
    "invoke_code_modify_agent",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
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

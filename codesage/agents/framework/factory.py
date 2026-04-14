from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Protocol

from codesage.agents.framework.base import AgentInstance, ManagedAgentInstance
from codesage.agents.framework.specs import AgentSpec


class AgentFactory(Protocol):
    def create(
        self,
        spec: AgentSpec,
        *,
        tools: list[Any],
        context: dict[str, Any] | None = None,
    ) -> AgentInstance:
        ...


class CallableAgentFactory:
    def __init__(
        self,
        builder: Callable[..., Any],
        *,
        wraps_runtime: bool = True,
    ) -> None:
        self._builder = builder
        self._wraps_runtime = wraps_runtime

    def create(
        self,
        spec: AgentSpec,
        *,
        tools: list[Any],
        context: dict[str, Any] | None = None,
    ) -> AgentInstance:
        runtime = self._builder(spec=spec, tools=tools, context=dict(context or {}))
        if isinstance(runtime, ManagedAgentInstance):
            return runtime
        if not self._wraps_runtime and isinstance(runtime, AgentInstance):
            return runtime
        return ManagedAgentInstance(
            spec=spec,
            runtime=runtime,
            metadata={
                "tool_count": len(tools),
                "context_keys": sorted(dict(context or {})),
            },
        )


class LazyImportAgentFactory:
    def __init__(
        self,
        import_path: str,
        *,
        wraps_runtime: bool = True,
        builder_kwargs_resolver: Callable[[AgentSpec, list[Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.import_path = str(import_path or "").strip()
        if not self.import_path or ":" not in self.import_path:
            raise ValueError("import_path must look like 'package.module:attribute'.")
        self._wraps_runtime = wraps_runtime
        self._builder_kwargs_resolver = builder_kwargs_resolver

    def create(
        self,
        spec: AgentSpec,
        *,
        tools: list[Any],
        context: dict[str, Any] | None = None,
    ) -> AgentInstance:
        builder = self._load_builder()
        resolved_context = dict(context or {})
        kwargs = (
            self._builder_kwargs_resolver(spec, tools, resolved_context)
            if self._builder_kwargs_resolver is not None
            else {}
        )
        runtime = builder(**kwargs)
        if isinstance(runtime, ManagedAgentInstance):
            return runtime
        if not self._wraps_runtime and isinstance(runtime, AgentInstance):
            return runtime
        return ManagedAgentInstance(
            spec=spec,
            runtime=runtime,
            metadata={
                "factory": "lazy_import",
                "import_path": self.import_path,
                "tool_count": len(tools),
                "context_keys": sorted(resolved_context),
            },
        )

    def _load_builder(self) -> Callable[..., Any]:
        module_name, attr_name = self.import_path.split(":", 1)
        module = import_module(module_name)
        builder = getattr(module, attr_name, None)
        if builder is None or not callable(builder):
            raise AttributeError(f"Cannot load callable builder from {self.import_path!r}.")
        return builder


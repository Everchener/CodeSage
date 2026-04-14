from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codesage.agents.framework.specs import AgentSpec


@runtime_checkable
class AgentInstance(Protocol):
    spec: "AgentSpec"

    def invoke(self, payload: dict[str, Any]) -> Any:
        ...

    def stream(
        self,
        payload: dict[str, Any],
        config: Any | None = None,
        stream_mode: str = "values",
    ) -> Any:
        ...

    def cancel(self) -> None:
        ...

    def dispose(self) -> None:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...

    def invoke_task(self, task: Any) -> Any:
        ...


@dataclass
class ManagedAgentInstance:
    spec: "AgentSpec"
    runtime: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def invoke(self, payload: dict[str, Any]) -> Any:
        if hasattr(self.runtime, "invoke"):
            return self.runtime.invoke(payload)
        if callable(self.runtime):
            return self.runtime(payload)
        raise TypeError(f"Agent {self.spec.name!r} runtime does not support invoke().")

    def invoke_task(self, task: Any) -> Any:
        if hasattr(self.runtime, "invoke_task"):
            return self.runtime.invoke_task(task)
        return self.invoke({"fork_task": task.to_dict() if hasattr(task, "to_dict") else task})

    def stream(
        self,
        payload: dict[str, Any],
        config: Any | None = None,
        stream_mode: str = "values",
    ) -> Any:
        if hasattr(self.runtime, "stream_invoke"):
            progress_callback = config if callable(config) else None
            return self.runtime.stream_invoke(payload, progress_callback=progress_callback)
        if hasattr(self.runtime, "stream"):
            return self.runtime.stream(payload, config=config, stream_mode=stream_mode)
        if hasattr(self.runtime, "invoke"):
            return self.runtime.invoke(payload)
        if callable(self.runtime):
            return self.runtime(payload)
        raise TypeError(f"Agent {self.spec.name!r} runtime does not support stream().")

    def cancel(self) -> None:
        if hasattr(self.runtime, "cancel"):
            self.runtime.cancel()

    def dispose(self) -> None:
        if hasattr(self.runtime, "dispose"):
            self.runtime.dispose()

    def snapshot(self) -> dict[str, Any]:
        runtime_snapshot: dict[str, Any] = {}
        if hasattr(self.runtime, "snapshot"):
            snapshot = self.runtime.snapshot()
            if isinstance(snapshot, dict):
                runtime_snapshot = dict(snapshot)
        return {
            "agent_name": self.spec.name,
            "factory_name": self.spec.factory_name,
            "runtime_type": type(self.runtime).__name__,
            "metadata": dict(self.metadata),
            "runtime": runtime_snapshot,
        }


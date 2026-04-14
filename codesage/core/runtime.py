from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from codesage.core.error_handling import safe_append_text, safe_json_loads, safe_read_text, safe_write_text

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_BASE_DIR = REPO_ROOT / ".codesage" / "runs"

RunStatus = Literal[
    "running",
    "completed",
    "error",
    "cancelled",
    "awaiting_confirmation",
    "timed_out",
]
RunTerminalState = Literal[
    "completed",
    "blocked_confirmation",
    "cancelled",
    "failed",
    "timed_out",
]
RecoveryReason = Literal["none", "timeout", "low_confidence", "tool_error", "manual_cancel"]
HookName = Literal[
    "before_agent_create",
    "after_agent_create",
    "before_agent_invoke",
    "after_agent_invoke",
    "before_agent_stream",
    "after_agent_stream",
    "agent_transition",
    "before_route",
    "after_route",
    "before_fork",
    "after_fork",
    "on_progress",
    "before_tool_call",
    "after_tool_call",
    "on_terminal",
]
HookKind = Literal["policy", "runtime", "observer"]
HookDecision = Literal["approve", "block", "continue"]
HookHandler = Callable[["HookContext"], "HookResponse | dict[str, Any] | None"]


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str
    kind: str = "text"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "path": self.path, "kind": self.kind}


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    route: str
    stage: str
    status: str
    summary: str
    event_type: str = "step"
    agent: str = ""
    tool: str = ""
    terminal_status: str = ""
    recovery_reason: str = ""
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.event_type,
            "run_id": self.run_id,
            "route": self.route,
            "stage": self.stage,
            "status": self.status,
            "summary": self.summary,
            "agent": self.agent,
            "tool": self.tool,
            "terminal_status": self.terminal_status,
            "recovery_reason": self.recovery_reason,
            "artifact_refs": self.artifact_refs,
        }
        payload.update(self.extra)
        return {key: value for key, value in payload.items() if value not in ("", None, [], {})}


@dataclass(frozen=True)
class HookContext:
    hook_name: str
    route: str
    agent: str
    tool_name: str = ""
    run_id: str = ""
    action: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_name": self.hook_name,
            "route": self.route,
            "agent": self.agent,
            "tool_name": self.tool_name,
            "run_id": self.run_id,
            "action": self.action,
            "payload": _json_safe(self.payload),
            "result": _json_safe(self.result),
            "error": self.error,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class HookResponse:
    decision: HookDecision = "continue"
    updated_payload: dict[str, Any] = field(default_factory=dict)
    additional_context: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "updated_payload": _json_safe(self.updated_payload),
            "additional_context": _json_safe(self.additional_context),
            "message": self.message,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class RegisteredHook:
    hook_name: str
    handler: HookHandler
    priority: int = 0
    hook_kind: HookKind = "observer"
    agent: str = ""
    route: str = ""
    tool_name: str = ""
    blocking: bool = False

    @property
    def handler_name(self) -> str:
        return getattr(self.handler, "__name__", self.handler.__class__.__name__)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "content"):
        return str(getattr(value, "content", ""))
    return str(value)


def ensure_run_dir(run_id: str) -> Path:
    root = RUNS_BASE_DIR / str(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_dir(run_id: str) -> Path:
    return ensure_run_dir(run_id)


def run_state_path(run_id: str) -> Path:
    return ensure_run_dir(run_id) / "state.json"


def run_events_path(run_id: str) -> Path:
    return ensure_run_dir(run_id) / "events.jsonl"


def run_tool_calls_path(run_id: str) -> Path:
    return ensure_run_dir(run_id) / "tool_calls.jsonl"


def write_run_state(run_id: str, payload: dict[str, Any]) -> Path:
    path = run_state_path(run_id)
    safe_write_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        logger=logger,
        module=__name__,
        operation="write run state",
    )
    return path


def read_run_state(run_id: str) -> dict[str, Any]:
    path = run_state_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Run `{run_id}` does not exist.")
    text = safe_read_text(
        path,
        fallback="",
        logger=logger,
        module=__name__,
        operation="read run state",
    )
    if not text:
        return {}
    payload = safe_json_loads(
        text,
        fallback={},
        logger=logger,
        module=__name__,
        operation="parse run state json",
        target=str(path),
    )
    return payload if isinstance(payload, dict) else {}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    safe_append_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False) + "\n",
        logger=logger,
        module=__name__,
        operation="append jsonl payload",
    )


def append_run_event(run_id: str, payload: dict[str, Any]) -> None:
    if not run_id:
        return
    event = dict(payload)
    event.setdefault("observed_at", time.time())
    append_jsonl(run_events_path(run_id), event)


def append_tool_call(run_id: str, payload: dict[str, Any]) -> None:
    if not run_id:
        return
    tool_call = dict(payload)
    tool_call.setdefault("observed_at", time.time())
    append_jsonl(run_tool_calls_path(run_id), tool_call)


def write_run_artifact(run_id: str, filename: str, content: str) -> ArtifactRef:
    root = ensure_run_dir(run_id)
    path = root / filename
    safe_write_text(
        path,
        content,
        logger=logger,
        module=__name__,
        operation="write run artifact",
    )
    return ArtifactRef(name=path.name, path=str(path), kind=path.suffix.lstrip(".") or "text")


def list_run_artifacts(run_id: str) -> list[dict[str, Any]]:
    root = ensure_run_dir(run_id)
    refs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        refs.append(
            ArtifactRef(
                name=path.name,
                path=str(path),
                kind=path.suffix.lstrip(".") or "text",
            ).to_dict()
        )
    return refs


class HookRegistry:
    def __init__(self) -> None:
        self._hooks: list[RegisteredHook] = []

    def register(
        self,
        hook_name: HookName | str,
        handler: HookHandler,
        *,
        priority: int = 0,
        hook_kind: HookKind = "observer",
        agent: str = "",
        route: str = "",
        tool_name: str = "",
        blocking: bool = False,
        overwrite: bool = False,
    ) -> RegisteredHook:
        registration = RegisteredHook(
            hook_name=str(hook_name or "").strip(),
            handler=handler,
            priority=int(priority or 0),
            hook_kind=hook_kind,
            agent=str(agent or "").strip(),
            route=str(route or "").strip(),
            tool_name=str(tool_name or "").strip(),
            blocking=bool(blocking),
        )
        if overwrite:
            self._hooks = [
                item
                for item in self._hooks
                if not self._same_registration(item, registration)
            ]
        self._hooks.append(registration)
        return registration

    def clear(self) -> None:
        self._hooks.clear()

    def list_hooks(self) -> list[RegisteredHook]:
        return list(self._hooks)

    def emit(self, context: HookContext) -> HookResponse:
        matched_hooks = self._matched_hooks(context)
        aggregated = HookResponse()
        failures: list[dict[str, str]] = []
        if not matched_hooks:
            return aggregated

        for hook in matched_hooks:
            try:
                response = self._normalize_response(hook.handler(context))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Hook `%s` failed: %s", context.hook_name, exc)
                failures.append({"hook_name": hook.handler_name, "error": str(exc)})
                continue

            aggregated = self._merge_response(aggregated, response, hook)

        self._audit_emission(context, aggregated, matched_hooks, failures)
        return aggregated

    def _matched_hooks(self, context: HookContext) -> list[RegisteredHook]:
        matched = [
            hook
            for hook in self._hooks
            if hook.hook_name == context.hook_name
            and (not hook.agent or hook.agent == context.agent)
            and (not hook.route or hook.route == context.route)
            and (not hook.tool_name or hook.tool_name == context.tool_name)
        ]
        return sorted(matched, key=lambda item: item.priority, reverse=True)

    @staticmethod
    def _normalize_response(value: HookResponse | dict[str, Any] | None) -> HookResponse:
        if value is None:
            return HookResponse()
        if isinstance(value, HookResponse):
            return value
        if isinstance(value, dict):
            decision = str(value.get("decision", "continue") or "continue").strip().lower()
            if decision not in {"approve", "block", "continue"}:
                decision = "continue"
            return HookResponse(
                decision=decision,  # type: ignore[arg-type]
                updated_payload=dict(value.get("updated_payload", {}) or {}),
                additional_context=dict(value.get("additional_context", {}) or {}),
                message=str(value.get("message", "") or ""),
                metadata=dict(value.get("metadata", {}) or {}),
            )
        return HookResponse()

    @staticmethod
    def _merge_response(
        current: HookResponse,
        incoming: HookResponse,
        hook: RegisteredHook,
    ) -> HookResponse:
        current_decision = current.decision
        if hook.hook_kind == "policy":
            if incoming.decision == "block":
                current_decision = "block"
            elif incoming.decision == "approve" and current_decision != "block":
                current_decision = "approve"

        updated_payload = dict(current.updated_payload)
        updated_payload.update(incoming.updated_payload)

        additional_context = dict(current.additional_context)
        additional_context.update(incoming.additional_context)

        metadata = dict(current.metadata)
        metadata.update(incoming.metadata)

        message = incoming.message or current.message

        if hook.hook_kind == "observer":
            updated_payload = dict(current.updated_payload)
            additional_context = dict(current.additional_context)
            current_decision = current.decision

        return HookResponse(
            decision=current_decision,
            updated_payload=updated_payload,
            additional_context=additional_context,
            message=message,
            metadata=metadata,
        )

    @staticmethod
    def _same_registration(left: RegisteredHook, right: RegisteredHook) -> bool:
        return (
            left.hook_name == right.hook_name
            and left.handler_name == right.handler_name
            and left.agent == right.agent
            and left.route == right.route
            and left.tool_name == right.tool_name
        )

    @staticmethod
    def _audit_emission(
        context: HookContext,
        aggregated: HookResponse,
        matched_hooks: list[RegisteredHook],
        failures: list[dict[str, str]],
    ) -> None:
        if not context.run_id:
            return

        payload = {
            "hook_name": context.hook_name,
            "hook_names": [hook.handler_name for hook in matched_hooks],
            "hook_count": len(matched_hooks),
            "decision": aggregated.decision,
            "hook_message": aggregated.message,
            "hook_metadata": dict(aggregated.metadata),
            "hook_additional_context": dict(aggregated.additional_context),
            "hook_errors": failures,
            **dict(aggregated.metadata),
        }
        status = "blocked" if aggregated.decision == "block" else "completed"

        if context.hook_name in {"before_tool_call", "after_tool_call"}:
            append_tool_call(
                context.run_id,
                {
                    **context.to_dict(),
                    **payload,
                    "status": status,
                },
            )
            return

        event_type = "terminal" if context.hook_name == "on_terminal" else "hook"
        append_run_event(
            context.run_id,
            {
                "type": event_type,
                "route": context.route,
                "agent": context.agent,
                "tool": context.tool_name,
                "stage": context.hook_name,
                "status": status,
                "summary": aggregated.message or f"Hook `{context.hook_name}` executed.",
                **payload,
            },
        )


DEFAULT_HOOK_REGISTRY = HookRegistry()


def register_hook(
    hook_name: HookName | str,
    handler: HookHandler,
    *,
    priority: int = 0,
    hook_kind: HookKind = "observer",
    agent: str = "",
    route: str = "",
    tool_name: str = "",
    blocking: bool = False,
    overwrite: bool = False,
) -> RegisteredHook:
    return DEFAULT_HOOK_REGISTRY.register(
        hook_name,
        handler,
        priority=priority,
        hook_kind=hook_kind,
        agent=agent,
        route=route,
        tool_name=tool_name,
        blocking=blocking,
        overwrite=overwrite,
    )


def emit_hook(context: HookContext) -> HookResponse:
    return DEFAULT_HOOK_REGISTRY.emit(context)


def clear_hooks() -> None:
    DEFAULT_HOOK_REGISTRY.clear()


def list_registered_hooks() -> list[RegisteredHook]:
    return DEFAULT_HOOK_REGISTRY.list_hooks()

from __future__ import annotations

import uuid
from typing import Any

from codesage.agents.fork.fork_models import ForkResult, ForkTaskSpec
from codesage.agents.fork.fork_policy import build_fork_payload, capability_scope_for_task, validate_fork_request
from codesage.agents.framework.factory import AgentFactory
from codesage.agents.framework.lifecycle import AgentHandle, AgentLifecycleManager, normalize_result_status
from codesage.agents.framework.specs import AgentSpec, normalize_agent_name
from codesage.core.runtime import HookContext, HookResponse, emit_hook
from codesage.core.observability import link_observed_runs, record_observed_event, start_observed_run
from codesage.tools.agent_tool_registry import AgentToolRegistry, DEFAULT_AGENT_TOOL_REGISTRY


class AgentManager:
    def __init__(
        self,
        *,
        tool_registry: AgentToolRegistry | None = None,
        lifecycle: AgentLifecycleManager | None = None,
    ) -> None:
        self.tool_registry = tool_registry or DEFAULT_AGENT_TOOL_REGISTRY
        self.lifecycle = lifecycle or AgentLifecycleManager()
        self._specs: dict[str, AgentSpec] = {}
        self._factories: dict[str, AgentFactory] = {}
        self._singleton_runs: dict[str, str] = {}
        self._session_runs: dict[tuple[str, str], str] = {}
        self._fork_results: dict[str, ForkResult] = {}

    def register_spec(self, spec: AgentSpec, *, overwrite: bool = False) -> AgentSpec:
        existing = self._specs.get(spec.name)
        if existing is not None and not overwrite:
            raise ValueError(f"Agent spec {spec.name!r} is already registered.")
        self._specs[spec.name] = spec
        return spec

    def register_factory(
        self,
        factory_name: str,
        factory: AgentFactory,
        *,
        overwrite: bool = False,
    ) -> AgentFactory:
        normalized_factory_name = normalize_agent_name(factory_name)
        existing = self._factories.get(normalized_factory_name)
        if existing is not None and not overwrite:
            raise ValueError(f"Agent factory {normalized_factory_name!r} is already registered.")
        self._factories[normalized_factory_name] = factory
        return factory

    def get_spec(self, agent_name: str) -> AgentSpec:
        normalized_agent_name = normalize_agent_name(agent_name)
        spec = self._specs.get(normalized_agent_name)
        if spec is None:
            raise KeyError(f"Unknown agent {normalized_agent_name!r}.")
        return spec

    def get_factory(self, factory_name: str) -> AgentFactory:
        normalized_factory_name = normalize_agent_name(factory_name)
        factory = self._factories.get(normalized_factory_name)
        if factory is None:
            raise KeyError(f"Unknown agent factory {normalized_factory_name!r}.")
        return factory

    def list_specs(self) -> list[AgentSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def create_agent(
        self,
        agent_name: str,
        *,
        session_id: str = "",
        context: dict[str, Any] | None = None,
        run_id: str = "",
        parent_run_id: str = "",
        task_id: str = "",
        task_type: str = "",
        child_agent_mode: str = "",
        fork_reason: str = "",
        capability_scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentHandle:
        spec = self.get_spec(agent_name)
        cached = self._get_cached_handle(spec, session_id=session_id)
        if cached is not None:
            return cached

        create_payload = {
            "session_id": str(session_id or "").strip(),
            "context": dict(context or {}),
            "run_id": str(run_id or "").strip(),
            "parent_run_id": str(parent_run_id or "").strip(),
            "task_id": str(task_id or "").strip(),
            "task_type": str(task_type or "").strip(),
            "child_agent_mode": str(child_agent_mode or "").strip(),
            "fork_reason": str(fork_reason or "").strip(),
            "capability_scope": dict(capability_scope or {}),
            "metadata": dict(metadata or {}),
        }
        before_create = emit_hook(
            HookContext(
                hook_name="before_agent_create",
                route=create_payload["task_type"],
                agent=spec.name,
                run_id=create_payload["run_id"],
                action="create_agent",
                payload=create_payload,
            )
        )
        if before_create.additional_context:
            create_payload["context"].update(before_create.additional_context)
        if before_create.updated_payload:
            for key, value in before_create.updated_payload.items():
                if key == "context" and isinstance(value, dict):
                    create_payload["context"].update(dict(value))
                elif key in create_payload:
                    create_payload[key] = value
        if before_create.decision == "block":
            raise RuntimeError(before_create.message or f"Agent `{spec.name}` 的创建被 Hook 阻止。")

        factory = self.get_factory(spec.factory_name)
        instance = factory.create(
            spec,
            tools=self._resolve_tools(spec),
            context=dict(create_payload["context"]),
        )
        handle = AgentHandle(
            run_id=str(create_payload["run_id"] or uuid.uuid4().hex),
            spec=spec,
            instance=instance,
            session_id=str(create_payload["session_id"]),
            parent_run_id=str(create_payload["parent_run_id"]),
            task_id=str(create_payload["task_id"]),
            task_type=str(create_payload["task_type"]),
            child_agent_mode=str(create_payload["child_agent_mode"]),
            fork_reason=str(create_payload["fork_reason"]),
            capability_scope=dict(create_payload["capability_scope"]),
            metadata=dict(create_payload["metadata"]),
        )
        self.lifecycle.add(handle)
        self.lifecycle.transition(handle.run_id, "ready")
        self._cache_handle(handle)
        after_create = emit_hook(
            HookContext(
                hook_name="after_agent_create",
                route=handle.task_type,
                agent=handle.agent_name,
                run_id=handle.run_id,
                action="create_agent",
                payload=create_payload,
                result=handle.snapshot(),
                metadata={"status": handle.status},
            )
        )
        if after_create.metadata:
            handle.metadata.update(after_create.metadata)
        return handle

    def invoke(
        self,
        agent_name: str,
        payload: dict[str, Any],
        *,
        session_id: str = "",
        context: dict[str, Any] | None = None,
        run_id: str = "",
    ) -> Any:
        handle = self.create_agent(
            agent_name,
            session_id=session_id,
            context=context,
            run_id=run_id,
        )
        before_invoke = emit_hook(
            HookContext(
                hook_name="before_agent_invoke",
                route=handle.task_type,
                agent=handle.agent_name,
                run_id=handle.run_id,
                action="invoke",
                payload=dict(payload or {}),
                metadata={"session_id": session_id, "context": dict(context or {})},
            )
        )
        effective_payload = dict(payload or {})
        if before_invoke.updated_payload:
            effective_payload.update(before_invoke.updated_payload)
        if before_invoke.decision == "block":
            blocked = self._build_blocked_result(before_invoke)
            self.lifecycle.transition(
                handle.run_id,
                "failed",
                error=blocked["error"],
                result=blocked,
                metadata=dict(before_invoke.metadata),
            )
            return blocked

        self.lifecycle.transition(handle.run_id, "running", metadata=dict(before_invoke.metadata))
        try:
            result = handle.instance.invoke(effective_payload)
        except Exception as exc:
            self.lifecycle.transition(handle.run_id, "failed", error=str(exc))
            raise

        self.lifecycle.transition(
            handle.run_id,
            normalize_result_status(result),
            result=result,
        )
        after_invoke = emit_hook(
            HookContext(
                hook_name="after_agent_invoke",
                route=handle.task_type,
                agent=handle.agent_name,
                run_id=handle.run_id,
                action="invoke",
                payload=effective_payload,
                result=result,
                metadata={"status": normalize_result_status(result)},
            )
        )
        if after_invoke.metadata:
            handle.metadata.update(after_invoke.metadata)
        return result

    def stream(
        self,
        agent_name: str,
        payload: dict[str, Any],
        *,
        session_id: str = "",
        context: dict[str, Any] | None = None,
        run_id: str = "",
        config: Any | None = None,
        stream_mode: str = "values",
    ) -> Any:
        handle = self.create_agent(
            agent_name,
            session_id=session_id,
            context=context,
            run_id=run_id,
        )
        before_stream = emit_hook(
            HookContext(
                hook_name="before_agent_stream",
                route=handle.task_type,
                agent=handle.agent_name,
                run_id=handle.run_id,
                action="stream",
                payload=dict(payload or {}),
                metadata={"session_id": session_id, "context": dict(context or {})},
            )
        )
        effective_payload = dict(payload or {})
        if before_stream.updated_payload:
            effective_payload.update(before_stream.updated_payload)
        if before_stream.decision == "block":
            blocked = self._build_blocked_result(before_stream)
            self.lifecycle.transition(
                handle.run_id,
                "failed",
                error=blocked["error"],
                result=blocked,
                metadata=dict(before_stream.metadata),
            )
            return [blocked]

        self.lifecycle.transition(handle.run_id, "running", metadata=dict(before_stream.metadata))
        try:
            result = handle.instance.stream(
                effective_payload,
                config=config,
                stream_mode=stream_mode,
            )
        except Exception as exc:
            self.lifecycle.transition(handle.run_id, "failed", error=str(exc))
            raise
        self.lifecycle.transition(
            handle.run_id,
            normalize_result_status(result),
            result=result,
        )
        after_stream = emit_hook(
            HookContext(
                hook_name="after_agent_stream",
                route=handle.task_type,
                agent=handle.agent_name,
                run_id=handle.run_id,
                action="stream",
                payload=effective_payload,
                result=result,
                metadata={"status": normalize_result_status(result)},
            )
        )
        if after_stream.metadata:
            handle.metadata.update(after_stream.metadata)
        return result

    def cancel(self, run_id: str) -> AgentHandle:
        handle = self.lifecycle.get(run_id)
        handle.instance.cancel()
        return self.lifecycle.transition(handle.run_id, "cancelled")

    def dispose(self, run_id: str) -> AgentHandle:
        handle = self.lifecycle.get(run_id)
        handle.instance.dispose()
        disposed_handle = self.lifecycle.transition(handle.run_id, "disposed")
        self._drop_cache(disposed_handle)
        return disposed_handle

    def get_handle(self, run_id: str) -> AgentHandle:
        return self.lifecycle.get(run_id)

    def get_snapshot(self, run_id: str) -> dict[str, Any]:
        return self.lifecycle.snapshot(run_id)

    def fork(
        self,
        task: ForkTaskSpec,
        *,
        thread_id: str = "",
        parent_run_id: str = "",
    ) -> AgentHandle:
        effective_parent_run_id = str(parent_run_id or task.parent_run_id or "").strip()
        effective_task = task
        effective_thread_id = str(thread_id or "").strip()
        before_fork = emit_hook(
            HookContext(
                hook_name="before_fork",
                route=task.task_type,
                agent=task.child_agent_name,
                run_id=effective_parent_run_id,
                action="fork",
                payload={
                    "task": task.to_dict(),
                    "thread_id": effective_thread_id,
                    "parent_run_id": effective_parent_run_id,
                },
                metadata={"fork_reason": task.fork_reason},
            )
        )
        if before_fork.updated_payload:
            updated_task = before_fork.updated_payload.get("task")
            if isinstance(updated_task, dict):
                effective_task = ForkTaskSpec(**updated_task)
            effective_thread_id = str(
                before_fork.updated_payload.get("thread_id", effective_thread_id) or effective_thread_id
            )
            effective_parent_run_id = str(
                before_fork.updated_payload.get("parent_run_id", effective_parent_run_id)
                or effective_parent_run_id
            )
        if before_fork.decision == "block":
            raise RuntimeError(before_fork.message or "Fork 请求被 Hook 策略阻止。")

        parent_handle = self.get_handle(effective_parent_run_id) if effective_parent_run_id else None
        existing_child_count = (
            len(parent_handle.metadata.get("child_run_ids", [])) if parent_handle is not None else 0
        )
        validate_fork_request(parent_handle, effective_task, existing_child_count=existing_child_count)

        child_run_id = uuid.uuid4().hex
        capability_scope = capability_scope_for_task(effective_task)
        child_payload = build_fork_payload(effective_task, thread_id=effective_thread_id)

        start_observed_run(
            child_run_id,
            source="fork_child",
            thread_id=effective_thread_id,
            route=effective_task.task_type,
            mode=effective_task.child_agent_mode,
            agent=effective_task.child_agent_name,
            summary=effective_task.goal,
            parent_run_id=effective_parent_run_id,
            trigger_stage="fork",
        )
        link_observed_runs(
            effective_parent_run_id,
            child_run_id,
            trigger_stage="fork",
            summary=effective_task.goal,
        )
        record_observed_event(
            effective_parent_run_id,
            {
                "type": "fork_task_created",
                "thread_id": effective_thread_id,
                "stage": "fork",
                "route": effective_task.task_type,
                "mode": effective_task.child_agent_mode,
                "agent": effective_task.parent_agent,
                "status": "running",
                "summary": effective_task.goal,
                "task_id": effective_task.task_id,
                "parent_run_id": effective_parent_run_id,
                "child_run_id": child_run_id,
                "child_agent_name": effective_task.child_agent_name,
                "child_agent_mode": effective_task.child_agent_mode,
                "task_type": effective_task.task_type,
                "fork_reason": effective_task.fork_reason,
            },
        )
        record_observed_event(
            child_run_id,
            {
                "type": "fork_task_started",
                "thread_id": effective_thread_id,
                "stage": "fork",
                "route": effective_task.task_type,
                "mode": effective_task.child_agent_mode,
                "agent": effective_task.child_agent_name,
                "status": "running",
                "summary": effective_task.goal,
                "task_id": effective_task.task_id,
                "parent_run_id": effective_parent_run_id,
                "child_run_id": child_run_id,
                "child_agent_name": effective_task.child_agent_name,
                "child_agent_mode": effective_task.child_agent_mode,
                "task_type": effective_task.task_type,
                "fork_reason": effective_task.fork_reason,
            },
        )

        handle = self.create_agent(
            effective_task.child_agent_name,
            run_id=child_run_id,
            context=child_payload,
            parent_run_id=effective_parent_run_id,
            task_id=effective_task.task_id,
            task_type=effective_task.task_type,
            child_agent_mode=effective_task.child_agent_mode,
            fork_reason=effective_task.fork_reason,
            capability_scope=capability_scope,
            metadata={
                "thread_id": effective_thread_id,
                "child_run_ids": [],
                "parent_agent": effective_task.parent_agent,
                "goal": effective_task.goal,
            },
        )
        self.lifecycle.transition(
            handle.run_id,
            "running",
            metadata={
                "task_id": effective_task.task_id,
                "thread_id": effective_thread_id,
                **dict(before_fork.metadata),
            },
        )
        if parent_handle is not None:
            child_run_ids = list(parent_handle.metadata.get("child_run_ids", []) or [])
            if handle.run_id not in child_run_ids:
                child_run_ids.append(handle.run_id)
            self.lifecycle.transition(
                parent_handle.run_id,
                parent_handle.status,
                metadata={"child_run_ids": child_run_ids},
            )

        try:
            result = self._invoke_fork_task(handle, effective_task)
        except Exception as exc:
            self.lifecycle.transition(handle.run_id, "failed", error=str(exc))
            record_observed_event(
                child_run_id,
                {
                    "type": "fork_task_failed",
                    "thread_id": effective_thread_id,
                    "stage": "fork",
                    "route": effective_task.task_type,
                    "mode": effective_task.child_agent_mode,
                    "agent": effective_task.child_agent_name,
                    "status": "error",
                    "summary": effective_task.goal,
                    "task_id": effective_task.task_id,
                    "parent_run_id": effective_parent_run_id,
                    "child_run_id": child_run_id,
                    "child_agent_name": effective_task.child_agent_name,
                    "child_agent_mode": effective_task.child_agent_mode,
                    "task_type": effective_task.task_type,
                    "fork_reason": effective_task.fork_reason,
                    "content": str(exc),
                },
            )
            raise

        standardized = self._standardize_fork_result(effective_task, handle.run_id, result)
        self._fork_results[handle.run_id] = standardized
        self.lifecycle.transition(
            handle.run_id,
            normalize_result_status(standardized),
            result=standardized.to_dict(),
        )
        record_observed_event(
            child_run_id,
            {
                "type": "fork_task_completed",
                "thread_id": effective_thread_id,
                "stage": "fork",
                "route": effective_task.task_type,
                "mode": effective_task.child_agent_mode,
                "agent": effective_task.child_agent_name,
                "status": standardized.status,
                "summary": standardized.summary,
                "task_id": effective_task.task_id,
                "parent_run_id": effective_parent_run_id,
                "child_run_id": child_run_id,
                "child_agent_name": effective_task.child_agent_name,
                "child_agent_mode": effective_task.child_agent_mode,
                "task_type": effective_task.task_type,
                "fork_reason": effective_task.fork_reason,
            },
        )
        after_fork = emit_hook(
            HookContext(
                hook_name="after_fork",
                route=effective_task.task_type,
                agent=effective_task.child_agent_name,
                run_id=child_run_id,
                action="fork",
                payload={
                    "task": effective_task.to_dict(),
                    "thread_id": effective_thread_id,
                    "parent_run_id": effective_parent_run_id,
                },
                result=standardized.to_dict(),
                metadata={"status": standardized.status},
            )
        )
        if after_fork.metadata:
            handle.metadata.update(after_fork.metadata)
        return handle

    def collect_fork_result(self, child_run_id: str) -> ForkResult:
        child_handle = self.get_handle(child_run_id)
        cached = self._fork_results.get(child_handle.run_id)
        if cached is not None:
            return cached
        if child_handle.last_result is None:
            raise KeyError(f"No fork result found for child run {child_run_id!r}.")
        task = ForkTaskSpec(
            task_id=child_handle.task_id,
            parent_run_id=child_handle.parent_run_id,
            parent_agent=str(child_handle.metadata.get("parent_agent", "") or ""),
            child_agent_name=child_handle.agent_name,
            child_agent_mode=str(child_handle.child_agent_mode or "workflow") or "workflow",
            task_type=str(child_handle.task_type or "fork_task") or "fork_task",
            goal=str(child_handle.metadata.get("goal", "") or "") or child_handle.agent_name,
            allowed_tools=tuple(child_handle.capability_scope.get("allowed_tools", [])),
            allowed_actions=tuple(child_handle.capability_scope.get("allowed_actions", [])),
        )
        standardized = self._standardize_fork_result(task, child_handle.run_id, child_handle.last_result)
        self._fork_results[child_handle.run_id] = standardized
        return standardized

    def run_fork_task(
        self,
        task: ForkTaskSpec,
        *,
        thread_id: str = "",
        parent_run_id: str = "",
    ) -> ForkResult:
        handle = self.fork(task, thread_id=thread_id, parent_run_id=parent_run_id)
        return self.collect_fork_result(handle.run_id)

    @staticmethod
    def _build_blocked_result(hook_response: HookResponse) -> dict[str, Any]:
        message = hook_response.message or "请求被 Hook 策略阻止。"
        return {
            "status": "failed",
            "error": message,
            "hook_blocked": True,
            "hook_decision": hook_response.decision,
            "hook_metadata": dict(hook_response.metadata),
            "additional_context": dict(hook_response.additional_context),
        }

    def _resolve_tools(self, spec: AgentSpec) -> list[Any]:
        tools = self.tool_registry.get_tools_for_agent(spec.name)
        if tools:
            return tools

        resolved_tools: list[Any] = []
        for tool_name in spec.default_tools:
            resolved_tools.append(self.tool_registry.get_tool(tool_name))
        return resolved_tools

    def _get_cached_handle(self, spec: AgentSpec, *, session_id: str) -> AgentHandle | None:
        if spec.scope == "singleton":
            run_id = self._singleton_runs.get(spec.name)
            if run_id:
                handle = self.lifecycle.get(run_id)
                if not handle.is_terminal:
                    return handle
            return None

        if spec.scope == "session":
            session_key = (spec.name, str(session_id or "").strip())
            run_id = self._session_runs.get(session_key)
            if run_id:
                handle = self.lifecycle.get(run_id)
                if not handle.is_terminal:
                    return handle
        return None

    def _cache_handle(self, handle: AgentHandle) -> None:
        if handle.spec.scope == "singleton":
            self._singleton_runs[handle.agent_name] = handle.run_id
            return
        if handle.spec.scope == "session":
            self._session_runs[(handle.agent_name, handle.session_id)] = handle.run_id

    def _drop_cache(self, handle: AgentHandle) -> None:
        if handle.spec.scope == "singleton":
            self._singleton_runs.pop(handle.agent_name, None)
            return
        if handle.spec.scope == "session":
            self._session_runs.pop((handle.agent_name, handle.session_id), None)

    @staticmethod
    def _invoke_fork_task(handle: AgentHandle, task: ForkTaskSpec) -> Any:
        if hasattr(handle.instance, "invoke_task"):
            return handle.instance.invoke_task(task)
        return handle.instance.invoke({"fork_task": task.to_dict()})

    @staticmethod
    def _standardize_fork_result(task: ForkTaskSpec, child_run_id: str, result: Any) -> ForkResult:
        if isinstance(result, ForkResult):
            return ForkResult(
                task_id=result.task_id or task.task_id,
                child_run_id=result.child_run_id or child_run_id,
                child_agent_name=result.child_agent_name or task.child_agent_name,
                status=result.status,
                summary=result.summary,
                result_type=result.result_type,
                result_payload=result.result_payload,
                artifacts=result.artifacts,
                started_at=result.started_at,
                finished_at=result.finished_at,
                error=result.error,
            )
        if isinstance(result, dict):
            return ForkResult(
                task_id=str(result.get("task_id", "") or task.task_id),
                child_run_id=str(result.get("child_run_id", "") or child_run_id),
                child_agent_name=str(result.get("child_agent_name", "") or task.child_agent_name),
                status=str(result.get("status", "completed") or "completed").lower(),
                summary=str(result.get("summary", "") or result.get("output_result", "") or task.goal),
                result_type=str(result.get("result_type", "structured") or "structured"),
                result_payload=dict(result.get("result_payload", result) or {}),
                artifacts=list(result.get("artifacts", []) or []),
                error=str(result.get("error", "") or ""),
            )
        return ForkResult(
            task_id=task.task_id,
            child_run_id=child_run_id,
            child_agent_name=task.child_agent_name,
            status="completed",
            summary=str(result or task.goal),
            result_type="text",
            result_payload={"value": result},
        )


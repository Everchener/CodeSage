"""CodeSage Supervisor 路由与调用协调模块。"""

from __future__ import annotations

import re
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from codesage.agents.framework.bootstrap import get_default_agent_manager
from codesage.agents.fork.fork_models import ForkTaskSpec
from codesage.core.runtime import HookContext, emit_hook
from codesage.skills.registry import build_skill_execution_task, resolve_skill_execution
from codesage.tools.agent_tool_registry import DEFAULT_AGENT_TOOL_REGISTRY
from codesage.tools.llm_tools import call_llm_json
from codesage.tools.prompt_tools import build_json_output, build_list, build_prompt, build_system_prompt

ProgressCallback = Callable[[dict[str, Any]], None]

ROUTER_SYSTEM_PROMPT = build_system_prompt(
    role="你是 CodeSage 的路由协调助手，负责把用户请求分配给唯一且最合适的能力路径。",
    responsibilities=[
        "区分代码审查、代码问答、代码修改、索引协助和无需处理的请求。",
        "在明显可判定时优先使用快速规则路由，减少不必要的模型开销。",
        "输出稳定、可解析的路由结果，供 supervisor 工作流继续执行。",
    ],
    rules=[
        "同一请求只能返回一个 route。",
        "涉及 PR、diff 或代码审查的请求优先路由到 review。",
        "涉及修改、修复、重构、实现的请求优先路由到 modify。",
        "以仓库问答、定位代码、解释实现为主的请求路由到 rag。",
        "与建立索引或上传文档相关的请求路由到 index。",
        "闲聊、无效输入或无需调用 CodeSage 能力时路由到 none。",
    ],
    output_instruction="只返回合法 JSON，不要输出 Markdown、解释或思考过程。",
)


@dataclass(frozen=True)
class RouteDecision:
    """描述一次路由决策。"""

    route: str
    mode: str
    reason: str
    target_agent: str
    context_policy: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    hook_response = emit_hook(
        HookContext(
            hook_name="on_progress",
            route=str(payload.get("route", "") or ""),
            agent=str(payload.get("agent", "") or ""),
            run_id=str(payload.get("run_id", "") or ""),
            action="progress",
            payload=dict(payload),
            metadata={"status": str(payload.get("status", "") or "")},
        )
    )
    if hook_response.metadata:
        payload.update(hook_response.metadata)
    if progress_callback is not None:
        progress_callback(payload)


def _apply_tool_before_hook(*, tool_name: str, payload: dict[str, Any], agent: str = "supervisor_agent") -> tuple[dict[str, Any], str]:
    hook_response = emit_hook(
        HookContext(
            hook_name="before_tool_call",
            route="tool",
            agent=agent,
            tool_name=tool_name,
            action="tool_call",
            payload=payload,
        )
    )
    effective_payload = dict(payload)
    if hook_response.updated_payload:
        effective_payload.update(hook_response.updated_payload)
    if hook_response.decision == "block":
        return effective_payload, hook_response.message or "请求被 Hook 策略阻止。"
    return effective_payload, ""


def _emit_tool_after_hook(*, tool_name: str, payload: dict[str, Any], result: Any, agent: str = "supervisor_agent") -> None:
    emit_hook(
        HookContext(
            hook_name="after_tool_call",
            route="tool",
            agent=agent,
            tool_name=tool_name,
            action="tool_call",
            payload=payload,
            result=result,
        )
    )


def _invoke_with_supported_kwargs(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return fn(*args, **supported)


def _clean_thinking(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    for pattern in (
        r"<think>.*?</think>",
        r"<\|ankton\w*\|>.*?<\|/ankton\|>",
        r"<\|thinking\|>.*?<\|end\|>",
        r"<\|thought\|>.*?<\|end\|>",
    ):
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _safe_extract_result(result: Any) -> str:
    if result is None:
        return "未获得结果。"
    if isinstance(result, str):
        return _clean_thinking(result)
    if isinstance(result, dict):
        if result.get("status") == "cancelled":
            return _clean_thinking(str(result.get("output_result") or result.get("error") or "请求已取消。"))
        if result.get("status") == "awaiting_confirmation":
            pending_changes = result.get("pending_changes") or []
            risk_reasons = result.get("risk_reasons") or []
            pending_text = ", ".join(pending_changes) if pending_changes else "暂无待确认改动"
            risk_text = "；".join(risk_reasons) if risk_reasons else "未提供风险原因"
            return f"修改预览已生成，等待确认。\n\n待确认改动：{pending_text}\n\n风险原因：{risk_text}"
        if result.get("output_result"):
            return _clean_thinking(str(result["output_result"]))
        changes = result.get("applied_changes") or result.get("changes_made") or []
        verification = result.get("verification_result", "")
        if changes or verification:
            changes_text = ", ".join(changes) if changes else "未记录改动"
            verification_text = verification or "未提供验证结果"
            return f"代码修改已完成。\n\n改动：{changes_text}\n\n验证结果：\n{verification_text}"
        for field_name in ("output", "final_answer", "response", "result"):
            if result.get(field_name):
                return _clean_thinking(str(result[field_name]))
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content") and last_msg.content:
                return _clean_thinking(str(last_msg.content))
    return _clean_thinking(str(result))


def _build_route_decision(*, route: str, mode: str, reason: str, target_agent: str, context_policy: str, tool_name: str | None, tool_args: dict[str, Any], summary: str) -> RouteDecision:
    return RouteDecision(route=route, mode=mode, reason=reason, target_agent=target_agent, context_policy=context_policy, tool_name=tool_name, tool_args=tool_args, summary=summary)


def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _fast_lane_decision(user_input: str) -> RouteDecision | None:
    lowered = re.sub(r"\s+", " ", user_input.strip().lower())
    if not lowered:
        return None

    if lowered.startswith("diff --git") or ("github.com/" in lowered and "/pull/" in lowered):
        return _build_route_decision(route="review", mode="fast", reason="识别到原始 diff 或 GitHub PR 链接。", target_agent="pr_review_agent", context_policy="review", tool_name="review_code_with_codesage", tool_args={"request": user_input}, summary="快速路由命中 PR 审查请求。")

    if _starts_with_any(lowered, ("review ", "review:", "review this", "代码审查", "审查一下", "pr review")):
        return _build_route_decision(route="review", mode="fast", reason="识别到明确的代码审查指令。", target_agent="pr_review_agent", context_policy="review", tool_name="review_code_with_codesage", tool_args={"request": user_input}, summary="快速路由命中显式审查请求。")

    if _starts_with_any(lowered, ("fix ", "modify ", "refactor ", "implement ", "add ", "update ", "create ", "change ", "修复", "修改", "重构", "实现", "新增", "更新")):
        return _build_route_decision(route="modify", mode="fast", reason="识别到明确的代码修改指令。", target_agent="code_modify_agent", context_policy="modify", tool_name="modify_code", tool_args={"instruction": user_input, "working_dir": "."}, summary="快速路由命中代码修改任务。")

    index_cues = ("index repo", "index repository", "index docs", "upload docs", "upload document", "建立索引", "索引仓库", "索引文档", "上传文档")
    if _contains_any(lowered, index_cues):
        return _build_route_decision(route="index", mode="fast", reason="识别到索引或文档上传相关意图。", target_agent="index_helper", context_policy="none", tool_name=None, tool_args={}, summary="快速路由命中索引相关请求。")

    rag_prefixes = ("where is", "what is", "how does", "how do", "why does", "explain ", "show me", "在哪里", "是什么", "怎么", "为什么", "解释", "看看")
    rag_cues = (" where ", "?", "？", "在哪里", "是什么", "为什么", "怎么")
    has_rag_shape = _starts_with_any(lowered, rag_prefixes) or _contains_any(lowered, rag_cues)
    is_code_change_like = _contains_any(lowered, ("fix ", "modify ", "refactor ", "修复", "修改", "重构"))
    is_index_like = _contains_any(lowered, index_cues)
    if has_rag_shape and not is_code_change_like and not is_index_like:
        return _build_route_decision(route="rag", mode="fast", reason="识别到仓库问答或代码解释请求。", target_agent="enhanced_rag_agent", context_policy="rag", tool_name="answer_with_rag", tool_args={"question": user_input}, summary="快速路由命中代码库问答请求。")

    return None


def _llm_route_decision(user_input: str) -> RouteDecision:
    prompt = build_prompt(
        task="将用户请求准确路由到一条 CodeSage 能力路径。",
        context_sections=[
            ("可选路由", build_list(["review：处理 Pull Request、diff 和代码审查请求。", "rag：回答仓库问题、解释实现、定位代码。", "modify：执行代码修改、修复、重构或实现任务。", "index：指导用户建立索引或上传文档。", "none：不需要调用 CodeSage 能力。"])) ,
            ("用户请求", user_input),
        ],
        rules=[
            "如果请求同时包含多个意图，选择最直接的主要意图。",
            "只有当用户真的要修改代码时才选择 modify。",
            "只有当请求明显是审查 PR 或 diff 时才选择 review。",
            "仓库问答、解释、定位默认优先选择 rag。",
            "索引或上传文档相关请求选择 index。",
        ],
        examples=[
            ("https://github.com/acme/payments/pull/42", '{"route":"review","reason":"包含 GitHub PR 链接。","target_agent":"pr_review_agent","context_policy":"review","summary":"处理 PR 审查请求。"}'),
            ("帮我修复登录接口里的空指针问题", '{"route":"modify","reason":"明确要求修改代码并修复问题。","target_agent":"code_modify_agent","context_policy":"modify","summary":"执行代码修改任务。"}'),
            ("EnhancedRAGAgent 在哪里定义？", '{"route":"rag","reason":"这是仓库定位和解释问题。","target_agent":"enhanced_rag_agent","context_policy":"rag","summary":"执行仓库问答。"}'),
        ],
        output_format=build_json_output('{"route":"review|rag|modify|index|none","reason":"原因","target_agent":"agent_name","context_policy":"review|rag|modify|none","summary":"简要摘要"}', extra_rules=["只返回一个 JSON 对象。", "reason 和 summary 必须使用中文短句。"]),
    )
    payload = call_llm_json(prompt, system=ROUTER_SYSTEM_PROMPT, max_tokens=350)

    route = "rag"
    reason = "默认回退到仓库问答路径。"
    target_agent = "enhanced_rag_agent"
    context_policy = "rag"
    summary = "默认执行仓库问答。"
    if isinstance(payload, dict):
        route = str(payload.get("route", route)).strip().lower() or route
        reason = str(payload.get("reason", reason)).strip() or reason
        target_agent = str(payload.get("target_agent", target_agent)).strip() or target_agent
        context_policy = str(payload.get("context_policy", context_policy)).strip() or context_policy
        summary = str(payload.get("summary", summary)).strip() or summary

    tool_name_map = {"review": "review_code_with_codesage", "rag": "answer_with_rag", "modify": "modify_code", "index": None, "none": None}
    tool_args_map = {"review": {"request": user_input}, "rag": {"question": user_input}, "modify": {"instruction": user_input, "working_dir": "."}, "index": {}, "none": {}}
    safe_route = route if route in tool_args_map else "rag"
    safe_context_policy = context_policy if context_policy in {"review", "rag", "modify", "none"} else "rag"
    safe_target_agent = target_agent if target_agent else "enhanced_rag_agent"
    return _build_route_decision(route=safe_route, mode="slow", reason=reason, target_agent=safe_target_agent, context_policy=safe_context_policy, tool_name=tool_name_map.get(safe_route, "answer_with_rag"), tool_args=tool_args_map.get(safe_route, {"question": user_input}), summary=summary)


@tool
def review_code_with_codesage(request: str) -> str:
    """调用 PR 审查 Agent 处理 PR 或 diff 请求。"""
    effective_payload, blocked_message = _apply_tool_before_hook(
        tool_name="review_code_with_codesage",
        payload={"request": request},
    )
    if blocked_message:
        return blocked_message
    result = get_default_agent_manager().invoke(
        "pr_review_agent",
        {"messages": [HumanMessage(content=str(effective_payload.get("request", request)))]},
    )
    _emit_tool_after_hook(
        tool_name="review_code_with_codesage",
        payload=effective_payload,
        result=result,
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content") and last_msg.content:
            return _clean_thinking(str(last_msg.content))
    return _clean_thinking(str(result or ""))


@tool
def answer_with_rag(question: str) -> str:
    """调用 RAG Agent 回答仓库问题。"""
    effective_payload, blocked_message = _apply_tool_before_hook(
        tool_name="answer_with_rag",
        payload={"question": question},
    )
    if blocked_message:
        return blocked_message
    result = get_default_agent_manager().invoke(
        "enhanced_rag_agent",
        {"query": str(effective_payload.get("question", question)), "messages": []},
    )
    _emit_tool_after_hook(
        tool_name="answer_with_rag",
        payload=effective_payload,
        result=result,
    )
    if isinstance(result, dict) and result.get("final_answer"):
        return _clean_thinking(str(result.get("final_answer", "")))
    return _clean_thinking(str(result or ""))


@tool
def modify_code(instruction: str, working_dir: str = ".") -> str:
    """调用代码修改 Agent 执行修改任务。"""
    effective_payload, blocked_message = _apply_tool_before_hook(
        tool_name="modify_code",
        payload={"instruction": instruction, "working_dir": working_dir},
    )
    if blocked_message:
        return blocked_message
    result = get_default_agent_manager().invoke(
        "code_modify_agent",
        {
            "instruction": str(effective_payload.get("instruction", instruction)),
            "working_dir": str(effective_payload.get("working_dir", working_dir)),
        },
    )
    _emit_tool_after_hook(
        tool_name="modify_code",
        payload=effective_payload,
        result=result,
    )
    return _safe_extract_result(result)


def register_supervisor_tools(registry: Any | None = None) -> list[Any]:
    resolved_registry = registry or DEFAULT_AGENT_TOOL_REGISTRY
    resolved_registry.register_agent_tools("supervisor_agent", [review_code_with_codesage, answer_with_rag, modify_code], overwrite=True)
    return resolved_registry.get_tools_for_agent("supervisor_agent")


SUPERVISOR_TOOLS = register_supervisor_tools()


def _invoke_review(request: str, progress_callback: ProgressCallback | None, cancel_event: Any | None = None, parent_run_id: str = "") -> str:
    _emit_progress(progress_callback, type="step", stage="agent", agent="pr_review_agent", run_id=parent_run_id, status="running", summary="正在执行 PR 审查分析。")
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return "请求已取消。"
    task = ForkTaskSpec(parent_run_id=parent_run_id, parent_agent="supervisor_agent", child_agent_name="pr_review_agent", child_agent_mode="workflow", task_type="review", goal="审查提供的 Pull Request 或 diff 请求。", inputs={"request": request, "messages": [HumanMessage(content=request)], "cancel_event": cancel_event}, expected_output={"type": "review_report"}, allowed_actions=("review",), fork_reason="supervisor_review_route")
    result = get_default_agent_manager().run_fork_task(task, thread_id=parent_run_id, parent_run_id=parent_run_id)
    return _clean_thinking(result.summary or str(result.result_payload or ""))


def _invoke_rag(question: str, progress_callback: ProgressCallback | None, skill_context: dict[str, Any] | None = None, memory_context: str | None = None, cancel_event: Any | None = None, parent_run_id: str = "") -> str:
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return "请求已取消。"

    def rag_progress(stage: str, summary: str) -> None:
        _emit_progress(progress_callback, type="step", stage=stage, route="rag", agent="enhanced_rag_agent", run_id=parent_run_id, status="running", summary=summary)

    task = ForkTaskSpec(parent_run_id=parent_run_id, parent_agent="supervisor_agent", child_agent_name="enhanced_rag_agent", child_agent_mode="workflow", task_type="rag", goal="使用检索工作流回答仓库问题。", inputs={"question": question, "search_scope": {"skill_context": skill_context, "memory_context": memory_context}, "cancel_event": cancel_event}, expected_output={"type": "rag_answer"}, allowed_actions=("retrieve", "answer"), fork_reason="supervisor_rag_route")
    result = get_default_agent_manager().run_fork_task(task, thread_id=parent_run_id, parent_run_id=parent_run_id)
    if progress_callback is not None:
        rag_progress("fork", "RAG 分叉任务已完成。")
    return _clean_thinking(result.summary or str(result.result_payload or ""))


def _invoke_modify(instruction: str, working_dir: str, progress_callback: ProgressCallback | None, memory_context: str | None = None, skill_context: dict[str, Any] | None = None, cancel_event: Any | None = None, parent_run_id: str = "") -> dict[str, Any]:
    task = ForkTaskSpec(parent_run_id=parent_run_id, parent_agent="supervisor_agent", child_agent_name="code_modify_agent", child_agent_mode="workflow", task_type="modify", goal="执行请求的代码修改工作流。", inputs={"instruction": instruction, "working_dir": working_dir, "progress_callback": progress_callback, "memory_context": memory_context, "skill_context": skill_context, "cancel_event": cancel_event}, expected_output={"type": "code_modify"}, allowed_actions=("modify", "verify"), fork_reason="supervisor_modify_route")
    result = get_default_agent_manager().run_fork_task(task, thread_id=parent_run_id, parent_run_id=parent_run_id)
    payload = dict(result.result_payload or {})
    payload.setdefault("status", result.status)
    payload.setdefault("summary", result.summary)
    return payload


def _invoke_fork_worker(*, user_input: str, skill_context: dict[str, Any] | None, cancel_event: Any | None, parent_run_id: str) -> str:
    skill_execution = resolve_skill_execution(skill_context or {}, user_request=user_input, skill_args={"user_input": user_input}, allowed_actions=("skill", "analyze"), expected_output={"summary": "str", "result_payload": "object"})
    task = build_skill_execution_task(skill_execution, parent_run_id=parent_run_id, parent_agent="supervisor_agent", fork_reason="supervisor_skill_route")
    result = get_default_agent_manager().run_fork_task(task, thread_id=parent_run_id, parent_run_id=parent_run_id)
    return _clean_thinking(result.summary or str(result.result_payload or ""))


def _index_help_response(user_input: str) -> str:
    return "索引相关请求暂不直接进入自动执行流程。\n请使用 `/index` 建立仓库索引，或使用 `/index_docs` 上传并索引文档。\n\n原始请求：\n" + user_input


def _build_supervisor():
    class SimpleSupervisor:
        def __init__(self, tools: list[Any] | None = None) -> None:
            self.tools = list(tools or [])

        def route_request(self, user_input: str) -> RouteDecision:
            fast_lane = _fast_lane_decision(user_input)
            if fast_lane is not None:
                return fast_lane
            return _llm_route_decision(user_input)

        def invoke(self, input_dict: dict, progress_callback: ProgressCallback | None = None) -> dict:
            messages = input_dict.get("messages", [])
            skill_context = input_dict.get("skill_context")
            memory_context = input_dict.get("memory_context")
            cancel_event = input_dict.get("cancel_event")
            run_id = str(input_dict.get("run_id", "") or "")
            user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg
                    break
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_message = msg
                    break

            if not user_message:
                return {**input_dict, "messages": messages + [AIMessage(content="未找到可处理的用户消息。")]} 

            user_input = user_message.get("content", "") if isinstance(user_message, dict) else user_message.content
            before_route = emit_hook(
                HookContext(
                    hook_name="before_route",
                    route="route",
                    agent="supervisor_agent",
                    run_id=run_id,
                    action="route_request",
                    payload={
                        "user_input": user_input,
                        "skill_context": skill_context,
                        "memory_context": memory_context,
                    },
                )
            )
            if before_route.updated_payload:
                user_input = str(before_route.updated_payload.get("user_input", user_input))
            if before_route.additional_context:
                if "skill_context" in before_route.additional_context:
                    skill_context = before_route.additional_context.get("skill_context")
                if "memory_context" in before_route.additional_context:
                    memory_context = before_route.additional_context.get("memory_context")
            if before_route.decision == "block":
                blocked_text = f"请求被 Hook 策略阻止：{before_route.message or '未提供原因。'}"
                return {**input_dict, "messages": messages + [AIMessage(content=blocked_text)]}

            decision_payload = input_dict.get("route_decision")
            if isinstance(decision_payload, RouteDecision):
                decision = decision_payload
            elif isinstance(decision_payload, dict):
                decision = RouteDecision(**decision_payload)
            else:
                decision = self.route_request(user_input)
            emit_hook(
                HookContext(
                    hook_name="after_route",
                    route=decision.route,
                    agent="supervisor_agent",
                    run_id=run_id,
                    action="route_request",
                    payload={
                        "user_input": user_input,
                        "skill_context": skill_context,
                        "memory_context": memory_context,
                    },
                    result=decision.to_dict(),
                    metadata={"mode": decision.mode, "target_agent": decision.target_agent},
                )
            )

            _emit_progress(progress_callback, type="step", stage="route", route=decision.route, mode=decision.mode, agent=decision.target_agent, run_id=run_id, status="running", summary=decision.summary or decision.reason)

            agent_payload: dict[str, Any] | None = None
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                assistant_text = "请求已取消。"
            elif decision.route == "review":
                assistant_text = _invoke_review(str(decision.tool_args.get("request", user_input)), progress_callback, cancel_event=cancel_event, parent_run_id=run_id)
            elif decision.route == "modify":
                agent_payload = _invoke_with_supported_kwargs(
                    _invoke_modify,
                    str(decision.tool_args.get("instruction", user_input)),
                    str(decision.tool_args.get("working_dir", ".")),
                    progress_callback,
                    memory_context=str(memory_context or "").strip(),
                    skill_context=skill_context,
                    cancel_event=cancel_event,
                    parent_run_id=run_id,
                )
                assistant_text = _safe_extract_result(agent_payload)
            elif decision.route == "index":
                _emit_progress(progress_callback, type="step", stage="index_help", route=decision.route, mode=decision.mode, agent=decision.target_agent, run_id=run_id, status="running", summary="正在说明索引流程。")
                assistant_text = _index_help_response(user_input)
            elif decision.route == "none" and skill_context:
                assistant_text = _invoke_fork_worker(user_input=user_input, skill_context=skill_context, cancel_event=cancel_event, parent_run_id=run_id)
            elif decision.route == "none":
                assistant_text = "当前请求不需要调用 CodeSage 专用能力。"
            else:
                rag_kwargs: dict[str, Any] = {"skill_context": skill_context}
                if memory_context:
                    rag_kwargs["memory_context"] = str(memory_context).strip()
                rag_kwargs["cancel_event"] = cancel_event
                rag_kwargs["parent_run_id"] = run_id
                assistant_text = _invoke_with_supported_kwargs(
                    _invoke_rag,
                    str(decision.tool_args.get("question", user_input)),
                    progress_callback,
                    **rag_kwargs,
                )

            assistant_text = _clean_thinking(assistant_text)
            _emit_progress(progress_callback, type="step", stage="complete", route=decision.route, mode=decision.mode, agent=decision.target_agent, run_id=run_id, status="completed", summary="路由流程已完成。")
            return {**input_dict, "route_decision": decision.to_dict(), "agent_payload": agent_payload, "messages": messages + [AIMessage(content=assistant_text)]}

        def stream(self, input_dict: dict, config=None, stream_mode="values"):
            del config
            original_count = len(input_dict.get("messages", []))
            result = self.invoke(input_dict)
            new_messages = result.get("messages", [])[original_count:]
            for msg in new_messages:
                if stream_mode == "values":
                    yield {"messages": [msg]}
                else:
                    yield msg

        def snapshot(self) -> dict[str, Any]:
            return {"tool_names": [getattr(tool, "name", getattr(tool, "__name__", "")) for tool in self.tools]}

    return SimpleSupervisor(tools=SUPERVISOR_TOOLS)


def build_supervisor_runtime() -> Any:
    return _build_supervisor()


def get_supervisor():
    """获取 supervisor runtime 实例。"""
    handle = get_default_agent_manager().create_agent("supervisor_agent")
    return getattr(handle.instance, "runtime", handle.instance)


def invoke_supervisor(message: str, thread_id: str | None = None, skill_context: dict[str, Any] | None = None):
    """直接调用 supervisor 处理单条消息。"""
    del thread_id
    result = get_default_agent_manager().invoke("supervisor_agent", {"messages": [HumanMessage(content=message)], "skill_context": skill_context})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content") and last_msg.content:
            return _clean_thinking(str(last_msg.content))
    return _clean_thinking(str(result or ""))

from __future__ import annotations

import re
from typing import Any

from codesage.core.runtime import HookContext, HookResponse, list_registered_hooks, register_hook

_HIGH_RISK_PATTERN = re.compile(r"\b(delete|drop|reset|remove|rm)\b", flags=re.IGNORECASE)


def modify_approval_guard_hook(context: HookContext) -> HookResponse | None:
    payload = dict(context.payload or {})
    approval_mode = str(payload.get("approval_mode") or "").strip().lower()
    instruction = str(payload.get("instruction") or "").strip()
    if approval_mode != "off" or not instruction:
        return None
    if not _HIGH_RISK_PATTERN.search(instruction):
        return None
    return HookResponse(
        decision="block",
        message="检测到高风险修改指令，approval_mode=off 时不允许直接执行。",
        metadata={
            "hook_name": "ModifyApprovalGuardHook",
            "risk_level": "high",
            "matched_instruction": instruction,
        },
    )


def route_audit_hook(context: HookContext) -> HookResponse:
    result = dict(context.result or {})
    return HookResponse(
        metadata={
            "hook_name": "RouteAuditHook",
            "hook_route": str(result.get("route", "") or context.route),
            "hook_mode": str(result.get("mode", "") or ""),
            "hook_target_agent": str(result.get("target_agent", "") or context.agent),
        }
    )


def terminal_summary_hook(context: HookContext) -> HookResponse:
    status = str(context.metadata.get("status", "") or "")
    terminal_status = str(context.metadata.get("terminal_status", "") or status)
    summary = str(context.metadata.get("summary", "") or "")
    if not summary:
        summary = f"{context.agent or 'agent'} 已进入终态：{terminal_status or status or 'unknown'}"
    return HookResponse(
        metadata={
            "hook_name": "TerminalSummaryHook",
            "summary": summary,
            "terminal_status": terminal_status or status,
        }
    )


def register_builtin_hooks(*, overwrite: bool = True) -> list[dict[str, Any]]:
    register_hook(
        "before_agent_invoke",
        modify_approval_guard_hook,
        priority=100,
        hook_kind="policy",
        agent="code_modify_agent",
        blocking=True,
        overwrite=overwrite,
    )
    register_hook(
        "after_route",
        route_audit_hook,
        priority=10,
        hook_kind="observer",
        agent="supervisor_agent",
        overwrite=overwrite,
    )
    register_hook(
        "on_terminal",
        terminal_summary_hook,
        priority=10,
        hook_kind="runtime",
        overwrite=overwrite,
    )
    return [
        {
            "hook_name": hook.hook_name,
            "handler_name": hook.handler_name,
            "priority": hook.priority,
            "hook_kind": hook.hook_kind,
        }
        for hook in list_registered_hooks()
    ]

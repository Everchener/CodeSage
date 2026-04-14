from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_BASE_DIR = REPO_ROOT / ".codesage" / "previews"
PREVIEW_BASE_DIR.mkdir(parents=True, exist_ok=True)


def _emit(progress_callback, **payload: Any) -> None:
    if callable(progress_callback):
        progress_callback(payload)


def _preview_dir(preview_id: str) -> Path:
    return PREVIEW_BASE_DIR / preview_id


def _preview_file(preview_id: str) -> Path:
    return _preview_dir(preview_id) / "preview.json"


def _write_preview(preview_id: str, payload: dict[str, Any]) -> None:
    directory = _preview_dir(preview_id)
    directory.mkdir(parents=True, exist_ok=True)
    _preview_file(preview_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_preview(preview_id: str) -> dict[str, Any]:
    preview_path = _preview_file(preview_id)
    if not preview_path.exists():
        raise FileNotFoundError(f"预览 `{preview_id}` 不存在。")
    return json.loads(preview_path.read_text(encoding="utf-8"))


def discard_preview(preview_id: str) -> None:
    preview_dir = _preview_dir(preview_id)
    if not preview_dir.exists():
        raise FileNotFoundError(f"预览 `{preview_id}` 不存在。")
    shutil.rmtree(preview_dir)


def finalize_preview(preview_id: str) -> dict[str, Any]:
    payload = _read_preview(preview_id)
    result = {
        "status": "completed",
        "run_id": payload.get("run_id", ""),
        "parent_run_id": payload.get("parent_run_id", ""),
        "changes_made": list(payload.get("changes_made", []) or []),
        "applied_changes": list(payload.get("pending_changes", []) or []),
        "verification_result": payload.get("verification_result", "未执行自动验证。"),
        "output_result": payload.get("output_result", "已确认预览，但当前为简化实现，未自动落盘代码改动。"),
    }
    discard_preview(preview_id)
    return result


def get_code_modify_agent() -> dict[str, str]:
    return {"name": "code_modify_agent", "mode": "simplified"}


def invoke_code_modify_agent(
    instruction: str,
    working_dir: str = ".",
    progress_callback: Any | None = None,
    approval_mode: str = "high_risk",
    memory_context: str | None = None,
    skill_context: dict[str, Any] | None = None,
    cancel_event: Any | None = None,
    run_id: str = "",
    parent_run_id: str = "",
) -> dict[str, Any]:
    del memory_context, skill_context

    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return {
            "status": "cancelled",
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "output_result": "请求已取消。",
        }

    cleaned_instruction = str(instruction or "").strip()
    if not cleaned_instruction:
        return {
            "status": "error",
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "error": "缺少代码修改指令。",
        }

    _emit(progress_callback, stage="accepted", status="running", summary="代码修改工作流已接受。")
    _emit(progress_callback, stage="analyze", status="running", summary="正在分析代码修改请求。")
    _emit(progress_callback, stage="plan", status="running", summary="正在生成简化修改计划。")

    summary = (
        "当前仓库中的历史修改代理文件已损坏，系统已回退到简化兼容实现。"
        "该实现会保留指令、生成预览，并等待确认，但不会自动执行真实文件改动。"
    )
    verification_result = "未执行自动验证；这是一个仅保留接口兼容性的简化实现。"
    pending_changes = [f"工作目录：{working_dir}", f"修改指令：{cleaned_instruction}"]

    if approval_mode == "off":
        _emit(progress_callback, stage="output", status="completed", summary="已返回简化修改结果。")
        return {
            "status": "completed",
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "changes_made": [],
            "applied_changes": [],
            "verification_result": verification_result,
            "output_result": summary,
            "final_status": "completed",
        }

    preview_id = uuid.uuid4().hex
    _write_preview(
        preview_id,
        {
            "preview_id": preview_id,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "created_at": int(time.time()),
            "instruction": cleaned_instruction,
            "working_dir": working_dir,
            "changes_made": [],
            "pending_changes": pending_changes,
            "risk_reasons": ["当前使用简化兼容实现，未自动执行文件修改。"],
            "verification_result": verification_result,
            "output_result": summary,
            "diff_summary": "当前仅生成预览说明，没有实际代码 diff。",
        },
    )
    _emit(progress_callback, stage="confirmation", status="completed", summary="高风险修改预览已生成，等待用户确认。")
    return {
        "status": "awaiting_confirmation",
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "preview_id": preview_id,
        "changes_made": [],
        "pending_changes": pending_changes,
        "risk_reasons": ["当前使用简化兼容实现，未自动执行文件修改。"],
        "verification_result": verification_result,
        "output_result": summary,
        "diff_summary": "当前仅生成预览说明，没有实际代码 diff。",
        "requires_confirmation": True,
        "final_status": "awaiting_confirmation",
    }

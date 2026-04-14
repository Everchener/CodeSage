import asyncio
import json
import importlib.util
import logging
import os
import queue
import re
import threading
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from codesage.agents.modify.code_modify_agent import discard_preview, finalize_preview
from codesage.agents.review.pr_agent import PRReviewAgent
from codesage.api.runtime_services import get_app_agent_manager, get_api_runtime_services, initialize_api_runtime
from codesage.core.config import (
    GITHUB_WEBHOOK_SECRET,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MEMORY_ENABLED,
    PR_REVIEW_MAX_DIFF_BYTES,
    get_embedding_status,
)
from codesage.document_processor.processor import DocumentProcessor
from codesage.indexing.ingestion import index_repository
from codesage.memory import get_memory_service
from codesage.memory.service import resolve_context_policy
from codesage.core.observability import (
    get_observed_run,
    link_observed_runs,
    list_active_observed_runs,
    list_observed_runs,
    read_observed_events,
    read_observed_tool_calls,
    record_observed_event,
    start_observed_run,
)
from codesage.skills import (
    SUPPORTED_SKILL_ROUTES,
    SkillCommandError,
    SkillLoadError,
    SkillSelection,
    discover_skills,
    load_skill,
    parse_skill_command,
    select_skill,
)
from codesage.tools.github_tools import GitHubTransportError, get_pr_diff
from codesage.tools.llm_tools import call_llm, is_langchain_openai_available, sanitize_llm_output
from codesage.tools.milvus_tools import connect_milvus, ensure_memory_collection, search_codebase
from codesage.tools.review_guards import (
    ReviewInputError,
    WebhookAuthError,
    validate_repo_and_pr,
    validate_review_diff,
    verify_github_webhook_signature,
)

app = FastAPI(title="CodeSage", description="Multi-Agent intelligent code review platform")
MULTIPART_AVAILABLE = importlib.util.find_spec("multipart") is not None
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "45"))
ASK_TIMEOUT_SECONDS = float(os.getenv("ASK_TIMEOUT_SECONDS", "30"))
CANCEL_GRACE_PERIOD_SECONDS = float(os.getenv("CHAT_CANCEL_GRACE_SECONDS", "0.5"))
logger = logging.getLogger(__name__)
ACTIVE_CHAT_RUNS_LOCK = threading.Lock()
ACTIVE_CHAT_RUNS: dict[str, dict[str, Any]] = {}
app.mount("/static", StaticFiles(directory=str(TEMPLATE_DIR)), name="static")


@app.on_event("startup")
async def startup():
    try:
        initialize_api_runtime(app)
    except Exception as exc:
        app.state.agent_bootstrap_error = str(exc)
        app.state.agent_bootstrap_report = {"status": "error", "detail": str(exc)}
    else:
        app.state.agent_bootstrap_error = ""
    try:
        connect_milvus()
        if MEMORY_ENABLED:
            ensure_memory_collection()
    except Exception as exc:
        app.state.startup_milvus_error = str(exc)
    else:
        app.state.startup_milvus_error = ""


class WebhookPayload(BaseModel):
    action: str = ""
    number: int = 0
    repository: dict = {}
    pull_request: dict = {}


class AskRequest(BaseModel):
    question: str


class IndexRequest(BaseModel):
    repo_url: str = ""
    repo_path: str = ""
    mode: Literal["incremental", "rebuild"] = "incremental"

    @property
    def resolved_path(self) -> str:
        return self.repo_path or self.repo_url


class ReviewRequest(BaseModel):
    repo: str
    pr_number: int
    diff_text: str


class ChatRequest(BaseModel):
    message: str
    thread_id: str = ""


class ChatCancelRequest(BaseModel):
    run_id: str


class ModifyRequest(BaseModel):
    instruction: str
    working_dir: str = "."
    thread_id: str = ""
    approval_mode: Literal["off", "high_risk", "always"] = "high_risk"


class ModifyConfirmRequest(BaseModel):
    preview_id: str
    decision: Literal["approve", "reject"]
    thread_id: str = ""


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_chat_event(
    event_type: str,
    *,
    thread_id: str,
    run_id: str = "",
    content: str = "",
    stage: str = "",
    route: str = "",
    mode: str = "",
    agent: str = "",
    tool: str = "",
    status: str = "",
    summary: str = "",
    seq: int | None = None,
    schema: str = "chat_event_v1",
    context_layers: int | None = None,
    context_chars: int | None = None,
    skill_enabled: bool | None = None,
    skill_name: str = "",
    skill_source: str = "",
    skill_mode: str = "",
    **extra: Any,
) -> dict:
    payload = {
        "type": event_type,
        "content": content,
        "thread_id": thread_id,
        "run_id": run_id,
        "stage": stage,
        "route": route,
        "mode": mode,
        "agent": agent,
        "tool": tool,
        "status": status,
        "summary": summary,
        "seq": seq,
        "schema": schema,
        "context_layers": context_layers,
        "context_chars": context_chars,
        "skill_enabled": skill_enabled,
        "skill_name": skill_name,
        "skill_source": skill_source,
        "skill_mode": skill_mode,
    }
    payload.update(extra)
    return {key: value for key, value in payload.items() if value not in ("", None)}


def _format_modify_response(result: dict) -> dict:
    status = result.get("status", "completed")
    if status == "awaiting_confirmation":
        return {
            "status": "awaiting_confirmation",
            "run_id": result.get("run_id", ""),
            "parent_run_id": result.get("parent_run_id", ""),
            "preview_id": result.get("preview_id", ""),
            "risk_reasons": result.get("risk_reasons", []),
            "pending_changes": result.get("pending_changes", []),
            "diff_summary": result.get("diff_summary", ""),
        }
    if status == "error":
        return {
            "status": "error",
            "run_id": result.get("run_id", ""),
            "parent_run_id": result.get("parent_run_id", ""),
            "error": result.get("error", "Unknown error"),
        }
    return {
        "status": "completed",
        "run_id": result.get("run_id", ""),
        "parent_run_id": result.get("parent_run_id", ""),
        "changes_made": result.get("changes_made", []),
        "applied_changes": result.get("applied_changes", []),
        "verification": result.get("verification_result", ""),
    }


def _normalize_progress_event(
    payload: dict,
    *,
    thread_id: str,
    run_id: str,
    default_route: str,
    default_mode: str,
    default_agent: str,
    default_skill_enabled: bool | None = None,
    default_skill_name: str = "",
    default_skill_source: str = "",
    default_skill_mode: str = "",
) -> dict:
    normalized = dict(payload)
    normalized.setdefault("type", "step")
    normalized.setdefault("thread_id", thread_id)
    normalized.setdefault("run_id", run_id)
    normalized.setdefault("route", default_route)
    normalized.setdefault("mode", default_mode)
    normalized.setdefault("agent", default_agent)
    normalized.setdefault("skill_enabled", default_skill_enabled)
    normalized.setdefault("skill_name", default_skill_name)
    normalized.setdefault("skill_source", default_skill_source)
    normalized.setdefault("skill_mode", default_skill_mode)
    if normalized.get("type") == "step":
        normalized.setdefault("status", "running")
    normalized.setdefault("schema", "chat_event_v1")
    return normalized


def _build_skill_event_fields(skill_context: dict[str, Any] | None) -> dict[str, Any]:
    if not skill_context:
        return {"skill_enabled": False}
    return {
        "skill_enabled": True,
        "skill_name": str(skill_context.get("name", "")).strip(),
        "skill_source": str(skill_context.get("source", "")).strip(),
        "skill_mode": str(skill_context.get("selection_mode", "")).strip(),
    }


def _bind_route_tool_args(route_decision, user_message: str) -> dict[str, Any]:
    payload = route_decision.to_dict() if hasattr(route_decision, "to_dict") else dict(route_decision)
    route = str(payload.get("route", "")).strip().lower()

    if route == "review":
        payload["tool_args"] = {"request": user_message}
    elif route == "modify":
        payload["tool_args"] = {"instruction": user_message, "working_dir": "."}
    elif route == "rag":
        payload["tool_args"] = {"question": user_message}
    else:
        payload["tool_args"] = payload.get("tool_args", {})

    return payload


def _milvus_error_detail(exc: Exception) -> str:
    return f"Milvus is unavailable: {exc}"


def _register_active_chat_run(run_id: str, *, thread_id: str, cancel_event: threading.Event) -> None:
    now = time.time()
    with ACTIVE_CHAT_RUNS_LOCK:
        ACTIVE_CHAT_RUNS[run_id] = {
            "run_id": run_id,
            "thread_id": thread_id,
            "cancel_event": cancel_event,
            "status": "running",
            "created_at": now,
            "updated_at": now,
        }


def _set_active_chat_run_status(run_id: str, status: str) -> None:
    with ACTIVE_CHAT_RUNS_LOCK:
        record = ACTIVE_CHAT_RUNS.get(run_id)
        if record is not None:
            record["status"] = status
            record["updated_at"] = time.time()


def _update_active_chat_run(run_id: str, **fields: Any) -> None:
    if not fields:
        return
    with ACTIVE_CHAT_RUNS_LOCK:
        record = ACTIVE_CHAT_RUNS.get(run_id)
        if record is None:
            return
        record.update({key: value for key, value in fields.items() if value not in ("", None)})
        record["updated_at"] = time.time()


def _remove_active_chat_run(run_id: str) -> dict[str, Any] | None:
    with ACTIVE_CHAT_RUNS_LOCK:
        return ACTIVE_CHAT_RUNS.pop(run_id, None)


def _cancel_active_chat_run(run_id: str) -> bool:
    with ACTIVE_CHAT_RUNS_LOCK:
        record = ACTIVE_CHAT_RUNS.get(run_id)
        if record is None:
            return False
        record["status"] = "cancelling"
        cancel_event = record.get("cancel_event")

    if isinstance(cancel_event, threading.Event):
        cancel_event.set()
    return True


def get_supervisor(app_instance: Any) -> Any:
    manager = get_app_agent_manager(app_instance)
    handle = manager.create_agent("supervisor_agent")
    return getattr(handle.instance, "runtime", handle.instance)


async def _stop_supervisor_task(
    task: asyncio.Task,
    *,
    run_id: str,
    cancel_event: threading.Event,
) -> None:
    if not cancel_event.is_set():
        cancel_event.set()
    _set_active_chat_run_status(run_id, "cancelling")
    await asyncio.wait({task}, timeout=CANCEL_GRACE_PERIOD_SECONDS)
    if task.done():
        try:
            task.result()
        except Exception as exc:  # pragma: no cover - 防御性日志
            logger.warning("取消后 Supervisor 任务仍以错误结束：%s", exc)
    else:
        logger.warning(
            "超时取消后，Supervisor 任务在 %.2f 秒内仍未停止。",
            CANCEL_GRACE_PERIOD_SECONDS,
        )
        task.cancel()


def _ensure_milvus_available():
    try:
        connect_milvus()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_milvus_error_detail(exc)) from exc


def _is_probable_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _message_content(message) -> str:
    if message is None:
        return ""
    if hasattr(message, "content"):
        content = message.content
        return content if isinstance(content, str) else str(content)
    if isinstance(message, dict):
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(message)


def _is_assistant_message(message) -> bool:
    if isinstance(message, dict):
        return message.get("role") == "assistant"
    return getattr(message, "type", "") == "ai"


def _extract_assistant_contents(messages: list, user_input: str) -> list[str]:
    assistant_contents = []
    for message in messages:
        content = _message_content(message).strip()
        if not content:
            continue
        if _is_assistant_message(message):
            assistant_contents.append(content)

    if assistant_contents:
        return assistant_contents

    # 兼容旧流程：有些历史链路不会正确标记 assistant 消息。
    return [
        _message_content(message).strip()
        for message in messages
        if _message_content(message).strip() and _message_content(message).strip() != user_input.strip()
    ]


def _query_identifiers(query: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", str(query or ""))]


def _is_symbol_lookup_question(query: str) -> bool:
    normalized = str(query or "").strip().lower()
    if not normalized or not _query_identifiers(query):
        return False
    return any(
        cue in normalized
        for cue in ("在哪里定义", "在哪定义", "where is", "defined", "definition", "定义位置")
    )


def _match_symbol_lookup_hit(query: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    identifiers = [item.lower() for item in _query_identifiers(query)]
    if not identifiers:
        return None
    for identifier in reversed(identifiers):
        for hit in hits:
            func_name = str(hit.get("func_name", "") or "").strip().lower()
            if func_name == identifier:
                return hit
    for identifier in reversed(identifiers):
        for hit in hits:
            file_path = str(hit.get("file_path", "") or "").strip().lower()
            if identifier and identifier in file_path:
                return hit
    return hits[0] if hits else None


def _build_symbol_lookup_answer(hit: dict[str, Any]) -> str:
    func_name = str(hit.get("func_name", "") or "").strip()
    file_path = str(hit.get("file_path", "") or "").strip() or "unknown"
    if func_name:
        return f"`{func_name}` 定义在 `{file_path}`。"
    return f"相关定义位于 `{file_path}`。"


def build_readiness_report(app_instance: FastAPI | None = None) -> dict:
    langchain_ready = is_langchain_openai_available()
    embedding_status = get_embedding_status()
    checks = {
        "llm_api_key": {
            "ok": bool(LLM_API_KEY),
            "detail": "Configured" if LLM_API_KEY else "Missing LLM_API_KEY",
        },
        "llm_model": {
            "ok": bool(LLM_MODEL),
            "detail": LLM_MODEL if LLM_MODEL else "Missing LLM_MODEL",
        },
        "llm_base_url": {
            "ok": True,
            "detail": LLM_BASE_URL or "Using provider SDK default base URL",
        },
        "embedding": {
            "ok": bool(embedding_status.get("healthy", False)),
            "detail": embedding_status,
        },
        "langchain_openai": {
            "ok": langchain_ready,
            "detail": (
                "Installed"
                if langchain_ready
                else "Missing optional runtime dependency `langchain-openai`"
            ),
        },
        "python_multipart": {
            "ok": MULTIPART_AVAILABLE,
            "detail": (
                "已安装"
                if MULTIPART_AVAILABLE
                else "缺少可选运行时依赖 `python-multipart`"
            ),
        },
    }

    try:
        connect_milvus()
    except Exception as exc:
        checks["milvus"] = {"ok": False, "detail": str(exc)}
    else:
        checks["milvus"] = {"ok": True, "detail": "Connected"}

    if app_instance is not None:
        agent_bootstrap_error = str(getattr(app_instance.state, "agent_bootstrap_error", "") or "").strip()
        agent_bootstrap_report = getattr(app_instance.state, "agent_bootstrap_report", {}) or {}
        checks["agent_bootstrap"] = {
            "ok": not agent_bootstrap_error,
            "detail": agent_bootstrap_report if not agent_bootstrap_error else agent_bootstrap_error,
        }

    ready = all(item["ok"] for item in checks.values())
    return {"status": "ok" if ready else "error", "checks": checks}


@app.post("/webhook", summary="GitHub PR Webhook 鍏ュ彛")
async def webhook(request: Request):
    raw_body = await request.body()
    if not GITHUB_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook signing is not configured.")

    try:
        verify_github_webhook_signature(
            GITHUB_WEBHOOK_SECRET,
            raw_body,
            request.headers.get("X-Hub-Signature-256"),
        )
    except WebhookAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        payload = WebhookPayload.model_validate(json.loads(raw_body.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="Invalid webhook payload.") from exc

    if payload.action not in ("opened", "synchronize"):
        return {"status": "ignored"}

    repo = payload.repository.get("full_name", "")
    pr_number = payload.number
    try:
        validate_repo_and_pr(repo, pr_number)
    except ReviewInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        diff = get_pr_diff(repo, pr_number)
    except GitHubTransportError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub PR diff fetch failed: {exc}",
        ) from exc

    try:
        diff = validate_review_diff(diff, PR_REVIEW_MAX_DIFF_BYTES)
        get_app_agent_manager(app).invoke(
            "pr_review_agent",
            {
                "repo": repo,
                "pr_number": pr_number,
                "diff_text": diff,
            },
        )
    except ReviewInputError as exc:
        raise HTTPException(status_code=502, detail="Fetched PR diff is invalid.") from exc

    return {"status": "review triggered", "repo": repo, "pr": pr_number}


@app.post("/review", summary="直接触发代码审查")
async def review(req: ReviewRequest):
    try:
        validate_repo_and_pr(req.repo, req.pr_number)
        diff_text = validate_review_diff(req.diff_text, PR_REVIEW_MAX_DIFF_BYTES)
    except ReviewInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = get_app_agent_manager(app).invoke(
            "pr_review_agent",
            {
                "repo": req.repo,
                "pr_number": req.pr_number,
                "diff_text": diff_text,
            },
        )
    except ReviewInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Review unavailable: {exc}") from exc

    return {
        "status": "completed",
        "repo": req.repo,
        "pr": req.pr_number,
        "security_issues": [],
        "logic_issues": [],
        "final_comment": result.get("final_comment", ""),
    }


@app.post("/ask", summary="查询已索引知识库")
async def ask(req: AskRequest):
    try:
        hits = search_codebase(req.question, top_k=5)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable: {exc}") from exc

    if not hits:
        return {
            "answer": "未找到相关代码。请先通过 `/index` 为仓库建立索引。",
            "sources": [],
        }
    symbol_hit = _match_symbol_lookup_hit(req.question, hits) if _is_symbol_lookup_question(req.question) else None
    if symbol_hit is not None:
        return {
            "answer": _build_symbol_lookup_answer(symbol_hit),
            "sources": list({hit["file_path"] for hit in hits}),
        }

    context = "\n\n".join(
        f"[{hit['file_path']}::{hit['func_name']}]\n```python\n{hit['code'][:500]}\n```"
        for hit in hits
    )
    prompt = (
        "你是代码库问答助手。请基于检索到的代码回答用户问题。\n\n"
        f"相关代码：\n{context}\n\n"
        f"用户问题：{req.question}\n\n"
        "请简明回答，并在合适时引用文件路径。"
    )
    answer = call_llm(
        prompt,
        max_tokens=800,
        temperature=0.3,
        timeout=ASK_TIMEOUT_SECONDS,
    ).strip()
    answer = sanitize_llm_output(answer)
    if not answer and symbol_hit is not None:
        answer = _build_symbol_lookup_answer(symbol_hit)
    if not answer:
        raise HTTPException(status_code=503, detail="答案生成失败：模型返回为空。")

    return {"answer": answer, "sources": list({hit["file_path"] for hit in hits})}


@app.post("/index", summary="后台索引仓库")
async def index(req: IndexRequest, background_tasks: BackgroundTasks):
    path = req.resolved_path
    if not path:
        raise HTTPException(status_code=400, detail="A local repository path is required.")
    if _is_probable_url(path):
        raise HTTPException(
            status_code=400,
            detail="GitHub URLs are not supported yet. Please enter a local repository path.",
        )

    _ensure_milvus_available()
    background_tasks.add_task(index_repository, path, mode=req.mode)
    return {"status": "indexing started", "repo_path": path, "mode": req.mode}


if MULTIPART_AVAILABLE:
    @app.post("/index_docs", summary="上传并索引文档")
    async def index_docs(file: UploadFile = File(...)):
        _ensure_milvus_available()
        suffix = Path(file.filename or "upload").suffix
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            content = await file.read()
            with open(tmp_path, "wb") as handle:
                handle.write(content)
            result = DocumentProcessor().process_file(
                tmp_path,
                source_name=file.filename or Path(tmp_path).name,
            )
            if result.get("status") != "success":
                raise HTTPException(
                    status_code=400,
                    detail=result.get("message", "文档索引失败。"),
                )
            return result
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except PermissionError:
                    logger.warning("Temporary upload file could not be removed immediately: %s", tmp_path)
else:
    @app.post("/index_docs", summary="上传并索引文档")
    async def index_docs():
        raise HTTPException(
            status_code=503,
            detail="上传文档需要安装 `python-multipart`。",
        )


@app.post("/chat", summary="Supervisor Agent 对话入口")
async def chat(request: Request, req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    run_id = uuid.uuid4().hex
    agent_manager = get_app_agent_manager(app)
    supervisor = get_supervisor(app)
    memory_service = get_memory_service()

    async def generate():
        event_seq = 0

        def next_seq() -> int:
            nonlocal event_seq
            event_seq += 1
            return event_seq

        final_content = ""
        final_status = "success"
        route_decision = None
        routed_message = req.message
        cleaned_message = req.message
        agent_payload: dict[str, Any] | None = None
        context_metrics = {"layer_count": 0, "context_chars": 0}
        progress_events: queue.Queue[dict] = queue.Queue()
        cancel_event = threading.Event()
        skill_context: dict[str, Any] | None = None
        skill_event_fields = {"skill_enabled": False}
        agent_memory_context = ""
        active_run_finalized = False
        snapshot_persisted = False
        turn_memory_persisted = False

        def progress_callback(payload: dict) -> None:
            if cancel_event.is_set():
                return
            event_payload = dict(payload)
            source_run_id = str(event_payload.get("run_id", "") or "").strip()
            if source_run_id and source_run_id != run_id:
                event_payload["source_run_id"] = source_run_id
                event_payload["child_run_id"] = source_run_id
                event_payload.setdefault("parent_run_id", run_id)
            event_payload["run_id"] = run_id
            progress_events.put(event_payload)

        def build_event(event_type: str, **kwargs: Any) -> dict[str, Any]:
            return _build_chat_event(
                event_type,
                thread_id=thread_id,
                run_id=run_id,
                **kwargs,
            )

        def emit_recorded_event(event_name: str, payload: dict[str, Any]) -> str:
            observed_payload = record_observed_event(run_id, payload)
            _update_active_chat_run(
                run_id,
                route=observed_payload.get("route", ""),
                mode=observed_payload.get("mode", ""),
                agent=observed_payload.get("agent", ""),
                current_stage=observed_payload.get("stage", ""),
                last_summary=observed_payload.get("summary", ""),
                status=observed_payload.get("status", ""),
                event_count=int(observed_payload.get("seq", 0) or 0),
            )
            return _sse_event(event_name, observed_payload)

        def finalize_active_run(status: str) -> None:
            nonlocal active_run_finalized
            if active_run_finalized:
                return
            _set_active_chat_run_status(run_id, status)
            _remove_active_chat_run(run_id)
            active_run_finalized = True

        def persist_thread_snapshot() -> None:
            nonlocal snapshot_persisted
            if snapshot_persisted:
                return
            snapshot_persisted = True

        def persist_turn_memory() -> None:
            nonlocal turn_memory_persisted
            if turn_memory_persisted:
                return
            try:
                observed_events = read_observed_events(run_id, limit=0)
            except Exception:
                observed_events = []
            try:
                memory_service.record_turn_and_extract(
                    thread_id=thread_id,
                    user_input=cleaned_message,
                    assistant_output=final_content,
                    route=route_decision.route if route_decision else "",
                    agent=route_decision.target_agent if route_decision else "",
                    status=final_status,
                    observed_events=observed_events,
                    observed_at=time.time(),
                )
            except Exception as exc:
                logger.warning("鐠佹澘绻傞幐浣风畽閸栨牕銇戠拹銉窗%s", exc)
            finally:
                turn_memory_persisted = True

        _register_active_chat_run(run_id, thread_id=thread_id, cancel_event=cancel_event)
        start_observed_run(
            run_id,
            source="chat",
            thread_id=thread_id,
            summary="请求已接受，正在准备多 Agent 工作流。",
        )
        yield emit_recorded_event(
            "step",
            build_event(
                "step",
                stage="accepted",
                status="running",
                summary="请求已接受，正在准备多 Agent 工作流。",
                seq=next_seq(),
                **skill_event_fields,
            ),
        )

        try:
            abort_processing = False

            try:
                parsed_skill = parse_skill_command(req.message)
            except SkillCommandError as exc:
                parsed_skill = None
                abort_processing = True
                final_status = "error"
                final_content = str(exc)
                finalize_active_run("error")
                yield emit_recorded_event(
                    "error",
                    build_event(
                        "error",
                        content=final_content,
                        stage="skill",
                        status="error",
                        summary="技能命令解析失败。",
                        seq=next_seq(),
                        **skill_event_fields,
                    ),
                )

            if not abort_processing and parsed_skill is not None:
                cleaned_message = parsed_skill.user_request
                route_decision = supervisor.route_request(cleaned_message)
                yield emit_recorded_event(
                    "step",
                    build_event(
                        "step",
                        stage="route",
                        route=route_decision.route,
                        mode=route_decision.mode,
                        agent=route_decision.target_agent,
                        status="running",
                        summary=route_decision.summary or route_decision.reason,
                        seq=next_seq(),
                        **skill_event_fields,
                    ),
                )

                try:
                    context_policy = resolve_context_policy(
                        route=route_decision.route,
                        hinted_policy=route_decision.context_policy,
                    )
                    memory_context = memory_service.build_agent_context(
                        thread_id=thread_id,
                        user_input=cleaned_message,
                        context_policy=context_policy,
                    )
                    rendered_memory_context = memory_service.render_agent_context(memory_context)
                    if route_decision.route in {"rag", "modify"}:
                        routed_message = cleaned_message
                        agent_memory_context = rendered_memory_context
                    else:
                        routed_message = memory_service.compose_user_message(cleaned_message, memory_context)
                        agent_memory_context = ""
                    context_metrics = memory_service.summarize_context_payload(memory_context)
                    yield emit_recorded_event(
                        "step",
                        build_event(
                            "step",
                            stage="context",
                            route=route_decision.route,
                            mode=route_decision.mode,
                            agent=route_decision.target_agent,
                            status="running",
                            summary=(
                                f"已应用 {memory_context.get('template', '上下文模板')}，"
                                f"策略为 `{memory_context.get('policy', context_policy)}`。"
                            ),
                            seq=next_seq(),
                            context_layers=context_metrics["layer_count"],
                            context_chars=context_metrics["context_chars"],
                            **skill_event_fields,
                        ),
                    )
                except Exception as exc:
                    logger.warning("璁板繂涓婁笅鏂囨瀯寤哄け璐ワ細%s", exc)
                    routed_message = cleaned_message
                    agent_memory_context = ""

                try:
                    discovered_skills = discover_skills(project_root=PROJECT_ROOT)
                    if parsed_skill.is_explicit:
                        explicit_metadata = next(
                            (item for item in discovered_skills if item.name == parsed_skill.skill_name),
                            None,
                        )
                        if explicit_metadata is None:
                            raise SkillLoadError(f"未找到技能 `{parsed_skill.skill_name}`。")

                        explicit_fields = {
                            "skill_enabled": False,
                            "skill_name": explicit_metadata.name,
                            "skill_source": explicit_metadata.source,
                            "skill_mode": "explicit",
                        }
                        if route_decision.route not in SUPPORTED_SKILL_ROUTES:
                            yield emit_recorded_event(
                                "step",
                                build_event(
                                    "step",
                                    stage="skill",
                                    route=route_decision.route,
                                    mode=route_decision.mode,
                                    agent=route_decision.target_agent,
                                    status="skipped",
                                    summary="当前路由不支持技能，已忽略显式技能。",
                                    seq=next_seq(),
                                    **explicit_fields,
                                ),
                            )
                        else:
                            skill_context = load_skill(
                                SkillSelection(
                                    metadata=explicit_metadata,
                                    mode="explicit",
                                    reason="用户显式指定",
                                    user_request=cleaned_message,
                                )
                            ).to_context_dict()
                            skill_event_fields = _build_skill_event_fields(skill_context)
                            yield emit_recorded_event(
                                "step",
                                build_event(
                                    "step",
                                    stage="skill",
                                    route=route_decision.route,
                                    mode=route_decision.mode,
                                    agent=route_decision.target_agent,
                                    status="completed",
                                    summary=f"已启用显式技能 `{skill_context['name']}`。",
                                    seq=next_seq(),
                                    **skill_event_fields,
                                ),
                            )
                    elif route_decision.route in SUPPORTED_SKILL_ROUTES and discovered_skills:
                        auto_selection = select_skill(
                            user_input=cleaned_message,
                            route=route_decision.route,
                            skills=discovered_skills,
                        )
                        if auto_selection is not None:
                            skill_context = load_skill(auto_selection).to_context_dict()
                            skill_event_fields = _build_skill_event_fields(skill_context)
                            yield emit_recorded_event(
                                "step",
                                build_event(
                                    "step",
                                    stage="skill",
                                    route=route_decision.route,
                                    mode=route_decision.mode,
                                    agent=route_decision.target_agent,
                                    status="completed",
                                    summary=f"已自动启用技能 `{skill_context['name']}`。",
                                    seq=next_seq(),
                                    **skill_event_fields,
                                ),
                            )
                        else:
                            yield emit_recorded_event(
                                "step",
                                build_event(
                                    "step",
                                    stage="skill",
                                    route=route_decision.route,
                                    mode=route_decision.mode,
                                    agent=route_decision.target_agent,
                                    status="completed",
                                    summary="未自动命中适用技能，继续按默认链路处理。",
                                    seq=next_seq(),
                                    **skill_event_fields,
                                ),
                            )
                except (SkillLoadError, SkillCommandError) as exc:
                    abort_processing = True
                    final_status = "error"
                    final_content = str(exc)
                    finalize_active_run("error")
                    yield emit_recorded_event(
                        "error",
                        build_event(
                            "error",
                            content=final_content,
                            stage="skill",
                            route=route_decision.route if route_decision else "",
                            mode=route_decision.mode if route_decision else "",
                            agent=route_decision.target_agent if route_decision else "",
                            status="error",
                            summary="技能加载失败。",
                            seq=next_seq(),
                            **skill_event_fields,
                        ),
                    )

            if not abort_processing:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        supervisor.invoke,
                        {
                            "messages": [{"role": "user", "content": routed_message}],
                            "run_id": run_id,
                            "route_decision": _bind_route_tool_args(route_decision, routed_message),
                            "skill_context": skill_context,
                            "memory_context": agent_memory_context,
                            "cancel_event": cancel_event,
                        },
                        progress_callback=progress_callback,
                    )
                )
                deadline = time.monotonic() + CHAT_TIMEOUT_SECONDS

                while not task.done():
                    while True:
                        try:
                            payload = progress_events.get_nowait()
                        except queue.Empty:
                            break
                        normalized = _normalize_progress_event(
                            payload,
                            thread_id=thread_id,
                            run_id=run_id,
                            default_route=route_decision.route if route_decision else "",
                            default_mode=route_decision.mode if route_decision else "",
                            default_agent=route_decision.target_agent if route_decision else "",
                            default_skill_enabled=skill_event_fields.get("skill_enabled"),
                            default_skill_name=str(skill_event_fields.get("skill_name", "")),
                            default_skill_source=str(skill_event_fields.get("skill_source", "")),
                            default_skill_mode=str(skill_event_fields.get("skill_mode", "")),
                        )
                        normalized.setdefault("seq", next_seq())
                        yield emit_recorded_event(normalized["type"], normalized)

                    if cancel_event.is_set():
                        await _stop_supervisor_task(
                            task,
                            run_id=run_id,
                            cancel_event=cancel_event,
                        )
                        final_status = "cancelled"
                        final_content = "请求已取消。"
                        finalize_active_run("cancelled")
                        persist_turn_memory()
                        persist_thread_snapshot()
                        return

                    if await request.is_disconnected():
                        await _stop_supervisor_task(
                            task,
                            run_id=run_id,
                            cancel_event=cancel_event,
                        )
                        final_status = "cancelled"
                        final_content = "请求已取消。"
                        finalize_active_run("cancelled")
                        persist_turn_memory()
                        persist_thread_snapshot()
                        return

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        await _stop_supervisor_task(
                            task,
                            run_id=run_id,
                            cancel_event=cancel_event,
                        )
                        finalize_active_run("timeout")
                        raise asyncio.TimeoutError()
                    await asyncio.wait({task}, timeout=min(0.05, remaining))

                while True:
                    try:
                        payload = progress_events.get_nowait()
                    except queue.Empty:
                        break
                    normalized = _normalize_progress_event(
                        payload,
                        thread_id=thread_id,
                        run_id=run_id,
                        default_route=route_decision.route if route_decision else "",
                        default_mode=route_decision.mode if route_decision else "",
                        default_agent=route_decision.target_agent if route_decision else "",
                        default_skill_enabled=skill_event_fields.get("skill_enabled"),
                        default_skill_name=str(skill_event_fields.get("skill_name", "")),
                        default_skill_source=str(skill_event_fields.get("skill_source", "")),
                        default_skill_mode=str(skill_event_fields.get("skill_mode", "")),
                    )
                    normalized.setdefault("seq", next_seq())
                    yield emit_recorded_event(normalized["type"], normalized)

                result = task.result()
                agent_payload = result.get("agent_payload") if isinstance(result, dict) else None
                if cancel_event.is_set():
                    if (
                        route_decision
                        and route_decision.route == "modify"
                        and isinstance(agent_payload, dict)
                        and agent_payload.get("status") == "awaiting_confirmation"
                    ):
                        preview_id = str(agent_payload.get("preview_id", "")).strip()
                        if preview_id:
                            try:
                                discard_preview(preview_id)
                            except FileNotFoundError:
                                pass
                    final_status = "cancelled"
                    final_content = "请求已取消。"
                    finalize_active_run("cancelled")
                    persist_turn_memory()
                    persist_thread_snapshot()
                    return
                assistant_contents = _extract_assistant_contents(
                    result.get("messages", []),
                    routed_message,
                )
                if not assistant_contents:
                    raise RuntimeError("No assistant response was produced by the supervisor.")

                for content in assistant_contents:
                    final_content = content
                    yield emit_recorded_event(
                        "message",
                        build_event(
                            "message",
                            content=content,
                            stage="response",
                            route=route_decision.route if route_decision else "",
                            mode=route_decision.mode if route_decision else "",
                            agent=route_decision.target_agent if route_decision else "",
                            status="streaming",
                            summary="已生成助手回复。",
                            seq=next_seq(),
                            **skill_event_fields,
                        ),
                    )

                if route_decision and route_decision.route == "modify" and isinstance(agent_payload, dict):
                    if agent_payload.get("status") == "awaiting_confirmation":
                        final_status = "awaiting_confirmation"
                        finalize_active_run("awaiting_confirmation")
                        yield emit_recorded_event(
                            "confirmation_required",
                            build_event(
                                "confirmation_required",
                                content=final_content,
                                stage="confirmation",
                                route=route_decision.route,
                                mode=route_decision.mode,
                                agent=route_decision.target_agent,
                                status="awaiting_confirmation",
                                summary="高风险修改预览已生成，等待用户确认。",
                                seq=next_seq(),
                                preview_id=agent_payload.get("preview_id", ""),
                                pending_changes=agent_payload.get("pending_changes", []),
                                risk_reasons=agent_payload.get("risk_reasons", []),
                                diff_summary=agent_payload.get("diff_summary", ""),
                                **skill_event_fields,
                            ),
                        )

                persist_turn_memory()
                try:
                    memory_service.record_turn_and_extract(
                        thread_id=thread_id,
                        user_input=cleaned_message,
                        assistant_output=final_content,
                    )
                except Exception as exc:
                    logger.warning("璁板繂鎸佷箙鍖栧け璐ワ細%s", exc)
                persist_thread_snapshot()
        except asyncio.TimeoutError:
            final_status = "timeout"
            final_content = "等待模型或工具链响应超时。"
            finalize_active_run("timeout")
            yield emit_recorded_event(
                "error",
                build_event(
                    "error",
                    content=final_content,
                    stage="timeout",
                    route=route_decision.route if route_decision else "",
                    mode=route_decision.mode if route_decision else "",
                    agent=route_decision.target_agent if route_decision else "",
                    status="timeout",
                    summary="聊天请求超过了配置的超时时间。",
                    seq=next_seq(),
                    **skill_event_fields,
                ),
            )
        except Exception as exc:
            final_status = "error"
            final_content = f"处理失败：{exc}"
            finalize_active_run("error")
            yield emit_recorded_event(
                "error",
                build_event(
                    "error",
                    content=final_content,
                    stage="error",
                    route=route_decision.route if route_decision else "",
                    mode=route_decision.mode if route_decision else "",
                    agent=route_decision.target_agent if route_decision else "",
                    status="error",
                    summary="聊天请求在完成前发生失败。",
                    seq=next_seq(),
                    **skill_event_fields,
                ),
            )

        if not snapshot_persisted and final_status in {
            "completed",
            "success",
            "awaiting_confirmation",
            "cancelled",
            "timeout",
            "error",
        }:
            persist_turn_memory()
            persist_thread_snapshot()

        if not active_run_finalized:
            if final_status in {"success", "completed"}:
                finalize_active_run("completed")
            else:
                finalize_active_run(final_status)

        yield emit_recorded_event(
            "done",
            build_event(
                "done",
                content=final_content,
                stage="done",
                route=route_decision.route if route_decision else "",
                mode=route_decision.mode if route_decision else "",
                agent=route_decision.target_agent if route_decision else "",
                status=final_status,
                summary="流式对话已完成。",
                seq=next_seq(),
                context_layers=context_metrics["layer_count"],
                context_chars=context_metrics["context_chars"],
                **skill_event_fields,
            ),
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/cancel", summary="取消正在运行的聊天请求")
async def chat_cancel(req: ChatCancelRequest):
    if _cancel_active_chat_run(req.run_id):
        return {"status": "cancelling", "run_id": req.run_id}
    return {"status": "not_running", "run_id": req.run_id}


@app.get("/runs", summary="列出已观测的 Agent 运行记录")
async def list_runs(
    limit: int = 20,
    status: str = "",
    route: str = "",
):
    safe_limit = min(max(limit, 1), 200)
    return {
        "runs": list_observed_runs(
            limit=safe_limit,
            status=status or None,
            route=route or None,
        )
    }


@app.get("/runs/active", summary="列出活跃的 Agent 运行记录")
async def list_active_runs():
    with ACTIVE_CHAT_RUNS_LOCK:
        active_chat_runs = {
            run_id: {key: value for key, value in payload.items() if key != "cancel_event"}
            for run_id, payload in ACTIVE_CHAT_RUNS.items()
        }
    return {"runs": list_active_observed_runs(active_chat_runs)}


@app.get("/runs/{run_id}", summary="获取 Agent 运行详情")
async def get_run(run_id: str, limit: int = 200):
    safe_limit = min(max(limit, 1), 500)
    try:
        return get_observed_run(run_id, limit=safe_limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/events", summary="获取运行事件")
async def get_run_events(run_id: str, limit: int = 200):
    safe_limit = min(max(limit, 1), 500)
    try:
        return {"run_id": run_id, "events": read_observed_events(run_id, limit=safe_limit)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/tool_calls", summary="获取运行工具调用")
async def get_run_tool_calls(run_id: str, limit: int = 200):
    safe_limit = min(max(limit, 1), 500)
    try:
        return {"run_id": run_id, "tool_calls": read_observed_tool_calls(run_id, limit=safe_limit)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/modify", summary="代码修改 Agent 入口")
async def modify(req: ModifyRequest):
    try:
        result = get_app_agent_manager(app).invoke(
            "code_modify_agent",
            {
                "instruction": req.instruction,
                "working_dir": req.working_dir,
                "approval_mode": req.approval_mode,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"代码修改不可用：{exc}") from exc

    if result.get("status") == "error":
        raise HTTPException(status_code=503, detail=result.get("error", "代码修改不可用"))

    return _format_modify_response(result)


@app.post("/modify/confirm", summary="确认或取消代码修改预览")
async def modify_confirm(req: ModifyConfirmRequest):
    try:
        if req.decision == "reject":
            discard_preview(req.preview_id)
            return {
                "status": "cancelled",
                "preview_id": req.preview_id,
            }

        result = finalize_preview(req.preview_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"预览确认不可用：{exc}") from exc

    return {
        "status": "completed",
        "run_id": result.get("run_id", ""),
        "parent_run_id": result.get("parent_run_id", ""),
        "changes_made": result.get("changes_made", []),
        "applied_changes": result.get("applied_changes", []),
        "verification": result.get("verification_result", ""),
    }


@app.get("/", summary="前端入口")
async def root():
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/health", summary="存活探针")
async def health():
    return {"status": "ok"}


@app.get("/ready", summary="就绪探针")
async def ready():
    get_api_runtime_services(app)
    report = build_readiness_report(app)
    return JSONResponse(status_code=200 if report["status"] == "ok" else 503, content=report)


def main():
    import uvicorn

    uvicorn.run("codesage.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

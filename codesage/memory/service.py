from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from codesage.core.config import (
    MEMORY_ENABLED,
    MEMORY_LONG_TOP_K,
    MEMORY_SHORT_TTL_MINUTES,
    MEMORY_SHORT_WINDOW_TURNS,
    MEMORY_WRITE_MIN_CONFIDENCE,
)
from codesage.memory.event_store import EventStore
from codesage.memory.fact_store import FactStore
from codesage.memory.retriever import LongTermRetriever
from codesage.memory.session_store import SessionStore
from codesage.core.runtime import REPO_ROOT


logger = logging.getLogger(__name__)

ALLOWED_MEMORY_TYPES = {"preference", "constraint", "decision", "project_fact"}
DEFAULT_CONTEXT_BUDGET = 4000
DEFAULT_DEDUPE_THRESHOLD = 0.92
DEFAULT_POLICY_BUDGETS = {
    "default": 3000,
    "rag": 3400,
    "modify": 3600,
    "review": 2200,
    "none": 0,
}


@dataclass
class MemoryItem:
    memory_type: str
    content: str
    confidence: float
    summary: str = ""
    importance: float = 0.5
    created_at: int = 0
    updated_at: int = 0
    score: float = 0.0
    item_id: str | None = None
    scope: str = "thread"
    thread_id: str = ""
    project_id: str = ""
    status: str = "active"


@dataclass
class ShortTermState:
    recent_turns: list[dict[str, str]] = field(default_factory=list)
    rolling_summary: str = ""
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ContextPolicy:
    name: str
    include_session_memory: bool
    include_recent_events: bool
    include_long_term: bool
    working_context: str
    template_name: str
    context_header: str


CONTEXT_POLICIES: dict[str, ContextPolicy] = {
    "default": ContextPolicy(
        name="default",
        include_session_memory=True,
        include_recent_events=True,
        include_long_term=True,
        working_context="Use recent conversation state and stable project facts when relevant.",
        template_name="default_context_v2",
        context_header="General Assistant Context",
    ),
    "rag": ContextPolicy(
        name="rag",
        include_session_memory=True,
        include_recent_events=True,
        include_long_term=True,
        working_context="Focus on answering repository questions with precise context.",
        template_name="rag_context_v2",
        context_header="Repository Q&A Context",
    ),
    "modify": ContextPolicy(
        name="modify",
        include_session_memory=True,
        include_recent_events=True,
        include_long_term=True,
        working_context="Focus on the current code change task, user constraints, and implementation decisions.",
        template_name="modify_context_v2",
        context_header="Code Change Context",
    ),
    "review": ContextPolicy(
        name="review",
        include_session_memory=False,
        include_recent_events=True,
        include_long_term=True,
        working_context="Focus on repository constraints and review-specific facts.",
        template_name="review_context_v2",
        context_header="Code Review Context",
    ),
    "none": ContextPolicy(
        name="none",
        include_session_memory=False,
        include_recent_events=False,
        include_long_term=False,
        working_context="No additional memory context is required for this request.",
        template_name="none_context_v1",
        context_header="No Additional Context",
    ),
}

ROUTE_CONTEXT_POLICY: dict[str, str] = {
    "review": "review",
    "rag": "rag",
    "modify": "modify",
    "index": "none",
    "none": "none",
}


def resolve_context_policy(route: str, hinted_policy: str | None = None) -> str:
    normalized_hint = (hinted_policy or "").strip().lower()
    if normalized_hint in CONTEXT_POLICIES:
        return normalized_hint
    return ROUTE_CONTEXT_POLICY.get((route or "").strip().lower(), "default")


class LongTermMemoryStore(Protocol):
    def retrieve(
        self,
        *,
        thread_id: str,
        project_id: str,
        query: str,
        top_k: int,
        policy: str = "default",
    ) -> list[MemoryItem]:
        ...

    def upsert(self, *, thread_id: str, project_id: str, items: list[MemoryItem]) -> None:
        ...


class MemorySessionStore:
    def __init__(self, window_turns: int, ttl_minutes: int):
        self.window_turns = max(1, int(window_turns))
        self.ttl_seconds = max(1, int(ttl_minutes) * 60)
        self._sessions: dict[str, ShortTermState] = {}

    def add_turn(self, thread_id: str, user_message: str, assistant_message: str) -> None:
        if not thread_id:
            return
        self._prune_expired()
        state = self._sessions.setdefault(thread_id, ShortTermState())
        state.recent_turns.append(
            {
                "user": (user_message or "").strip(),
                "assistant": (assistant_message or "").strip(),
            }
        )
        self._compact(state)
        state.updated_at = time.time()

    def get_state(self, thread_id: str) -> ShortTermState:
        self._prune_expired()
        state = self._sessions.get(thread_id)
        if state is None:
            return ShortTermState()
        return ShortTermState(
            recent_turns=[dict(turn) for turn in state.recent_turns],
            rolling_summary=state.rolling_summary,
            updated_at=state.updated_at,
        )

    def render_context(self, thread_id: str, char_budget: int) -> str:
        state = self.get_state(thread_id)
        sections: list[str] = []
        if state.rolling_summary:
            sections.append(f"Recent summary:\n{state.rolling_summary}")
        if state.recent_turns:
            lines: list[str] = []
            for turn in state.recent_turns:
                user = self._truncate(turn.get("user", ""), 220)
                assistant = self._truncate(turn.get("assistant", ""), 260)
                lines.append(f"- User: {user}\n  Assistant: {assistant}")
            sections.append("Recent turns:\n" + "\n".join(lines))
        return self._trim_to_budget("\n\n".join(sections), char_budget)

    def _compact(self, state: ShortTermState) -> None:
        while len(state.recent_turns) > self.window_turns:
            oldest = state.recent_turns.pop(0)
            summary_line = (
                f"User asked: {self._truncate(oldest.get('user', ''), 120)} | "
                f"Assistant replied: {self._truncate(oldest.get('assistant', ''), 140)}"
            )
            if state.rolling_summary:
                state.rolling_summary = f"{state.rolling_summary}\n- {summary_line}"
            else:
                state.rolling_summary = f"- {summary_line}"
            state.rolling_summary = self._trim_to_budget(state.rolling_summary, 1200)

    def _prune_expired(self) -> None:
        now = time.time()
        expired = [
            key for key, value in self._sessions.items() if (now - value.updated_at) > self.ttl_seconds
        ]
        for key in expired:
            self._sessions.pop(key, None)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _trim_to_budget(text: str, char_budget: int) -> str:
        if char_budget <= 0:
            return ""
        if len(text) <= char_budget:
            return text
        return text[-char_budget:]


class MilvusLongTermMemoryStore:
    def __init__(self):
        try:
            from codesage.tools.milvus_tools import ensure_memory_collection

            ensure_memory_collection()
        except Exception as exc:  # pragma: no cover - 依赖于运行环境
            logger.warning("Skipped memory collection initialization: %s", exc)

    def retrieve(
        self,
        *,
        thread_id: str,
        project_id: str,
        query: str,
        top_k: int,
        policy: str = "default",
    ) -> list[MemoryItem]:
        from codesage.tools.milvus_tools import search_memory_items

        raw_items = search_memory_items(
            thread_id=thread_id,
            project_id=project_id,
            query=query,
            top_k=top_k,
            policy=policy,
        )
        return [self._from_dict(item) for item in raw_items]

    def upsert(self, *, thread_id: str, project_id: str, items: list[MemoryItem]) -> None:
        from codesage.tools.milvus_tools import upsert_memory_items

        payload = [
            {
                "memory_id": item.item_id,
                "project_id": item.project_id or project_id,
                "thread_id": item.thread_id or thread_id,
                "scope": item.scope,
                "memory_type": item.memory_type,
                "content": item.content,
                "summary": item.summary or item.content,
                "status": item.status,
                "confidence": item.confidence,
                "importance": item.importance,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in items
        ]
        upsert_memory_items(
            thread_id=thread_id,
            project_id=project_id,
            items=payload,
        )

    @staticmethod
    def _from_dict(raw: dict) -> MemoryItem:
        return MemoryItem(
            memory_type=str(raw.get("memory_type", "")),
            content=str(raw.get("content", "")),
            summary=str(raw.get("summary", "") or raw.get("content", "")),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            importance=float(raw.get("importance", 0.0) or 0.0),
            created_at=int(raw.get("created_at", 0) or 0),
            updated_at=int(raw.get("updated_at", 0) or 0),
            score=float(raw.get("score", 0.0) or 0.0),
            item_id=str(raw.get("memory_id", "") or raw.get("id", "") or ""),
            scope=str(raw.get("scope", "thread") or "thread"),
            thread_id=str(raw.get("thread_id", "") or ""),
            project_id=str(raw.get("project_id", "") or ""),
            status=str(raw.get("status", "active") or "active"),
        )


class InMemoryLongTermMemoryStore:
    def __init__(self):
        self._items: list[MemoryItem] = []

    def retrieve(
        self,
        *,
        thread_id: str,
        project_id: str,
        query: str,
        top_k: int,
        policy: str = "default",
    ) -> list[MemoryItem]:
        del policy
        query_tokens = _tokenize(query)
        scored: list[MemoryItem] = []
        for item in self._items:
            if item.status and item.status != "active":
                continue
            if item.scope == "thread" and item.thread_id != thread_id:
                continue
            if project_id and item.project_id not in {"", project_id}:
                continue
            score = _cosine_similarity(query_tokens, _tokenize(item.content))
            if item.scope == "thread":
                score += 0.3
            elif item.scope == "project":
                score += 0.15
            scored.append(
                MemoryItem(
                    memory_type=item.memory_type,
                    content=item.content,
                    summary=item.summary,
                    confidence=item.confidence,
                    importance=item.importance,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    score=score,
                    item_id=item.item_id,
                    scope=item.scope,
                    thread_id=item.thread_id,
                    project_id=item.project_id,
                    status=item.status,
                )
            )
        scored.sort(key=lambda value: value.score, reverse=True)
        return scored[: max(0, top_k)]

    def upsert(self, *, thread_id: str, project_id: str, items: list[MemoryItem]) -> None:
        for candidate in items:
            best_index = -1
            best_score = 0.0
            candidate_tokens = _tokenize(candidate.content)
            for index, existing in enumerate(self._items):
                if existing.scope != candidate.scope:
                    continue
                if existing.thread_id != (candidate.thread_id or thread_id):
                    continue
                if existing.project_id != (candidate.project_id or project_id):
                    continue
                score = _cosine_similarity(candidate_tokens, _tokenize(existing.content))
                if score > best_score:
                    best_score = score
                    best_index = index
            payload = MemoryItem(
                memory_type=candidate.memory_type,
                content=candidate.content,
                summary=candidate.summary,
                confidence=candidate.confidence,
                importance=candidate.importance,
                created_at=candidate.created_at,
                updated_at=candidate.updated_at,
                score=candidate.score,
                item_id=candidate.item_id,
                scope=candidate.scope,
                thread_id=candidate.thread_id or thread_id,
                project_id=candidate.project_id or project_id,
                status=candidate.status,
            )
            if best_index >= 0 and best_score >= DEFAULT_DEDUPE_THRESHOLD:
                self._items[best_index] = payload
            else:
                self._items.append(payload)

    def list_items(self, *, thread_id: str = "", project_id: str = "") -> list[MemoryItem]:
        rows = []
        for item in self._items:
            if thread_id and item.scope == "thread" and item.thread_id != thread_id:
                continue
            if project_id and item.project_id not in {"", project_id}:
                continue
            rows.append(item)
        return rows


class MemoryService:
    def __init__(
        self,
        short_store: MemorySessionStore,
        long_store: LongTermMemoryStore,
        *,
        event_store: EventStore | None = None,
        session_store: SessionStore | None = None,
        fact_store: FactStore | None = None,
        enabled: bool,
        long_top_k: int,
        min_confidence: float,
        context_char_budget: int = DEFAULT_CONTEXT_BUDGET,
        dedupe_threshold: float = DEFAULT_DEDUPE_THRESHOLD,
    ):
        self.short_store = short_store
        self.long_store = long_store
        self.event_store = event_store or EventStore()
        self.session_store = session_store or SessionStore()
        self.fact_store = fact_store or FactStore()
        self.retriever = LongTermRetriever(fact_store=self.fact_store, long_store=self.long_store)
        self.enabled = enabled
        self.long_top_k = max(1, int(long_top_k))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.context_char_budget = max(0, int(context_char_budget))
        self.dedupe_threshold = max(0.0, min(1.0, float(dedupe_threshold)))
        self.policy_char_budgets = {
            policy_name: max(0, int(budget))
            for policy_name, budget in DEFAULT_POLICY_BUDGETS.items()
        }
        self.policy_char_budgets["default"] = self.context_char_budget

    def build_context(self, thread_id: str, user_input: str) -> str:
        context = self.build_agent_context(
            thread_id=thread_id,
            user_input=user_input,
            context_policy="default",
        )
        return self.render_agent_context(context)

    def build_agent_context(
        self,
        thread_id: str,
        user_input: str,
        context_policy: str = "default",
    ) -> dict[str, object]:
        policy_name = resolve_context_policy(route="", hinted_policy=context_policy)
        policy = CONTEXT_POLICIES.get(policy_name, CONTEXT_POLICIES["default"])
        policy_budget = self.policy_char_budgets.get(policy.name, self.context_char_budget)
        payload: dict[str, object] = {
            "policy": policy.name,
            "template": policy.template_name,
            "header": policy.context_header,
            "char_budget": policy_budget,
            "layers": [],
            "coverage": {},
        }
        if not self.enabled or not thread_id:
            if policy.working_context:
                payload["layers"] = [
                    {"layer": "agent_working_context", "content": policy.working_context}
                ]
            return payload

        layers: list[dict[str, str]] = []
        coverage: dict[str, int] = {}
        if policy.working_context:
            layers.append({"layer": "agent_working_context", "content": policy.working_context})

        if policy.include_session_memory:
            session_budget = max(400, policy_budget // 3)
            session_content, covers_turn = self._render_session_memory(thread_id, char_budget=session_budget)
            if session_content:
                layers.append({"layer": "conversation_state", "content": session_content})
                coverage["session_covers_until_turn"] = covers_turn

        if policy.include_recent_events:
            recent_budget = max(500, policy_budget // 3)
            recent_content, recent_from_turn = self._render_recent_events(thread_id, char_budget=recent_budget)
            if recent_content:
                layers.append({"layer": "recent_events", "content": recent_content})
                coverage["recent_from_turn"] = recent_from_turn

        if policy.include_long_term:
            project_id = self._project_id()
            try:
                memories = self.retriever.retrieve(
                    thread_id=thread_id,
                    project_id=project_id,
                    query=user_input,
                    policy=policy.name,
                    top_k=self.long_top_k,
                )
                long_context = self._format_long_term_context(memories, layers=layers)
                if long_context:
                    layers.append({"layer": "stable_memory", "content": long_context})
            except Exception as exc:  # pragma: no cover - 防御性分支
                logger.warning("Long-term memory retrieval failed: %s", exc)

        payload["layers"] = layers
        payload["coverage"] = coverage
        return payload

    def render_agent_context(self, context_payload: dict[str, object]) -> str:
        layers = context_payload.get("layers", [])
        if not isinstance(layers, list):
            return ""
        policy_name = str(context_payload.get("policy", "default"))
        header = str(context_payload.get("header", "General Assistant Context")).strip()
        label_map = self._layer_label_map(policy_name)

        rendered_blocks: list[str] = []
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            layer_name = str(layer.get("layer", "")).strip()
            content = str(layer.get("content", "")).strip()
            if not layer_name or not content:
                continue
            title = label_map.get(layer_name, layer_name.replace("_", " ").title())
            rendered_blocks.append(f"{title}:\n{content}")

        if not rendered_blocks:
            return ""
        rendered = f"{header}\n\n" + "\n\n".join(rendered_blocks)
        context_budget = context_payload.get("char_budget", self.context_char_budget)
        try:
            budget = int(context_budget)
        except (TypeError, ValueError):
            budget = self.context_char_budget
        return self._trim_to_budget(rendered, budget)

    @staticmethod
    def _layer_label_map(policy_name: str) -> dict[str, str]:
        if policy_name == "review":
            return {
                "agent_working_context": "Review Goal",
                "stable_memory": "Relevant Review Facts",
                "recent_events": "Recent Review Events",
            }
        if policy_name == "rag":
            return {
                "agent_working_context": "Answering Goal",
                "conversation_state": "Session Memory",
                "recent_events": "Recent Thread Events",
                "stable_memory": "Relevant Long-Term Facts",
            }
        if policy_name == "modify":
            return {
                "agent_working_context": "Change Goal",
                "conversation_state": "Session Memory",
                "recent_events": "Recent Thread Events",
                "stable_memory": "Relevant Long-Term Facts",
            }
        return {
            "agent_working_context": "Working Context",
            "conversation_state": "Session Memory",
            "recent_events": "Recent Events",
            "stable_memory": "Long-Term Memory",
        }

    def compose_user_message(
        self,
        user_input: str,
        memory_context: str | dict[str, object],
    ) -> str:
        if isinstance(memory_context, dict):
            rendered_context = self.render_agent_context(memory_context)
            context_header = str(memory_context.get("header", "Context Bundle")).strip()
        else:
            rendered_context = memory_context
            context_header = "Context Bundle"

        if not rendered_context:
            return user_input
        return (
            f"{context_header} (use only when relevant):\n"
            f"{rendered_context}\n\n"
            f"Current user request:\n{user_input}"
        )

    def summarize_context_payload(self, context_payload: dict[str, object]) -> dict[str, int]:
        layers = context_payload.get("layers", [])
        if not isinstance(layers, list):
            return {"layer_count": 0, "context_chars": 0}
        char_count = 0
        layer_count = 0
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            content = str(layer.get("content", "")).strip()
            if not content:
                continue
            layer_count += 1
            char_count += len(content)
        return {"layer_count": layer_count, "context_chars": char_count}

    def record_turn_and_extract(
        self,
        thread_id: str,
        user_input: str,
        assistant_output: str,
        *,
        route: str = "",
        agent: str = "",
        status: str = "completed",
        observed_events: list[dict[str, object]] | None = None,
        observed_at: float | int | None = None,
    ) -> None:
        if not self.enabled or not thread_id:
            return

        existing_recent = self.event_store.read_recent_events(thread_id, turns=1)
        if existing_recent:
            latest_user = next(
                (
                    str(event.get("content", "") or "")
                    for event in reversed(existing_recent)
                    if str(event.get("event_type", event.get("type", "")) or "") == "user_message"
                ),
                "",
            )
            latest_assistant = next(
                (
                    str(event.get("content", "") or "")
                    for event in reversed(existing_recent)
                    if str(event.get("event_type", event.get("type", "")) or "") == "assistant_final"
                ),
                "",
            )
            if latest_user.strip() == str(user_input or "").strip() and latest_assistant.strip() == str(assistant_output or "").strip():
                return

        self.short_store.add_turn(
            thread_id=thread_id,
            user_message=user_input,
            assistant_message=assistant_output,
        )

        turn_events = self._build_turn_events(
            thread_id=thread_id,
            user_input=user_input,
            assistant_output=assistant_output,
            route=route,
            agent=agent,
            status=status,
            observed_events=observed_events or [],
            observed_at=observed_at,
        )
        turn_id = self.event_store.next_turn_id(thread_id)
        persisted_events = self.event_store.append_turn_events(
            thread_id=thread_id,
            turn_id=turn_id,
            events=turn_events,
            observed_at=observed_at,
        )
        previous_session = self.session_store.read(thread_id)
        self.session_store.update_from_events(thread_id, persisted_events, previous_state=previous_session)

        try:
            extracted = self._extract_long_term_memories(
                user_input=user_input,
                assistant_output=assistant_output,
                thread_id=thread_id,
                project_id=self._project_id(),
            )
            if extracted:
                active_facts = self.fact_store.upsert_facts(
                    thread_id=thread_id,
                    project_id=self._project_id(),
                    items=[
                        {
                            "memory_id": item.item_id,
                            "scope": item.scope,
                            "memory_type": item.memory_type,
                            "content": item.content,
                            "summary": item.summary or item.content,
                            "confidence": item.confidence,
                            "importance": item.importance,
                            "updated_at": item.updated_at,
                            "created_at": item.created_at,
                            "source_turn_id": turn_id,
                        }
                        for item in extracted
                    ],
                    source_turn_id=turn_id,
                )
                persisted_items = [
                    MemoryItem(
                        memory_type=str(row.get("memory_type", "")),
                        content=str(row.get("content", "")),
                        summary=str(row.get("summary", "")),
                        confidence=float(row.get("confidence", 0.0) or 0.0),
                        importance=float(row.get("importance", 0.0) or 0.0),
                        created_at=int(row.get("created_at", 0) or 0),
                        updated_at=int(row.get("updated_at", 0) or 0),
                        item_id=str(row.get("memory_id", "") or ""),
                        scope=str(row.get("scope", "thread") or "thread"),
                        thread_id=str(row.get("thread_id", "") or thread_id),
                        project_id=str(row.get("project_id", "") or self._project_id()),
                        status=str(row.get("status", "active") or "active"),
                    )
                    for row in active_facts
                ]
                if persisted_items:
                    self.long_store.upsert(
                        thread_id=thread_id,
                        project_id=self._project_id(),
                        items=persisted_items,
                    )
        except Exception as exc:  # pragma: no cover - 防御性分支
            logger.warning("Memory extraction or persistence failed: %s", exc)

    def _render_session_memory(self, thread_id: str, *, char_budget: int) -> tuple[str, int]:
        session_content, covers_turn = self.session_store.render_prompt_context(thread_id, char_budget=char_budget)
        return session_content, covers_turn

    def _render_recent_events(self, thread_id: str, *, char_budget: int) -> tuple[str, int]:
        content, from_turn = self.event_store.render_recent_context(thread_id, char_budget=char_budget)
        if content:
            return content, from_turn
        fallback = self.short_store.render_context(thread_id, char_budget=char_budget)
        return fallback, 0

    def _extract_long_term_memories(
        self,
        *,
        user_input: str,
        assistant_output: str,
        thread_id: str,
        project_id: str,
    ) -> list[MemoryItem]:
        try:
            from codesage.tools.llm_tools import call_llm_json
        except ModuleNotFoundError:
            return []

        prompt = (
            "提取值得在后续多轮对话中保留的稳定事实。\n"
            "允许的 memory_type 取值：preference、constraint、decision、project_fact。\n"
            "每一项还可以包含 scope，取值为 `thread` 或 `project`。\n"
            "只返回如下格式的合法 JSON：\n"
            '{"memories":[{"memory_type":"preference","scope":"thread","content":"...","confidence":0.82,"importance":0.7}]}\n'
            '如果没有值得存储的内容，请返回 {"memories":[]}。\n\n'
            f"用户消息：\n{user_input}\n\n"
            f"助手消息：\n{assistant_output}"
        )
        payload = call_llm_json(prompt, max_tokens=500)
        return normalize_extracted_memories(
            payload,
            self.min_confidence,
            thread_id=thread_id,
            project_id=project_id,
        )

    @staticmethod
    def _format_long_term_context(items: list[MemoryItem], *, layers: list[dict[str, str]]) -> str:
        if not items:
            return ""
        existing_text = " ".join(str(layer.get("content", "") or "") for layer in layers).lower()
        lines = []
        for item in items:
            content = item.summary or item.content
            normalized = content.lower().strip()
            if not normalized or normalized in existing_text:
                continue
            prefix = item.scope or "thread"
            lines.append(f"- ({prefix}/{item.memory_type}) {content}")
        if not lines:
            return ""
        return "Relevant long-term facts:\n" + "\n".join(lines)

    def _build_turn_events(
        self,
        *,
        thread_id: str,
        user_input: str,
        assistant_output: str,
        route: str,
        agent: str,
        status: str,
        observed_events: list[dict[str, object]],
        observed_at: float | int | None,
    ) -> list[dict[str, object]]:
        timestamp = float(observed_at or time.time())
        events: list[dict[str, object]] = [
            {
                "event_type": "user_message",
                "content": user_input,
                "summary": _truncate_text(user_input, 200),
                "status": "completed",
            },
            {
                "event_type": "route_decision",
                "route": route,
                "agent": agent,
                "summary": f"Route selected: {route or 'unknown'}",
                "status": "completed",
            },
        ]
        for observed in observed_events:
            tool = str(observed.get("tool", "") or "").strip()
            observed_type = str(observed.get("type", "") or "").strip()
            if tool:
                events.append(
                    {
                        "event_type": "tool_call",
                        "tool": tool,
                        "summary": str(observed.get("summary", "") or observed.get("content", "") or "")[:220],
                        "status": str(observed.get("status", "") or "running"),
                    }
                )
                if observed_type in {"message", "done", "error"} or observed.get("content"):
                    events.append(
                        {
                            "event_type": "tool_result",
                            "tool": tool,
                            "content": str(observed.get("content", "") or ""),
                            "summary": str(observed.get("summary", "") or "")[:220],
                            "status": str(observed.get("status", "") or "completed"),
                        }
                    )
        events.extend(
            [
                {
                    "event_type": "assistant_final",
                    "content": assistant_output,
                    "summary": _truncate_text(assistant_output, 220),
                    "status": status,
                },
                {
                    "event_type": "status_change",
                    "status": status,
                    "route": route,
                    "agent": agent,
                    "summary": f"Turn finished with status `{status or 'completed'}`.",
                },
            ]
        )
        for item in events:
            item.setdefault("thread_id", thread_id)
            item.setdefault("timestamp", timestamp)
            item.setdefault("route", route)
            item.setdefault("agent", agent)
        return events

    @staticmethod
    def _project_id() -> str:
        return hashlib.sha1(str(REPO_ROOT).lower().encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _trim_to_budget(text: str, char_budget: int) -> str:
        if char_budget <= 0:
            return ""
        if len(text) <= char_budget:
            return text
        return text[-char_budget:]


def normalize_extracted_memories(
    payload: dict | list | None,
    min_confidence: float,
    *,
    thread_id: str = "",
    project_id: str = "",
) -> list[MemoryItem]:
    if payload is None:
        return []

    raw_items: list = []
    if isinstance(payload, dict):
        maybe_memories = payload.get("memories", [])
        if isinstance(maybe_memories, list):
            raw_items = maybe_memories
    elif isinstance(payload, list):
        raw_items = payload

    now = int(time.time())
    normalized: list[MemoryItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        memory_type = str(raw.get("memory_type", "")).strip().lower()
        if memory_type not in ALLOWED_MEMORY_TYPES:
            continue

        content = str(raw.get("content", "")).strip()
        if not content:
            continue

        confidence = _normalize_score(raw.get("confidence", 0.0))
        if confidence < min_confidence:
            continue

        scope = str(raw.get("scope", "thread") or "thread").strip().lower()
        if scope not in {"thread", "project", "user"}:
            scope = "thread"
        importance = _normalize_score(raw.get("importance", _default_importance(memory_type)))
        summary = str(raw.get("summary", "") or content).strip()
        subject = _normalize_subject(summary or content)
        memory_id = hashlib.sha1(
            f"{scope}|{memory_type}|{subject}|{project_id}|{thread_id if scope == 'thread' else ''}".encode("utf-8")
        ).hexdigest()[:20]
        normalized.append(
            MemoryItem(
                memory_type=memory_type,
                content=content,
                summary=summary,
                confidence=confidence,
                importance=importance,
                created_at=now,
                updated_at=now,
                item_id=memory_id,
                scope=scope,
                thread_id=thread_id if scope == "thread" else "",
                project_id=project_id,
                status="active",
            )
        )
    return normalized


def _normalize_subject(content: str) -> str:
    tokens = re.sub(r"[^a-z0-9]+", " ", str(content or "").lower()).split()
    return " ".join(tokens[:12])


def _default_importance(memory_type: str) -> float:
    if memory_type == "constraint":
        return 0.85
    if memory_type == "decision":
        return 0.8
    if memory_type == "preference":
        return 0.65
    return 0.6


def _normalize_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _truncate_text(value: str, max_chars: int) -> str:
    value = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _tokenize(text: str) -> dict[str, float]:
    tokens: dict[str, float] = {}
    for token in (text or "").lower().split():
        token = token.strip()
        if not token:
            continue
        tokens[token] = tokens.get(token, 0.0) + 1.0
    return tokens


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, value in a.items():
        dot += value * b.get(key, 0.0)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_memory_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService(
            short_store=MemorySessionStore(
                window_turns=MEMORY_SHORT_WINDOW_TURNS,
                ttl_minutes=MEMORY_SHORT_TTL_MINUTES,
            ),
            long_store=MilvusLongTermMemoryStore(),
            event_store=EventStore(),
            session_store=SessionStore(),
            fact_store=FactStore(),
            enabled=MEMORY_ENABLED,
            long_top_k=MEMORY_LONG_TOP_K,
            min_confidence=MEMORY_WRITE_MIN_CONFIDENCE,
        )
    return _memory_service

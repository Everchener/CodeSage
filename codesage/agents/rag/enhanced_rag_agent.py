"""增强版 RAG Agent，支持查询改写、Self-RAG、重排和父块合并。"""

import operator
import re
from collections import defaultdict
from typing import Annotated, Any, Dict, List, Tuple, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from codesage.skills import render_skill_prompt_section
from codesage.tools.prompt_tools import build_json_output, build_prompt, build_system_prompt
from codesage.tools.llm_tools import call_llm, call_llm_json, sanitize_llm_output
from codesage.tools.milvus_tools import search_knowledge_base
from codesage.tools.parent_chunk_store import get_parent_chunk_store


AUTO_MERGE_THRESHOLD = 2
REWRITE_MEMORY_MAX_HINT_LINES = 4
REWRITE_MEMORY_MAX_HINT_CHARS = 240
GROUNDEDNESS_PASS_THRESHOLD = 0.80
GROUNDEDNESS_REVISE_THRESHOLD = 0.50
ABSTAIN_ANSWER_TEMPLATE = "现有检索证据不足，无法可靠回答这个问题。"

SELF_RAG_SYSTEM = build_system_prompt(
    role="你是检索判定助手，负责判断一个问题是否必须结合当前代码知识库才能准确回答。",
    responsibilities=[
        "区分通用编程知识和仓库特定知识。",
        "在证据不足时优先选择更稳妥的检索路径。",
        "输出稳定、可解析的 JSON 结果。",
    ],
    rules=[
        "涉及具体文件、类、函数、调用链、配置、实现细节时，应倾向 needs_retrieval=true。",
        "纯语言特性、通用算法、与当前仓库无关的基础知识问题，可返回 needs_retrieval=false。",
        "reason 必须是中文短句，直接说明判定依据。",
    ],
    output_instruction="只返回 JSON 对象，不要输出 Markdown、解释或思考过程。",
)

DIRECT_ANSWER_SYSTEM = build_system_prompt(
    role="你是代码知识助手，负责在无需检索时直接回答通用编程问题。",
    responsibilities=[
        "直接回答用户问题。",
        "在无法确定时明确说明需要检索代码库后才能给出可靠结论。",
    ],
    rules=[
        "回答要简洁、准确，不要假装知道仓库特定细节。",
    ],
    output_instruction="直接输出中文答案，不要输出思考过程。",
)

REWRITE_SYSTEM = build_system_prompt(
    role="你是检索查询改写助手，负责把用户问题压缩成适合代码检索的表达。",
    responsibilities=[
        "提取核心概念、符号名和限定词。",
        "保留类名、函数名、路径名、模块名等精确信号。",
    ],
    rules=[
        "不要改变用户意图，不要补充原问题中不存在的术语。",
        "不要输出解释、前缀或多句话。",
    ],
    output_instruction="只输出一条检索语句。",
)

STEP_BACK_QUESTION_SYSTEM = build_system_prompt(
    role="你是 step-back 提问助手，负责将具体问题上提为更通用的背景问题。",
    responsibilities=[
        "抽取问题背后的通用原则或设计主题。",
    ],
    rules=[
        "只输出一句问题。",
        "不要引入与原问题无关的新术语。",
    ],
    output_instruction="只输出问题本身，不要解释。",
)

STEP_BACK_ANSWER_SYSTEM = build_system_prompt(
    role="你是背景知识助手，负责回答一个抽象层更高的通用问题。",
    responsibilities=[
        "提供简洁背景知识，帮助后续检索与作答。",
    ],
    rules=[
        "只回答通用知识，不要伪造仓库细节。",
        "答案控制在 120 字以内。",
    ],
    output_instruction="直接输出中文答案，不要输出思考过程。",
)

RERANK_SYSTEM = build_system_prompt(
    role="你是检索证据选择助手，负责在候选检索结果中挑出主要证据和备用证据。",
    responsibilities=[
        "优先识别真正能回答问题的主要证据。",
        "当多个来源互补时，允许保留跨源证据。",
        "当某个候选可能有用但不足以直接回答问题时，将其放入备用证据。",
    ],
    rules=[
        "primary_indices 最多保留 4 个，backup_indices 最多保留 2 个。",
        "不要只按关键词重合选择，要优先考虑是否能直接支持答案。",
        "当不确定时，更倾向于把候选放入 backup，而不是直接丢弃。",
    ],
    output_instruction="只返回 JSON 对象，不要输出解释或思考过程。",
)

GENERATE_SYSTEM = build_system_prompt(
    role="你是代码知识回答助手，负责根据检索结果生成可追溯、不过度推断的答案。",
    responsibilities=[
        "优先根据检索到的证据回答用户问题。",
        "在证据不足时明确说明，而不是强行下结论。",
        "尽量引用来源标题，便于用户回溯。",
    ],
    rules=[
        "只根据给定上下文作答。",
        "不要输出思考过程。",
    ],
    output_instruction="直接输出中文答案。",
)


VERIFY_GROUNDING_SYSTEM = build_system_prompt(
    role="你是 groundedness 校验助手，负责判断回答是否被检索证据直接支撑。",
    responsibilities=[
        "只检查回答是否被给定证据支持，不重新回答原问题。",
        "识别回答中的 unsupported claims 和缺失证据。",
        "输出稳定、可解析的 JSON 结果。",
    ],
    rules=[
        "只能依据给定的检索内容、证据摘要和候选回答进行判断。",
        "如果回答包含证据中没有出现的确定性事实，应降低 groundedness_score。",
        "当回答整体可由证据推出时，groundedness_score 应接近 1.0。",
        "unsupported_claims 与 missing_evidence 必须是简短中文短句列表。",
    ],
    output_instruction="只返回 JSON 对象，不要输出解释、Markdown 或思考过程。",
)

REVISE_GROUNDED_ANSWER_SYSTEM = build_system_prompt(
    role="你是 groundedness 修订助手，负责在不新增事实的前提下改写回答。",
    responsibilities=[
        "删除未被证据支持的内容。",
        "优先保留有明确来源支撑的事实和结论。",
        "当证据不足时，明确说明不能可靠下结论。",
    ],
    rules=[
        "只能使用给定证据中的信息，不得补充新事实。",
        "如果 unsupported_claims 无法被删除后保留完整答案，应输出证据不足的保守说明。",
        "保留现有的 [source_label] 来源标注风格。",
    ],
    output_instruction="直接输出中文修订答案，不要输出思考过程。",
)


class EnhancedRAGState(TypedDict):
    """增强版 RAG Agent 的状态。"""

    query: str
    rewritten_query: str
    expanded_query: str
    step_back_context: str
    retrieved_docs: list
    merged_docs: list
    reranked_docs: list
    primary_docs: list
    backup_docs: list
    evidence_bundle: list
    draft_answer: str
    needs_retrieval: bool
    retrieval_target: str
    grounding_passed: bool
    grounding_score: float
    unsupported_claims: list
    missing_evidence: list
    verification_action: str
    verification_reason: str
    revision_attempted: bool
    final_answer: str
    memory_context: str | None
    skill_context: dict[str, Any] | None
    messages: Annotated[list, operator.add]
    progress_callback: Any
    cancel_event: Any
    cancelled: bool


def _compact_memory_for_rewrite(
    memory_context: str | None,
    *,
    max_lines: int = REWRITE_MEMORY_MAX_HINT_LINES,
    max_chars: int = REWRITE_MEMORY_MAX_HINT_CHARS,
) -> str:
    """提取少量记忆线索，帮助改写时消解省略和指代。"""
    text = str(memory_context or "").strip()
    if not text:
        return ""

    normalized_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if "context package" in line.lower():
            continue
        if line.endswith(":"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if not line or line in seen:
            continue
        seen.add(line)
        normalized_lines.append(line)

    if not normalized_lines:
        return ""

    selected: list[str] = []
    current_chars = 0
    for line in normalized_lines:
        addition = len(line) + (1 if selected else 0)
        if selected and current_chars + addition > max_chars:
            break
        if not selected and len(line) > max_chars:
            selected.append(line[: max_chars - 3].rstrip() + "...")
            break
        selected.append(line)
        current_chars += addition
        if len(selected) >= max_lines:
            break

    return "\n".join(f"- {line}" for line in selected)


def _clean_rewritten_query(text: str | None) -> str:
    cleaned = re.sub(r"<\|ankton\w*\|>.*?<\|/ankton\|>", "", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip().strip('"\n ')


def _is_cancellation_requested(state: EnhancedRAGState) -> bool:
    cancel_event = state.get("cancel_event")
    return bool(cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)())


def _cancelled_state(state: EnhancedRAGState, *, stage: str) -> EnhancedRAGState:
    return {
        **state,
        "cancelled": True,
        "final_answer": f"Request cancelled during {stage}.",
    }


def _route_after_self_rag(state: EnhancedRAGState) -> str:
    if state.get("cancelled"):
        return "cancelled"
    return "rewrite" if state.get("needs_retrieval", True) else "direct_answer"


def _route_if_not_cancelled(next_stage: str):
    def _route(state: EnhancedRAGState) -> str:
        return "cancelled" if state.get("cancelled") else next_stage

    return _route


def _build_context_bundle(
    state: EnhancedRAGState,
    *,
    include_backup_docs: bool = True,
) -> tuple[list[dict], str, str, str]:
    primary_docs = list(state.get("primary_docs") or state.get("reranked_docs", []))
    backup_docs = list(state.get("backup_docs", []))
    docs = list(primary_docs)
    if include_backup_docs and len(docs) < 2 and backup_docs:
        existing = {_doc_identity(doc) for doc in docs}
        for doc in backup_docs:
            key = _doc_identity(doc)
            if key in existing:
                continue
            docs.append(doc)
            existing.add(key)

    context_parts: list[str] = []
    for doc in docs:
        text = doc.get("text") or doc.get("content") or doc.get("code", "")
        title = doc.get("source_label") or doc.get("title") or doc.get("source") or doc.get("file_path", "unknown")
        context_parts.append(f"[{title}]\n{text[:500]}")

    context = "\n\n".join(context_parts)
    evidence_summary = "\n".join(_build_evidence_summary_lines(list(state.get("evidence_bundle", [])))).strip()
    memory_context = str(state.get("memory_context") or "").strip()
    return docs, context, evidence_summary, memory_context


def _normalize_answer(text: str | None) -> str:
    return sanitize_llm_output(text or "").strip()


def _coerce_grounding_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def _decide_verification_action(score: float, *, revision_attempted: bool) -> str:
    if score >= GROUNDEDNESS_PASS_THRESHOLD:
        return "pass"
    if score >= GROUNDEDNESS_REVISE_THRESHOLD and not revision_attempted:
        return "revise_once"
    return "abstain"


def _route_after_verification(state: EnhancedRAGState) -> str:
    if state.get("cancelled"):
        return "cancelled"
    action = str(state.get("verification_action") or "abstain")
    if action == "pass":
        return "pass"
    if action == "revise_once":
        return "revise"
    return "abstain"


def _query_identifiers(query: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", str(query or ""))]


def _is_symbol_lookup_query(query: str) -> bool:
    normalized = str(query or "").strip().lower()
    if not normalized or not _query_identifiers(query):
        return False
    return any(
        cue in normalized
        for cue in ("在哪里定义", "在哪定义", "where is", "defined", "definition", "定义位置")
    )


def _match_symbol_lookup_doc(query: str, docs: list[dict]) -> dict | None:
    identifiers = [item.lower() for item in _query_identifiers(query)]
    if not identifiers:
        return None
    for identifier in reversed(identifiers):
        for doc in docs:
            func_name = str(doc.get("func_name", "") or "").strip().lower()
            if func_name == identifier:
                return doc
    for identifier in reversed(identifiers):
        for doc in docs:
            source_label = str(doc.get("source_label", "") or "").strip().lower()
            file_path = str(doc.get("file_path", "") or "").strip().lower()
            if identifier and (identifier in source_label or identifier in file_path):
                return doc
    return docs[0] if docs else None


def _build_symbol_lookup_answer(doc: dict) -> str:
    func_name = str(doc.get("func_name", "") or "").strip()
    file_path = str(doc.get("file_path", "") or "").strip()
    location = (
        file_path
        or str(doc.get("source_label", "") or "").strip()
        or str(doc.get("title", "") or "").strip()
        or str(doc.get("source", "") or "").strip()
        or "unknown"
    )
    if func_name:
        return f"`{func_name}` 定义在 `{location}`。"
    return f"相关定义位于 `{location}`。"


def self_rag_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """判断当前问题是否需要检索代码知识库。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="self_rag")
    query = state["query"]
    if _is_symbol_lookup_query(query):
        new_state = {**state, "needs_retrieval": True}
        callback = new_state.get("progress_callback")
        if callback:
            callback("self_rag", "正在识别符号定位问题...")
        return new_state
    prompt = build_prompt(
        task="判断用户问题是否必须先检索代码知识库才能准确回答。",
        context_sections=[("用户问题", query)],
        rules=[
            "如果问题指向当前仓库中的具体实现、定义位置、行为差异、配置或调用关系，应返回 needs_retrieval=true。",
            "如果问题是通用编程知识，且不依赖当前仓库上下文，应返回 needs_retrieval=false。",
            "当你拿不准时，优先返回 needs_retrieval=true。",
        ],
        examples=[
            (
                "如何实现 Python 装饰器？",
                '{"needs_retrieval": false, "reason": "这是通用编程概念，不依赖当前仓库。"}',
            ),
            (
                "SupervisorAgent 的 run 方法在哪里定义？",
                '{"needs_retrieval": true, "reason": "问题涉及当前代码库中的具体符号定义位置。"}',
            ),
            (
                "为什么这个仓库的 /chat 路由会走到 RAG？",
                '{"needs_retrieval": true, "reason": "问题依赖当前仓库的实际路由实现与配置。"}',
            ),
        ],
        output_format=build_json_output(
            """{
  "needs_retrieval": true,
  "reason": "中文原因"
}"""
        ),
    )

    result = call_llm_json(prompt, system=SELF_RAG_SYSTEM, max_tokens=250)
    if result and isinstance(result, dict):
        needs_retrieval = result.get("needs_retrieval", True)
    else:
        needs_retrieval = True

    new_state = {**state, "needs_retrieval": needs_retrieval}
    callback = new_state.get("progress_callback")
    if callback:
        callback("self_rag", "正在分析问题...")
    return new_state


def direct_answer_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """无需检索时，直接基于通用知识回答。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="direct_answer")
    query = state["query"]
    prompt = build_prompt(
        task="直接回答这个无需检索的用户问题。",
        context_sections=[("用户问题", query)],
        rules=[
            "只回答通用知识，不要假设当前仓库里一定存在某个实现。",
            "如果无法可靠回答，请明确说明需要检索代码库后才能确认。",
            "回答保持简洁清晰。",
        ],
        output_format="直接输出中文答案。",
    )
    prompt = (
        f"{prompt}\n\n{render_skill_prompt_section(state.get('skill_context'), include_full_content=True)}"
        if state.get("skill_context")
        else prompt
    )

    answer = _normalize_answer(call_llm(prompt, system=DIRECT_ANSWER_SYSTEM, max_tokens=500))
    new_state = {
        **state,
        "draft_answer": "",
        "grounding_passed": False,
        "grounding_score": 0.0,
        "unsupported_claims": [],
        "missing_evidence": [],
        "verification_action": "skipped_no_retrieval",
        "verification_reason": "No retrieval was performed for this request.",
        "final_answer": answer or "无法回答该问题。",
    }
    callback = new_state.get("progress_callback")
    if callback:
        callback("direct_answer", "正在生成答案...")
    return new_state


def rewrite_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """对原始问题做查询改写，并追加退一步问题扩展。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="rewrite")
    query = state["query"]
    if _is_symbol_lookup_query(query):
        identifiers = _query_identifiers(query)
        rewritten = f"{identifiers[-1]} 定义位置" if identifiers else query
        callback = state.get("progress_callback")
        if callback:
            callback("rewrite", "正在改写符号定位查询...")
        return {
            **state,
            "rewritten_query": rewritten,
            "expanded_query": rewritten,
            "step_back_context": "",
        }
    memory_hint = _compact_memory_for_rewrite(state.get("memory_context"))
    context_sections = [("原问题", query)]
    if memory_hint:
        context_sections.append(("补充记忆线索", memory_hint))
    if memory_hint:
        prompt = build_prompt(
            task="先判断补充记忆线索是否和当前问题相关，再输出适合代码检索的改写查询语句。",
            context_sections=context_sections,
            rules=[
                "只有当补充记忆线索能帮助消解代词、省略、上文指代或项目内既有叫法时，才将 use_memory 设为 true。",
                "如果当前问题已经足够自洽，或补充记忆线索与当前问题无关，必须将 use_memory 设为 false，并忽略这些线索。",
                "如果补充记忆线索与当前问题冲突，优先以当前问题为准。",
                "rewritten_query 必须保留函数名、类名、路径名、接口名、配置名等精确术语。",
                "不要扩展不存在的术语或同义词猜测。",
                "rewritten_query 控制在 50 字以内。",
                "reason 使用简短中文说明是否使用了记忆线索。",
            ],
            examples=[
                (
                    "原问题：那个模块里的重试逻辑在哪里？ 补充记忆线索：- The user is tracing auth retry flow.",
                    '{"use_memory": true, "reason": "当前问题存在省略，记忆线索可补足主题", "rewritten_query": "auth retry 重试逻辑 模块位置"}',
                ),
                (
                    "原问题：如何实现 Python 装饰器？ 补充记忆线索：- The user is tracing auth retry flow.",
                    '{"use_memory": false, "reason": "当前问题自洽，记忆线索无关", "rewritten_query": "Python 装饰器 实现方式"}',
                ),
            ],
            output_format=build_json_output(
                """{
  "use_memory": false,
  "reason": "中文原因",
  "rewritten_query": "改写后的检索语句"
}""",
                extra_rules=[
                    "只返回 JSON 对象，不要输出额外解释。",
                ],
            ),
        )
        prompt = (
            f"{prompt}\n\n{render_skill_prompt_section(state.get('skill_context'), include_full_content=False)}"
            if state.get("skill_context")
            else prompt
        )
        result = call_llm_json(prompt, system=REWRITE_SYSTEM, max_tokens=180)
        rewritten = ""
        if result and isinstance(result, dict):
            rewritten = _clean_rewritten_query(str(result.get("rewritten_query", "") or ""))
        if not rewritten:
            rewritten = query
    else:
        prompt = build_prompt(
            task="把用户问题改写成适合代码检索的简洁查询语句。",
            context_sections=context_sections,
            rules=[
                "保留函数名、类名、路径名、接口名、配置名等精确术语。",
                "删除口语化修饰，但不要改变用户原意。",
                "不要扩展不存在的术语或同义词猜测。",
                "输出控制在 50 字以内。",
            ],
            examples=[
                (
                    "我想知道 SupervisorAgent 这个类在哪个文件里定义。",
                    "SupervisorAgent 类定义 文件位置",
                ),
                (
                    "为什么 api/main.py 里的 /chat 会走到监督路由器",
                    "api/main.py /chat 监督路由 调用流程",
                ),
            ],
            output_format="只输出改写后的检索语句。",
        )
        prompt = (
            f"{prompt}\n\n{render_skill_prompt_section(state.get('skill_context'), include_full_content=False)}"
            if state.get("skill_context")
            else prompt
        )
        rewritten = _clean_rewritten_query(call_llm(prompt, system=REWRITE_SYSTEM, max_tokens=100))

    callback = state.get("progress_callback")
    if callback:
        callback("rewrite", "正在改写查询...")

    step_back_context = ""
    expanded_query = rewritten or query
    try:
        sb_prompt = build_prompt(
            task="把用户的具体问题抽象成一个更高层的退一步问题，用于补充通用背景知识。",
            context_sections=[("用户问题", query)],
            rules=[
                "退一步问题应保留原问题的主题，不要偏离到无关方向。",
                "只输出一句问题。",
            ],
            output_format="只输出退一步问题。",
        )
        sb_question = (
            call_llm(sb_prompt, system=STEP_BACK_QUESTION_SYSTEM, max_tokens=80) or ""
        ).strip()
        if sb_question:
            sb_ans_prompt = build_prompt(
                task="回答这个退一步问题，为后续检索提供通用背景知识。",
                context_sections=[("退一步问题", sb_question)],
                rules=[
                    "只提供通用背景知识，不要猜测当前仓库里的具体实现。",
                    "控制在 120 字以内。",
                ],
                output_format="只输出答案。",
            )
            sb_answer = (
                call_llm(sb_ans_prompt, system=STEP_BACK_ANSWER_SYSTEM, max_tokens=150) or ""
            ).strip()
            step_back_context = f"退一步问题：{sb_question}\n退一步答案：{sb_answer}"
            expanded_query = f"{rewritten or query}\n\n{step_back_context}"
    except Exception:
        pass

    return {
        **state,
        "rewritten_query": rewritten or query,
        "expanded_query": expanded_query,
        "step_back_context": step_back_context,
    }


def retrieve_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """从已配置的知识库集合中执行检索。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="retrieve")
    query = state.get("expanded_query") or state.get("rewritten_query") or state["query"]

    callback = state.get("progress_callback")
    if callback:
        callback("retrieve", "正在检索知识库...")

    try:
        hits = search_knowledge_base(query, top_k=12)
    except Exception:
        hits = []

    return {**state, "retrieved_docs": hits}


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [pid for pid, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_store = get_parent_chunk_store()
    parent_docs = parent_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        parent_doc["merged_from_children"] = True
        merged.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen: set = set()
    for item in merged:
        key = _doc_identity(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _doc_identity(doc: dict) -> tuple[Any, ...]:
    chunk_id = str(doc.get("chunk_id", "")).strip()
    if chunk_id:
        return ("chunk", chunk_id)

    source_label = str(doc.get("source_label", "")).strip()
    file_path = str(doc.get("file_path", "")).strip()
    func_name = str(doc.get("func_name", "")).strip()
    if file_path or func_name:
        return ("codebase", file_path, func_name)

    doc_id = str(doc.get("doc_id", "")).strip()
    title = str(doc.get("title", "")).strip()
    source = str(doc.get("source", "")).strip()
    if doc_id or title:
        return ("apidocs", doc_id, source, title)

    chunk_index = doc.get("chunk_index", doc.get("chunk_idx", None))
    if source or chunk_index is not None:
        return ("documents", source, chunk_index)
    if source_label:
        return ("documents", source_label)
    return ("documents", id(doc))


def _select_docs_by_indices(docs: list[dict], indices: list[Any], *, limit: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple[Any, ...]] = set()
    for idx in indices[:limit]:
        if not isinstance(idx, int) or idx < 0 or idx >= len(docs):
            continue
        doc = docs[idx]
        key = _doc_identity(doc)
        if key in seen:
            continue
        selected.append(doc)
        seen.add(key)
    return selected


def _build_evidence_summary_lines(evidence_bundle: list[dict]) -> list[str]:
    lines: list[str] = []
    for evidence in evidence_bundle:
        source_label = str(evidence.get("source_label", "")).strip() or "unknown"
        reason = str(evidence.get("reason", "")).strip()
        key_points = [
            str(item).strip()
            for item in evidence.get("key_points", [])
            if str(item).strip()
        ]
        pieces = [f"[{source_label}]"]
        if reason:
            pieces.append(reason)
        if key_points:
            pieces.append("；".join(key_points))
        lines.append(" ".join(piece for piece in pieces if piece))
    return lines


def auto_merge_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """把命中的子块向上合并到父块，减少答案上下文碎片化。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="auto_merge")
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {**state, "merged_docs": []}

    callback = state.get("progress_callback")
    if callback:
        callback("auto_merge", "正在合并检索结果...")

    merged, _ = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged, _ = _merge_to_parent_level(merged, threshold=AUTO_MERGE_THRESHOLD)
    merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return {**state, "merged_docs": merged[:12]}


def rerank_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """使用 LLM 从候选检索结果中选择主要证据和备用证据。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="rerank")
    docs = state.get("merged_docs") or state.get("retrieved_docs", [])
    if not docs:
        return {
            **state,
            "reranked_docs": [],
            "primary_docs": [],
            "backup_docs": [],
            "evidence_bundle": [],
        }
    if _is_symbol_lookup_query(state["query"]):
        exact_doc = _match_symbol_lookup_doc(state["query"], docs)
        if exact_doc is not None:
            source_label = (
                str(exact_doc.get("file_path", "") or "").strip()
                or str(exact_doc.get("source_label", "") or "").strip()
                or "unknown"
            )
            backup_docs = [doc for doc in docs if _doc_identity(doc) != _doc_identity(exact_doc)][:1]
            return {
                **state,
                "reranked_docs": [exact_doc],
                "primary_docs": [exact_doc],
                "backup_docs": backup_docs,
                "evidence_bundle": [
                    {
                        "index": 0,
                        "source_type": exact_doc.get("source_type", "codebase"),
                        "source_label": source_label,
                        "reason": "精确符号命中，直接定位到定义。",
                        "key_points": [str(exact_doc.get("func_name", "") or source_label)],
                    }
                ],
            }

    callback = state.get("progress_callback")
    if callback:
        callback("rerank", "正在重排...")

    query = state["query"]
    doc_list = []
    for i, doc in enumerate(docs):
        snippet = doc.get("text") or doc.get("content") or doc.get("code", "")
        source_type = doc.get("source_type", "unknown")
        source_label = (
            doc.get("source_label")
            or doc.get("title")
            or doc.get("source")
            or doc.get("file_path", "unknown")
        )
        doc_list.append(
            f"[{i}] source_type={source_type} source_label={source_label} "
            f"fusion_score={float(doc.get('fusion_score') or 0.0):.4f}\n{snippet[:280]}"
        )

    docs_text = "\n\n".join(doc_list)
    prompt = build_prompt(
        task="从候选检索结果中选择主要证据和备用证据，并提炼应保留的关键信息。",
        context_sections=[
            ("用户问题", query),
            ("候选检索结果", docs_text),
        ],
        rules=[
            "primary_indices 最多保留 4 个，backup_indices 最多保留 2 个。",
            "优先选择能直接回答问题的证据；当多个来源互补时，可以同时保留。",
            "不要只按关键词重合选择，优先考虑来源类型、符号名、模块职责和是否直接支持答案。",
            "当不确定时，把候选放入 backup，而不是直接丢弃。",
        ],
        examples=[
            (
                "用户问题：SupervisorAgent 的 run 方法在哪里定义？ 候选 0：README 中提到 SupervisorAgent。候选 1：supervisor.py 中的 class SupervisorAgent 有 run 方法定义。",
                '{"primary_indices": [1], "backup_indices": [0], "evidence": [{"index": 1, "source_type": "codebase", "source_label": "supervisor.py::SupervisorAgent", "reason": "直接包含 run 方法定义", "key_points": ["SupervisorAgent 类中包含 run 方法"]}]}',
            ),
        ],
        output_format=build_json_output(
            """{
  "primary_indices": [1, 4, 2],
  "backup_indices": [0, 3],
  "evidence": [
    {
      "index": 1,
      "source_type": "documents",
      "source_label": "team_playbook.md",
      "reason": "直接回答发布前流程要求",
      "key_points": ["Blue Canary 发布前必须先运行 smoke checklist"]
    }
  ]
}""",
            extra_rules=[
                "只返回 JSON 对象，不要输出额外解释。",
            ],
        ),
    )

    result = call_llm_json(prompt, system=RERANK_SYSTEM, max_tokens=220)

    primary_docs: list[dict] = []
    backup_docs: list[dict] = []
    evidence_bundle: list[dict] = []
    if result and isinstance(result, dict):
        primary_docs = _select_docs_by_indices(docs, list(result.get("primary_indices", [])), limit=4)
        used_keys = {_doc_identity(doc) for doc in primary_docs}
        backup_candidates = _select_docs_by_indices(docs, list(result.get("backup_indices", [])), limit=2)
        backup_docs = [doc for doc in backup_candidates if _doc_identity(doc) not in used_keys][:2]

        raw_evidence = result.get("evidence", [])
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= len(docs):
                    continue
                doc = docs[idx]
                evidence_bundle.append(
                    {
                        "index": idx,
                        "source_type": item.get("source_type") or doc.get("source_type", "unknown"),
                        "source_label": item.get("source_label")
                        or doc.get("source_label")
                        or doc.get("title")
                        or doc.get("source")
                        or doc.get("file_path", "unknown"),
                        "reason": str(item.get("reason", "")).strip(),
                        "key_points": [
                            str(point).strip()
                            for point in item.get("key_points", [])
                            if str(point).strip()
                        ],
                    }
                )

    if not primary_docs:
        primary_docs = docs[:4]
    if not backup_docs:
        used_keys = {_doc_identity(doc) for doc in primary_docs}
        backup_docs = [
            doc for doc in docs[4:6]
            if _doc_identity(doc) not in used_keys
        ][:2]

    if not evidence_bundle:
        for index, doc in enumerate(primary_docs):
            source_label = (
                doc.get("source_label")
                or doc.get("title")
                or doc.get("source")
                or doc.get("file_path", "unknown")
            )
            snippet = str(doc.get("text") or doc.get("content") or doc.get("code", "")).strip()
            evidence_bundle.append(
                {
                    "index": index,
                    "source_type": doc.get("source_type", "unknown"),
                    "source_label": source_label,
                    "reason": "fallback 保留的高优先级证据",
                    "key_points": [snippet[:140]] if snippet else [],
                }
            )

    return {
        **state,
        "reranked_docs": primary_docs,
        "primary_docs": primary_docs,
        "backup_docs": backup_docs,
        "evidence_bundle": evidence_bundle,
    }


def generate_draft_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """基于重排序后的上下文生成最终答案。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="generate_draft")
    query = state["query"]
    docs, context, evidence_summary, memory_context = _build_context_bundle(state)
    if _is_symbol_lookup_query(query):
        exact_doc = _match_symbol_lookup_doc(query, docs)
        if exact_doc is not None:
            answer = _build_symbol_lookup_answer(exact_doc)
            return {
                **state,
                "draft_answer": answer,
                "final_answer": answer,
            }

    callback = state.get("progress_callback")
    if callback:
        callback("generate_draft", "正在生成答案草稿...")

    if not docs:
        return {
            **state,
            "draft_answer": ABSTAIN_ANSWER_TEMPLATE,
            "grounding_passed": False,
            "grounding_score": 0.0,
            "unsupported_claims": [],
            "missing_evidence": ["知识库中未找到相关内容。"],
            "verification_action": "abstain",
            "verification_reason": "No supporting evidence was retrieved.",
            "final_answer": ABSTAIN_ANSWER_TEMPLATE,
        }

    step_back = state.get("step_back_context", "")
    sb_section = f"\n\n背景知识：\n{step_back}" if step_back else ""
    context_sections = [("相关内容", f"{context}{sb_section}")]
    if evidence_summary:
        context_sections.append(("证据摘要", evidence_summary))
    if memory_context:
        context_sections.append(("补充记忆上下文", memory_context))
    context_sections.append(("用户问题", query))
    prompt = build_prompt(
        task="根据检索到的代码上下文回答用户问题。",
        context_sections=context_sections,
        rules=[
            "先给出直接结论，再按来源说明各自提供了什么信息。",
            "只允许使用相关内容和证据摘要中出现的信息，不要补充未提供的事实。",
            "如果多个来源共同支持结论，请分别说明它们提供了什么信息。",
            "如果证据不足或来源之间无法支持结论，必须明确说明。",
            "不要输出思考过程。",
        ],
        output_format="直接输出中文答案；当引用来源时，用 `[source_label]` 标注即可。",
    )
    prompt = (
        f"{prompt}\n\n{render_skill_prompt_section(state.get('skill_context'), include_full_content=True)}"
        if state.get("skill_context")
        else prompt
    )
    answer = _normalize_answer(call_llm(prompt, system=GENERATE_SYSTEM, max_tokens=800))
    return {
        **state,
        "draft_answer": answer or ABSTAIN_ANSWER_TEMPLATE,
        "final_answer": answer or ABSTAIN_ANSWER_TEMPLATE,
    }

    step_back = state.get("step_back_context", "")
    sb_section = f"\n\n背景知识：\n{step_back}" if step_back else ""
    context_sections = [("相关内容", f"{context}{sb_section}")]
    if evidence_summary:
        context_sections.append(("证据摘要", evidence_summary))
    if memory_context:
        context_sections.append(("补充记忆上下文", memory_context))
    context_sections.append(("用户问题", query))
    prompt = build_prompt(
        task="根据检索到的代码上下文回答用户问题。",
        context_sections=context_sections,
        rules=[
            "先给出直接结论，再按来源说明各自提供了什么信息。",
            "只允许使用相关内容和证据摘要中出现的信息，不要补充未提供的事实。",
            "如果多个来源共同支持结论，请分别说明它们提供了什么信息。",
            "如果证据不足或来源之间无法支撑结论，必须明确说明。",
            "不要输出思考过程。",
        ],
        output_format="直接输出中文答案；当引用来源时，用 `[source_label]` 标注即可。",
    )
    prompt = (
        f"{prompt}\n\n{render_skill_prompt_section(state.get('skill_context'), include_full_content=True)}"
        if state.get("skill_context")
        else prompt
    )

    answer = call_llm(prompt, system=GENERATE_SYSTEM, max_tokens=800)
    for pattern in (
        r"<\|ankton\w*\|>.*?<\|/ankton\|>",
        r"<\|thinking\|>.*?<\|end\|>",
        r"<\|thought\|>.*?<\|end\|>",
    ):
        answer = re.sub(pattern, "", answer or "", flags=re.DOTALL)
    answer = answer.strip()

    return {**state, "final_answer": answer or "无法基于检索结果生成答案。"}


def verify_grounding_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """校验答案是否被检索证据充分支撑。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="verify_grounding")

    docs, context, evidence_summary, _ = _build_context_bundle(state)
    if not docs:
        return {
            **state,
            "grounding_passed": False,
            "grounding_score": 0.0,
            "unsupported_claims": [],
            "missing_evidence": ["知识库中未找到相关内容。"],
            "verification_action": "abstain",
            "verification_reason": "No supporting evidence was retrieved.",
            "final_answer": ABSTAIN_ANSWER_TEMPLATE,
        }
    if _is_symbol_lookup_query(state["query"]) and _match_symbol_lookup_doc(state["query"], docs) is not None:
        return {
            **state,
            "grounding_passed": True,
            "grounding_score": 1.0,
            "unsupported_claims": [],
            "missing_evidence": [],
            "verification_action": "pass",
            "verification_reason": "Exact symbol lookup answer is directly grounded by the retrieved definition.",
            "final_answer": state.get("draft_answer", "") or state.get("final_answer", ""),
        }

    callback = state.get("progress_callback")
    if callback:
        callback("verify_grounding", "正在校验回答是否被证据支撑...")

    prompt = build_prompt(
        task="判断候选回答是否被检索证据直接支撑。",
        context_sections=[
            ("用户问题", state["query"]),
            ("候选回答", state.get("draft_answer", "")),
            ("检索内容", context),
            ("证据摘要", evidence_summary or "（无证据摘要）"),
        ],
        rules=[
            "grounding_score 取 0 到 1 之间的小数。",
            "只要回答中包含无法从证据直接推出的确定性结论，就要列入 unsupported_claims。",
            "missing_evidence 用来指出回答还缺哪些证据。",
        ],
        output_format=build_json_output(
            """{
  "grounding_passed": true,
  "grounding_score": 0.92,
  "unsupported_claims": [],
  "missing_evidence": [],
  "verification_reason": "回答中的结论均能从检索证据推出。"
}"""
        ),
    )

    result = call_llm_json(prompt, system=VERIFY_GROUNDING_SYSTEM, max_tokens=300)
    revision_attempted = bool(state.get("revision_attempted"))
    grounding_score = _coerce_grounding_score(
        result.get("grounding_score") if isinstance(result, dict) else None,
        default=1.0 if isinstance(result, dict) and result.get("grounding_passed") else 0.0,
    )
    action = _decide_verification_action(grounding_score, revision_attempted=revision_attempted)
    verification_reason = (
        str(result.get("verification_reason", "")).strip()
        if isinstance(result, dict)
        else ""
    ) or "Grounding verification fell back to a conservative decision."
    unsupported_claims = _normalize_text_list(result.get("unsupported_claims") if isinstance(result, dict) else [])
    missing_evidence = _normalize_text_list(result.get("missing_evidence") if isinstance(result, dict) else [])
    grounding_passed = action == "pass"
    final_answer = state.get("draft_answer", "")
    if action == "abstain":
        final_answer = ABSTAIN_ANSWER_TEMPLATE

    return {
        **state,
        "grounding_passed": grounding_passed,
        "grounding_score": grounding_score,
        "unsupported_claims": unsupported_claims,
        "missing_evidence": missing_evidence,
        "verification_action": action,
        "verification_reason": verification_reason,
        "final_answer": final_answer,
    }


def revise_answer_node(state: EnhancedRAGState) -> EnhancedRAGState:
    """在 groundedness 校验未通过时保守重写答案。"""
    if _is_cancellation_requested(state):
        return _cancelled_state(state, stage="revise_answer")

    docs, context, evidence_summary, memory_context = _build_context_bundle(state)
    callback = state.get("progress_callback")
    if callback:
        callback("revise_answer", "正在根据校验结果保守重写答案...")

    if not docs:
        return {
            **state,
            "revision_attempted": True,
            "draft_answer": ABSTAIN_ANSWER_TEMPLATE,
            "final_answer": ABSTAIN_ANSWER_TEMPLATE,
        }

    context_sections = [
        ("用户问题", state["query"]),
        ("原始回答", state.get("draft_answer", "")),
        ("未被证据支持的内容", "\n".join(state.get("unsupported_claims", [])) or "（无）"),
        ("缺失证据", "\n".join(state.get("missing_evidence", [])) or "（无）"),
        ("检索内容", context),
    ]
    if evidence_summary:
        context_sections.append(("证据摘要", evidence_summary))
    if memory_context:
        context_sections.append(("补充记忆上下文", memory_context))
    prompt = build_prompt(
        task="在不增加新事实的前提下重写回答，删除 unsupported claims。",
        context_sections=context_sections,
        rules=[
            "只能保留检索内容中能直接支撑的事实。",
            "如果删除 unsupported claims 后无法形成可靠答案，就明确说明证据不足。",
            "保持答案简洁，并继续使用 `[source_label]` 标注来源。",
        ],
        output_format="直接输出中文修订答案。",
    )

    revised_answer = _normalize_answer(
        call_llm(prompt, system=REVISE_GROUNDED_ANSWER_SYSTEM, max_tokens=600)
    )
    return {
        **state,
        "revision_attempted": True,
        "draft_answer": revised_answer or ABSTAIN_ANSWER_TEMPLATE,
        "final_answer": revised_answer or ABSTAIN_ANSWER_TEMPLATE,
    }


def build_enhanced_rag_graph():
    """构建增强版 RAG 工作流图。"""
    graph = StateGraph(EnhancedRAGState)

    graph.add_node("self_rag", self_rag_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("auto_merge", auto_merge_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_draft", generate_draft_node)
    graph.add_node("verify_grounding", verify_grounding_node)
    graph.add_node("revise_answer", revise_answer_node)

    graph.set_entry_point("self_rag")
    graph.add_conditional_edges(
        "self_rag",
        _route_after_self_rag,
        {"cancelled": END, "rewrite": "rewrite", "direct_answer": "direct_answer"},
    )

    graph.add_edge("direct_answer", END)
    graph.add_conditional_edges(
        "rewrite",
        _route_if_not_cancelled("retrieve"),
        {"cancelled": END, "retrieve": "retrieve"},
    )
    graph.add_conditional_edges(
        "retrieve",
        _route_if_not_cancelled("auto_merge"),
        {"cancelled": END, "auto_merge": "auto_merge"},
    )
    graph.add_conditional_edges(
        "auto_merge",
        _route_if_not_cancelled("rerank"),
        {"cancelled": END, "rerank": "rerank"},
    )
    graph.add_conditional_edges(
        "rerank",
        _route_if_not_cancelled("generate_draft"),
        {"cancelled": END, "generate_draft": "generate_draft"},
    )
    graph.add_conditional_edges(
        "generate_draft",
        _route_if_not_cancelled("verify_grounding"),
        {"cancelled": END, "verify_grounding": "verify_grounding"},
    )
    graph.add_conditional_edges(
        "verify_grounding",
        _route_after_verification,
        {"cancelled": END, "pass": END, "revise": "revise_answer", "abstain": END},
    )
    graph.add_conditional_edges(
        "revise_answer",
        _route_if_not_cancelled("verify_grounding"),
        {"cancelled": END, "verify_grounding": "verify_grounding"},
    )

    return graph.compile()


_enhanced_rag_graph = build_enhanced_rag_graph()


class EnhancedRAGAgent:
    """提供查询改写、检索、重排和生成能力的增强版 RAG Agent。"""

    def __init__(self):
        self._graph = _enhanced_rag_graph

    def invoke(self, input_dict: dict) -> dict:
        """同步执行增强版 RAG 问答。"""
        query = input_dict.get("query", "")
        messages = input_dict.get("messages", [])

        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                query = last_msg.get("content", query)
            else:
                query = last_msg.content

        if not query:
            return {**input_dict, "final_answer": "未提供查询内容。"}

        initial_state: EnhancedRAGState = {
            "query": query,
            "rewritten_query": "",
            "expanded_query": "",
            "step_back_context": "",
            "retrieved_docs": [],
            "merged_docs": [],
            "reranked_docs": [],
            "primary_docs": [],
            "backup_docs": [],
            "evidence_bundle": [],
            "draft_answer": "",
            "needs_retrieval": True,
            "retrieval_target": "codebase",
            "grounding_passed": False,
            "grounding_score": 0.0,
            "unsupported_claims": [],
            "missing_evidence": [],
            "verification_action": "",
            "verification_reason": "",
            "revision_attempted": False,
            "final_answer": "",
            "memory_context": str(input_dict.get("memory_context") or "").strip() or None,
            "skill_context": input_dict.get("skill_context"),
            "messages": [],
            "progress_callback": None,
            "cancel_event": input_dict.get("cancel_event"),
            "cancelled": False,
        }

        result = self._graph.invoke(initial_state)
        return {
            **input_dict,
            "final_answer": result.get("final_answer", ""),
            "messages": messages + [AIMessage(content=result.get("final_answer", ""))],
            "_rag_result": result,
        }

    def stream_invoke(self, input_dict: dict, progress_callback=None) -> dict:
        """流式执行增强版 RAG 问答，并通过回调上报阶段进度。"""
        query = input_dict.get("query", "")
        messages = input_dict.get("messages", [])

        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                query = last_msg.get("content", query)
            else:
                query = last_msg.content

        if not query:
            return {**input_dict, "final_answer": "未提供查询内容。"}

        initial_state: EnhancedRAGState = {
            "query": query,
            "rewritten_query": "",
            "expanded_query": "",
            "step_back_context": "",
            "retrieved_docs": [],
            "merged_docs": [],
            "reranked_docs": [],
            "primary_docs": [],
            "backup_docs": [],
            "evidence_bundle": [],
            "draft_answer": "",
            "needs_retrieval": True,
            "retrieval_target": "codebase",
            "grounding_passed": False,
            "grounding_score": 0.0,
            "unsupported_claims": [],
            "missing_evidence": [],
            "verification_action": "",
            "verification_reason": "",
            "revision_attempted": False,
            "final_answer": "",
            "memory_context": str(input_dict.get("memory_context") or "").strip() or None,
            "skill_context": input_dict.get("skill_context"),
            "messages": [],
            "progress_callback": progress_callback,
            "cancel_event": input_dict.get("cancel_event"),
            "cancelled": False,
        }

        result = self._graph.invoke(initial_state)
        return {
            **input_dict,
            "final_answer": result.get("final_answer", ""),
            "messages": messages + [AIMessage(content=result.get("final_answer", ""))],
            "_rag_result": result,
        }


_cached_agent: EnhancedRAGAgent | None = None


def get_enhanced_rag_agent() -> EnhancedRAGAgent:
    """获取增强版 RAG Agent 单例。"""
    global _cached_agent
    if _cached_agent is None:
        _cached_agent = EnhancedRAGAgent()
    return _cached_agent

"""CodeSage PR 审查编排模块。"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from codesage.agents.review import LogicAgent, RAGAgent, SecurityAgent, SupervisorAgent
from codesage.core.config import PR_REVIEW_MAX_DIFF_BYTES
from codesage.tools.github_tools import GitHubTransportError, get_pr_diff, post_review_comment
from codesage.tools.llm_tools import call_llm
from codesage.tools.prompt_tools import build_prompt, build_system_prompt
from codesage.tools.review_guards import ReviewInputError, validate_review_diff

logger = logging.getLogger(__name__)
CANCELLED_REVIEW_MESSAGE = "PR 审查在完成前已被取消。"


class ReviewState(TypedDict):
    repo: str
    pr_number: int
    diff_text: str
    diff_chunks: list
    py_chunks: list
    run_rag: bool
    run_security: bool
    rag_summary: str
    rag_context_by_file: dict
    security_issues: list
    logic_issues: list
    final_comment: str
    cancel_event: Any
    cancelled: bool


def _is_cancellation_requested(state: ReviewState) -> bool:
    cancel_event = state.get("cancel_event")
    return bool(cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)())


class PRReviewAgent:
    """编排完整 PR 审查工作流。"""

    def __init__(self):
        self.name = "PRReviewAgent"
        self.graph = None

    def _build_graph(self):
        graph = StateGraph(ReviewState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("checks", self._checks_node)
        graph.add_node("logic", self._logic_node)
        graph.add_node("aggregator", self._aggregator_node)
        graph.add_node("post_comment", self._post_comment_node)

        graph.set_entry_point("supervisor")
        graph.add_edge("supervisor", "checks")
        graph.add_edge("checks", "logic")
        graph.add_edge("logic", "aggregator")
        graph.add_edge("aggregator", "post_comment")
        graph.add_edge("post_comment", END)
        return graph.compile()

    def _supervisor_node(self, state: ReviewState) -> ReviewState:
        if _is_cancellation_requested(state):
            return {**state, "cancelled": True, "final_comment": CANCELLED_REVIEW_MESSAGE}
        agent = SupervisorAgent()
        result = agent.run(state["diff_text"])
        return {**state, **result}

    def _rag_node(self, state: ReviewState) -> ReviewState:
        if state.get("cancelled") or _is_cancellation_requested(state):
            return {**state, "cancelled": True, "rag_summary": "", "rag_context_by_file": {}}
        if not state.get("run_rag"):
            return {**state, "rag_summary": "", "rag_context_by_file": {}}

        agent = RAGAgent()
        context = agent.run(state["py_chunks"])
        hit_count = int(context.get("hit_count") or 0)
        by_file = dict(context.get("by_file") or {})
        hits_by_file = dict(context.get("hits_by_file") or {})
        fallback_used = bool(context.get("fallback_used") or False)
        logger.info(
            "PR review RAG: run_rag=%s hit_count=%s files=%s per_file=%s fallback=%s",
            True,
            hit_count,
            len(by_file),
            {file_path: len(hits) for file_path, hits in hits_by_file.items()},
            fallback_used,
        )
        return {
            **state,
            "rag_summary": str(context.get("summary", "") or ""),
            "rag_context_by_file": by_file,
        }

    def _security_node(self, state: ReviewState) -> ReviewState:
        if state.get("cancelled") or _is_cancellation_requested(state):
            return {**state, "cancelled": True, "security_issues": []}
        if not state.get("run_security"):
            return {**state, "security_issues": []}
        agent = SecurityAgent()
        issues = agent.run(state["diff_chunks"])
        return {**state, "security_issues": issues}

    def _checks_node(self, state: ReviewState) -> ReviewState:
        if state.get("cancelled") or _is_cancellation_requested(state):
            return {
                **state,
                "cancelled": True,
                "rag_summary": "",
                "rag_context_by_file": {},
                "security_issues": [],
            }
        run_rag = bool(state.get("run_rag"))
        run_security = bool(state.get("run_security"))

        if not run_rag and not run_security:
            return {**state, "rag_summary": "", "rag_context_by_file": {}, "security_issues": []}

        if run_rag and run_security:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pr-review") as executor:
                rag_future = executor.submit(self._rag_node, dict(state))
                security_future = executor.submit(self._security_node, dict(state))
                rag_result = rag_future.result()
                security_result = security_future.result()
            return {
                **state,
                "rag_summary": rag_result.get("rag_summary", ""),
                "rag_context_by_file": rag_result.get("rag_context_by_file", {}),
                "security_issues": security_result.get("security_issues", []),
            }

        if run_rag:
            rag_result = self._rag_node(state)
            return {
                **state,
                "rag_summary": rag_result.get("rag_summary", ""),
                "rag_context_by_file": rag_result.get("rag_context_by_file", {}),
                "security_issues": [],
            }

        security_result = self._security_node(state)
        return {
            **state,
            "rag_summary": "",
            "rag_context_by_file": {},
            "security_issues": security_result.get("security_issues", []),
        }

    def _logic_node(self, state: ReviewState) -> ReviewState:
        if state.get("cancelled") or _is_cancellation_requested(state):
            return {**state, "cancelled": True, "logic_issues": []}
        agent = LogicAgent()
        issues = agent.run(
            state["diff_chunks"],
            state.get("rag_context_by_file", {}),
            state.get("rag_summary", ""),
        )
        return {**state, "logic_issues": issues}

    def _aggregator_node(self, state: ReviewState) -> ReviewState:
        if state.get("cancelled") or _is_cancellation_requested(state):
            return {**state, "cancelled": True, "final_comment": CANCELLED_REVIEW_MESSAGE}
        security_issues = state.get("security_issues", [])
        logic_issues = state.get("logic_issues", [])
        rag_summary = state.get("rag_summary", "")

        findings_parts = []
        if security_issues:
            findings_parts.append(
                "### 安全问题\n"
                + "\n".join(
                    f"- {item['file']}: {item['issue']} (`{item['snippet']}`)"
                    for item in security_issues
                )
            )
        if logic_issues:
            findings_parts.append(
                "### 逻辑与实现问题\n"
                + "\n".join(
                    f"- {item['file']} [{item.get('type', 'unknown')}]: {item['description'][:120]}"
                    for item in logic_issues
                )
            )

        if not findings_parts:
            comment = "## CodeSage 审查报告\n\n本次变更整体风险较低，未发现明确的安全或逻辑问题。"
            return {**state, "final_comment": comment}

        findings = "\n\n".join(findings_parts)
        context_hint = f"\n\n### 历史上下文参考\n{rag_summary[:400]}" if rag_summary else ""
        system_prompt = build_system_prompt(
            role="你是代码审查汇总专家，负责把多个子 Agent 的发现整合成一条专业、可执行的 PR 评论。",
            responsibilities=[
                "优先突出会影响正确性、安全性和兼容性的发现。",
                "把问题表述成便于开发者采取行动的建议。",
                "输出结构清晰、适合直接发布到 PR 的 Markdown。",
            ],
            rules=[
                "不要夸大不确定问题；证据不足时保持克制。",
                "没有问题时要明确说明未发现问题。",
                "避免重复描述同一问题。",
            ],
            output_instruction="直接输出中文 Markdown，不要输出额外说明或思考过程。",
        )
        prompt = build_prompt(
            task="根据自动检测结果生成一条专业的代码审查评论。",
            context_sections=[("检测发现", f"{findings}{context_hint}")],
            rules=[
                "标题使用 `## CodeSage 审查报告`。",
                "先列问题，再给出简短结论。",
                "如果包含历史上下文，只保留和本次变更直接相关的部分。",
                "避免输出无法从已知信息直接支持的判断。",
            ],
            output_format="""请输出 Markdown，结构建议如下：
## CodeSage 审查报告
### 安全问题
- ...

### 逻辑与实现问题
- ...

### 总结
- ...""",
        )

        try:
            comment = call_llm(prompt, system=system_prompt, max_tokens=800)
        except Exception:
            parts = ["## CodeSage 审查报告\n"]
            if security_issues:
                parts.append("### 安全问题")
                for item in security_issues:
                    parts.append(f"- **{item['file']}**: {item['issue']}（片段：`{item['snippet']}`）")
            if logic_issues:
                parts.append("### 逻辑与实现问题")
                for item in logic_issues:
                    parts.append(f"- **{item['file']}** [{item.get('type', '')}]: {item['description'][:120]}")
            if rag_summary:
                parts.append(f"\n### 历史上下文参考\n```\n{rag_summary[:800]}\n```")
            comment = "\n".join(parts)

        return {**state, "final_comment": comment}

    def _post_comment_node(self, state: ReviewState) -> ReviewState:
        if state.get("cancelled") or _is_cancellation_requested(state):
            return state
        if state["repo"] in ("test/repo", "") or state["pr_number"] == 0:
            return state
        post_review_comment(state["repo"], state["pr_number"], state["final_comment"])
        return state

    def _resolve_diff_or_url(self, content: str) -> tuple[str, str, int]:
        content = content.strip()

        pr_url_pattern = r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)"
        match = re.search(pr_url_pattern, content)
        if match:
            owner, repo_name, pr_num = match.groups()
            repo_full = f"{owner}/{repo_name}"
            diff = get_pr_diff(repo_full, int(pr_num))
            return diff, repo_full, int(pr_num)

        diff_match = re.match(r"review\s+diff:\s*", content, re.IGNORECASE)
        if diff_match:
            diff_text = content[diff_match.end() :].strip()
            return diff_text, "review/diff", 0

        review_this_match = re.match(r"review\s+this:\s*", content, re.IGNORECASE)
        if review_this_match:
            diff_text = content[review_this_match.end() :].strip()
            return diff_text, "review/file", 0

        review_prefix_match = re.match(r"review\s+", content, re.IGNORECASE)
        if review_prefix_match:
            diff_text = content[review_prefix_match.end() :].strip()
            return diff_text, "review/raw", 0

        return content, "review/raw", 0

    def run(self, repo: str, pr_number: int, diff_text: str, cancel_event: Any | None = None) -> dict[str, Any]:
        return self.invoke(
            {
                "repo": repo,
                "pr_number": pr_number,
                "diff_text": diff_text,
                "cancel_event": cancel_event,
            }
        )

    def invoke(self, request: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(request, dict):
            cancel_event = request.get("cancel_event")
            messages = list(request.get("messages", []) or [])
            if messages and not request.get("diff_text"):
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    content = str(last_message.get("content", "") or "")
                else:
                    content = str(getattr(last_message, "content", "") or "")
                try:
                    raw_diff, repo, pr_number = self._resolve_diff_or_url(content)
                except GitHubTransportError as exc:
                    return {"final_comment": f"获取 PR diff 失败：{exc}", "security_issues": [], "logic_issues": []}
            else:
                repo = str(request.get("repo", "") or "")
                pr_number = int(request.get("pr_number", 0) or 0)
                raw_diff = str(request.get("diff_text", "") or "")
        else:
            try:
                raw_diff, repo, pr_number = self._resolve_diff_or_url(str(request))
            except GitHubTransportError as exc:
                return {"final_comment": f"获取 PR diff 失败：{exc}", "security_issues": [], "logic_issues": []}
            cancel_event = None

        try:
            diff_text = validate_review_diff(raw_diff, PR_REVIEW_MAX_DIFF_BYTES)
        except ReviewInputError as exc:
            return {"final_comment": f"审查输入无效：{exc}", "security_issues": [], "logic_issues": []}
        except GitHubTransportError as exc:
            return {"final_comment": f"获取 PR diff 失败：{exc}", "security_issues": [], "logic_issues": []}

        if self.graph is None:
            self.graph = self._build_graph()

        initial_state: ReviewState = {
            "repo": repo,
            "pr_number": pr_number,
            "diff_text": diff_text,
            "diff_chunks": [],
            "py_chunks": [],
            "run_rag": False,
            "run_security": False,
            "rag_summary": "",
            "rag_context_by_file": {},
            "security_issues": [],
            "logic_issues": [],
            "final_comment": "",
            "cancel_event": cancel_event,
            "cancelled": False,
        }
        state = self.graph.invoke(initial_state)
        return {
            "final_comment": state.get("final_comment", ""),
            "security_issues": state.get("security_issues", []),
            "logic_issues": state.get("logic_issues", []),
            "rag_summary": state.get("rag_summary", ""),
        }

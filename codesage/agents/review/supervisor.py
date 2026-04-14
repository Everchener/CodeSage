import logging
import re

from codesage.tools.code_tools import parse_diff
from codesage.tools.llm_tools import call_llm_json
from codesage.tools.prompt_tools import build_json_output, build_prompt, build_system_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = build_system_prompt(
    role="你是 PR 审查协调器，负责根据 diff 判断是否需要额外的安全检查和历史上下文补充。",
    responsibilities=[
        "理解 PR 变更的目标、范围和潜在风险。",
        "仅在有明确信号时开启 security 和 rag，避免无关变更触发重流程。",
        "输出稳定、可解析的 JSON 决策结果。",
    ],
    rules=[
        "只有当 diff 涉及认证、授权、加密、输入处理、命令执行、SQL、文件访问、网络请求等敏感逻辑时，才将 run_security 设为 true。",
        "只有当 diff 涉及业务逻辑变更、重要重构、新增功能或需要参考历史实现时，才将 run_rag 设为 true。",
        "文案、注释、简单重命名和纯格式调整通常应返回 false/false。",
        "reason 必须简洁并直接说明判断依据。",
    ],
    output_instruction="只返回 JSON 对象，不要输出 Markdown、解释或思考过程。",
)


class SupervisorAgent:
    """分析 diff 并决定 PR 审查编排策略。"""

    def __init__(self):
        self.name = "SupervisorAgent"

    def _llm_plan(self, diff_text: str) -> dict | None:
        prompt = build_prompt(
            task="分析下面的 GitHub PR 变更，并决定是否需要运行安全检查和 RAG 上下文补充。",
            context_sections=[("PR 变更内容", f"```diff\n{diff_text[:3000]}\n```")],
            rules=[
                "如果 diff 只体现注释、文案、类型标注、简单重命名或纯样式调整，优先返回 false/false。",
                "如果 diff 引入敏感数据处理、外部输入拼接、权限判断、鉴权分支或危险调用，应将 run_security 设为 true。",
                "如果 diff 明显依赖历史实现语义、旧接口约束或跨模块协作关系，应将 run_rag 设为 true。",
                "不要仅因文件后缀是 Python 就自动开启 run_security 或 run_rag。",
            ],
            examples=[
                (
                    """diff --git a/auth.py b/auth.py
+if user.is_admin and request.headers[\"X-Bypass\"] == secret:
+    return grant_access()""",
                    """{"run_security": true, "run_rag": true, "reason": "涉及鉴权判断，且需要结合历史认证流程审查。"}""",
                ),
                (
                    """diff --git a/report.py b/report.py
-def load():
+def load_report():""",
                    """{"run_security": false, "run_rag": false, "reason": "看起来只是局部重命名，未体现敏感逻辑或历史依赖。"}""",
                ),
            ],
            output_format=build_json_output(
                """{
  "run_security": true,
  "run_rag": false,
  "reason": "原因说明"
}""",
                extra_rules=["reason 必须使用中文短句。", "只输出这三个字段。"],
            ),
        )

        result = call_llm_json(prompt, SYSTEM_PROMPT, max_tokens=300)
        if result:
            logger.info(
                "PR 编排决策: security=%s, rag=%s, reason=%s",
                result.get("run_security"),
                result.get("run_rag"),
                result.get("reason", ""),
            )
        return result

    def _meaningful_changed_lines(self, chunk: dict) -> list[str]:
        lines: list[str] = []
        for raw_line in str(chunk.get("lines", "")).splitlines():
            if not raw_line.startswith(("+", "-")) or raw_line.startswith(("+++", "---")):
                continue
            stripped = raw_line[1:].strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped in {'"""', "'''"}:
                continue
            lines.append(stripped)
        return lines

    def _is_rename_only_chunk(self, chunk: dict) -> bool:
        added_lines = [line[1:].strip() for line in str(chunk.get("lines", "")).splitlines() if line.startswith("+") and not line.startswith("+++")]
        removed_lines = [line[1:].strip() for line in str(chunk.get("lines", "")).splitlines() if line.startswith("-") and not line.startswith("---")]

        def_match_added = [re.match(r"^(async\s+def|def)\s+\w+\((.*)\)\s*:?\s*$", line) for line in added_lines]
        def_match_removed = [re.match(r"^(async\s+def|def)\s+\w+\((.*)\)\s*:?\s*$", line) for line in removed_lines]
        def_match_added = [match for match in def_match_added if match]
        def_match_removed = [match for match in def_match_removed if match]
        if len(def_match_added) == len(def_match_removed) == 1:
            return def_match_added[0].group(1) == def_match_removed[0].group(1) and def_match_added[0].group(2) == def_match_removed[0].group(2)

        class_match_added = [re.match(r"^class\s+\w+(\((.*)\))?\s*:?\s*$", line) for line in added_lines]
        class_match_removed = [re.match(r"^class\s+\w+(\((.*)\))?\s*:?\s*$", line) for line in removed_lines]
        class_match_added = [match for match in class_match_added if match]
        class_match_removed = [match for match in class_match_removed if match]
        if len(class_match_added) == len(class_match_removed) == 1:
            return (class_match_added[0].group(2) or "") == (class_match_removed[0].group(2) or "")

        return False

    def _should_run_rag_fallback(self, py_chunks: list[dict]) -> bool:
        if not py_chunks:
            return False

        risk_patterns = (
            r"\bif\b|\belif\b|\belse\b|\bfor\b|\bwhile\b|\btry\b|\bexcept\b|\bwith\b",
            r"^\s*(async\s+def|def|class)\s+",
            r"\breturn\b|\braise\b|\byield\b|\bawait\b",
            r"\bfrom\s+[A-Za-z0-9_\.]+\s+import\b|\bimport\s+[A-Za-z0-9_\.]+",
            r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\s*\(",
        )

        for chunk in py_chunks[:5]:
            meaningful_lines = self._meaningful_changed_lines(chunk)
            if not meaningful_lines:
                continue
            if self._is_rename_only_chunk(chunk) and len(meaningful_lines) <= 2:
                continue
            for line in meaningful_lines:
                if any(re.search(pattern, line) for pattern in risk_patterns):
                    return True
        return False

    def _fallback_plan(self, chunks: list[dict]) -> dict:
        py_chunks = [chunk for chunk in chunks if chunk["file"].endswith(".py")]
        return {
            "run_security": len(chunks) > 0,
            "run_rag": self._should_run_rag_fallback(py_chunks),
            "reason": "降级策略：基于文件类型和改动结构判断。",
        }

    def run(self, diff_text: str) -> dict:
        chunks = parse_diff(diff_text)
        py_chunks = [chunk for chunk in chunks if chunk["file"].endswith(".py")]

        llm_result = self._llm_plan(diff_text)
        if llm_result:
            return {
                "diff_chunks": chunks,
                "py_chunks": py_chunks,
                "run_rag": llm_result.get("run_rag", self._should_run_rag_fallback(py_chunks)),
                "run_security": llm_result.get("run_security", len(chunks) > 0),
                "reason": llm_result.get("reason", "LLM 决策"),
            }

        fallback = self._fallback_plan(chunks)
        logger.warning("PR 编排决策降级为兜底规则：%s", fallback["reason"])
        return {
            "diff_chunks": chunks,
            "py_chunks": py_chunks,
            "run_rag": fallback["run_rag"],
            "run_security": fallback["run_security"],
            "reason": fallback["reason"],
        }

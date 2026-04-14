import logging

from codesage.tools.code_tools import run_linter
from codesage.tools.llm_tools import call_llm_json
from codesage.tools.prompt_tools import build_json_output, build_prompt, build_system_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = build_system_prompt(
    role="你是 Python 代码审查专家，负责发现新增代码中的逻辑、性能和必要的风格问题。",
    responsibilities=[
        "优先识别逻辑错误、边界条件缺失、异常处理缺口、资源泄漏和性能隐患。",
        "在静态 lint 已覆盖的情况下避免重复报告同类 style 问题。",
        "输出稳定、可解析的 JSON 数组结果。",
    ],
    rules=[
        "不要输出 security 类型问题；安全问题由独立的 SecurityAgent 负责。",
        "只有从新增代码和提供的上下文中能直接看出的风险，才允许报告。",
        "如果没有明确问题，必须返回空数组 []。",
    ],
    output_instruction="只返回 JSON 数组，不要输出 Markdown、解释或思考过程。",
)


class LogicAgent:
    """负责逻辑和规范审查的 Agent。"""

    def __init__(self):
        self.name = "LogicAgent"

    def _lint(self, code: str) -> str:
        """运行 flake8 检查。"""
        return run_linter(code)

    def _llm_analyze(self, code: str, file: str, rag_context: str, lint_result: str) -> list[dict]:
        """使用 LLM 分析逻辑风险。"""
        context_sections = [
            ("文件", file),
            ("新增代码", f"```python\n{code[:1500]}\n```"),
        ]
        if rag_context:
            context_sections.append(("相关历史代码上下文", rag_context[:800]))
        if lint_result and lint_result != "No issues found.":
            context_sections.append(("已知静态检查结果（这些问题不要重复报告）", lint_result[:300]))

        prompt = build_prompt(
            task="分析新增 Python 代码中的逻辑、性能和必要的风格问题。",
            context_sections=context_sections,
            rules=[
                "优先输出 logic 和 performance 问题；只有 style 问题未被 lint 覆盖时才输出 style。",
                "不要重复转述静态检查已覆盖的格式、缩进、导入顺序等问题。",
                "不要猜测未展示代码中的状态；证据不足时返回空数组。",
                "description 必须说明具体风险和触发原因，避免泛泛而谈。",
            ],
            examples=[
                (
                    """文件：orders.py
新增代码：```python
return items[0]
```""",
                    """[{"type": "logic", "description": "直接访问 items[0]，在空列表输入时会触发 IndexError。"}]""",
                ),
                (
                    """文件：orders.py
新增代码：```python
return [item.id for item in items]
```""",
                    "[]",
                ),
            ],
            output_format=build_json_output(
                """[
  {
    "type": "logic|style|performance",
    "description": "具体问题描述"
  }
]"""
            ),
        )

        result = call_llm_json(prompt, SYSTEM_PROMPT, max_tokens=500)
        if not result or not isinstance(result, list):
            return []

        issues = []
        for item in result:
            if isinstance(item, dict):
                item["file"] = file
                issues.append(item)
        return issues

    def run(
        self,
        diff_chunks: list[dict],
        rag_context_by_file: dict[str, str],
        rag_summary: str = "",
    ) -> list[dict]:
        """执行逻辑审查。"""
        del rag_summary
        issues = []
        py_chunks = [chunk for chunk in diff_chunks if chunk["file"].endswith(".py")]

        for chunk in py_chunks[:3]:
            added_lines = [line[1:] for line in chunk["lines"].splitlines() if line.startswith("+")]
            code = "\n".join(added_lines)
            if not code.strip():
                continue

            lint_result = self._lint(code)
            if lint_result and lint_result != "No issues found.":
                issues.append(
                    {
                        "file": chunk["file"],
                        "type": "style",
                        "description": lint_result[:300],
                    }
                )

            file_rag_context = str((rag_context_by_file or {}).get(chunk["file"], "") or "")
            issues.extend(self._llm_analyze(code, chunk["file"], file_rag_context, lint_result))

        return issues

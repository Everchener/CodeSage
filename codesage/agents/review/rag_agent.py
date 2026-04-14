import logging

from codesage.agents.review.review_rag_agent import RAGAgent as StructuredRAGAgent, ReviewRAGResult
from codesage.tools.llm_tools import call_llm
from codesage.tools.milvus_tools import search_codebase
from codesage.tools.prompt_tools import build_prompt, build_system_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = build_system_prompt(
    role="你是代码审查上下文助手，负责把检索到的历史代码压缩成可供 PR 审查使用的高价值摘要。",
    responsibilities=[
        "说明历史代码片段的职责和用途。",
        "指出它与当前 PR 变更的直接关系。",
        "提炼审查时应关注的兼容性、约束和行为风险。",
    ],
    rules=[
        "仅保留和本次 PR 审查直接相关的信息。",
        "不要大段复述代码，重点总结可操作的审查线索。",
    ],
    output_instruction="直接输出中文 Markdown 摘要，不要输出额外解释或思考过程。",
)


class LegacyRAGAgent:
    """兼容旧调用方式的 RAG 审查上下文代理。"""

    def __init__(self):
        self.name = "RAGAgent"

    def _llm_summarize(self, chunks: list[dict], query: str) -> str:
        """使用 LLM 汇总与本次 PR 相关的历史上下文。"""
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks[:5]:
            context_parts.append(
                f"--- 历史代码 ({chunk['file_path']}::{chunk['func_name']}) ---\n{chunk['code'][:500]}"
            )
        context_text = "\n\n".join(context_parts)

        prompt = build_prompt(
            task="根据当前 PR 变更和检索到的历史代码，生成稳定、可读的审查上下文摘要。",
            context_sections=[
                ("PR 相关查询", f"```diff\n{query[:500]}\n```"),
                ("检索到的历史上下文", context_text),
            ],
            rules=[
                "优先总结行为约束、兼容性依赖和潜在回归风险。",
                "如果历史上下文与当前变更关系有限，必须明确说明“关系有限”。",
                "不要臆测仓库中未提供的事实。",
            ],
            output_format="""请输出 Markdown，结构如下：
### 历史职责
- ...

### 与当前变更的关系
- ...

### 审查关注点
- ...""",
        )

        return call_llm(prompt, SYSTEM_PROMPT, max_tokens=800)

    def _search(self, diff_chunks: list[dict]) -> tuple[list[dict], str]:
        """基于 diff 中的改动搜索相关历史代码。"""
        all_chunks = []
        query_texts = []

        for chunk in diff_chunks[:3]:
            query = chunk["lines"][:500]
            query_texts.append(query)
            hits = search_codebase(query, top_k=3)
            all_chunks.extend(hits)

        return all_chunks, "\n".join(query_texts)

    def run(self, diff_chunks: list[dict]) -> str:
        """执行审查上下文检索并返回摘要。"""
        all_chunks, combined_query = self._search(diff_chunks)
        if not all_chunks:
            return "代码库中未找到相关历史上下文。"

        summary = self._llm_summarize(all_chunks, combined_query)
        if summary:
            logger.info("RAG 审查上下文摘要长度：%s", len(summary))
            return summary

        logger.warning("RAG 摘要生成失败，回退为原始上下文拼接。")
        contexts = [f"[{item['file_path']}::{item['func_name']}]\n{item['code']}" for item in all_chunks]
        return "\n\n---\n\n".join(contexts)


RAGAgent = StructuredRAGAgent

__all__ = ["RAGAgent", "ReviewRAGResult", "LegacyRAGAgent"]

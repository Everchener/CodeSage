import logging
import re

from codesage.tools.prompt_tools import build_json_output, build_prompt, build_system_prompt
from codesage.tools.llm_tools import call_llm_json

logger = logging.getLogger(__name__)

# 轻量正则规则，优先做低成本预筛查。
DANGEROUS_PATTERNS = [
    (r'(?i)(password|secret|api_key|token)\s*=\s*["\'][^"\']{4,}["\']', "硬编码凭证"),
    (r"\beval\s*\(", "危险函数 eval()"),
    (r"\bexec\s*\(", "危险函数 exec()"),
    (r"\bos\.system\s*\(", "危险函数 os.system()"),
    (r"\bsubprocess\.call\s*\(.*shell\s*=\s*True", "shell=True 注入风险"),
    (r"\bpickle\.loads?\s*\(", "不安全的 pickle 反序列化"),
]

SYSTEM_PROMPT = build_system_prompt(
    role="你是一名安全审查专家，负责识别新增代码中有证据支撑的安全风险。",
    responsibilities=[
        "关注正则规则不易覆盖的漏洞，例如 SQL 注入、路径遍历、SSRF、XSS、命令注入等。",
        "识别认证、授权、密钥处理、加密哈希等安全敏感逻辑中的明显缺陷。",
        "避免没有代码证据的泛化猜测，降低误报。",
    ],
    rules=[
        "只有当风险能从给出的新增代码中直接看出时才报告。",
        "如果代码片段不足以支持结论，必须返回空数组 []。",
        "description 必须写清触发点，severity 只能是 low、medium、high。",
    ],
    output_instruction="只返回 JSON 数组，不要输出 Markdown、解释或思考过程。",
)


class SecurityAgent:
    """负责安全审查的 Agent。"""

    def __init__(self):
        self.name = "SecurityAgent"
        self.patterns = DANGEROUS_PATTERNS

    def _llm_scan(self, code: str, file: str) -> list[dict]:
        """使用 LLM 检查复杂安全问题。"""
        prompt = build_prompt(
            task="分析新增代码中的安全漏洞，只报告有明确代码证据支撑的问题。",
            context_sections=[
                ("文件", file),
                ("新增代码", f"```python\n{code[:1500]}\n```"),
            ],
            rules=[
                "不要重复报告简单硬编码凭证或危险函数等正则规则已经容易捕获的问题，除非存在更高层的复合风险。",
                "不要因为代码里出现 request、sql、token 等关键词就机械判定漏洞；必须说明具体危险点。",
                "如果代码已经采用安全做法或证据不足，请返回 []。",
            ],
            examples=[
                (
                    """文件：db.py
新增代码：```python
sql = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(sql)
```""",
                    """[{"type": "SQL 注入", "description": "使用字符串拼接构造 SQL 并直接执行，用户输入可能进入查询语句。", "severity": "high"}]""",
                ),
                (
                    """文件：db.py
新增代码：```python
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```""",
                    "[]",
                ),
            ],
            output_format=build_json_output(
                """[
  {
    "type": "漏洞类型",
    "description": "基于代码证据的说明",
    "severity": "low|medium|high"
  }
]"""
            ),
        )

        result = call_llm_json(prompt, SYSTEM_PROMPT, max_tokens=500)
        if not result or not isinstance(result, list):
            return []

        issues = []
        for item in result:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "file": file,
                    "issue": f"[LLM] {item.get('type', '未知漏洞')}",
                    "snippet": item.get("description", "")[:100],
                    "severity": item.get("severity", "medium"),
                }
            )
        return issues

    def _regex_scan(self, code: str, file: str) -> list[dict]:
        """使用正则规则做快速预筛查。"""
        issues = []
        for pattern, desc in self.patterns:
            for match in re.finditer(pattern, code):
                issues.append(
                    {
                        "file": file,
                        "issue": desc,
                        "snippet": match.group(0)[:80],
                        "severity": "high",
                    }
                )
        return issues

    def run(self, diff_chunks: list[dict]) -> list[dict]:
        """执行安全审查。"""
        issues = []
        seen = set()

        for chunk in diff_chunks:
            added_lines = [line[1:] for line in chunk["lines"].splitlines() if line.startswith("+")]
            code = "\n".join(added_lines)
            file = chunk["file"]
            if not code.strip():
                continue

            for issue in self._regex_scan(code, file):
                key = (issue["file"], issue["issue"], issue["snippet"][:40])
                if key not in seen:
                    seen.add(key)
                    issues.append(issue)

            for issue in self._llm_scan(code, file):
                key = (issue["file"], issue["issue"], issue["snippet"][:40])
                if key not in seen:
                    seen.add(key)
                    issues.append(issue)

        return issues

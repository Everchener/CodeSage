import json
import logging
import os
import re
from typing import Any, Optional

from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - 依赖于运行环境
    ChatOpenAI = None

from codesage.core.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)

logger = logging.getLogger(__name__)
JSON_LLM_TIMEOUT_SECONDS = float(os.getenv("JSON_LLM_TIMEOUT_SECONDS", "20"))
TEXT_LLM_TIMEOUT_SECONDS = float(os.getenv("TEXT_LLM_TIMEOUT_SECONDS", "30"))


def is_langchain_openai_available() -> bool:
    return ChatOpenAI is not None


def _build_llm(
    max_tokens: int = 500,
    temperature: float = 0,
    timeout: float = TEXT_LLM_TIMEOUT_SECONDS,
):
    """构建供工具调用链路使用的 LangChain 客户端。"""
    if ChatOpenAI is None:
        raise RuntimeError(
            "langchain_openai is required for LangChain tool calling. "
            "Install it with `pip install langchain-openai`."
        )

    kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": 0,
    }
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return ChatOpenAI(**kwargs)


def _build_openai_client(timeout: float = TEXT_LLM_TIMEOUT_SECONDS) -> OpenAI:
    kwargs: dict[str, Any] = {
        "api_key": LLM_API_KEY,
        "timeout": timeout,
        "max_retries": 0,
    }
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "content"):
        content = value.content
        if isinstance(content, str):
            return content
        return str(content)
    if isinstance(value, str):
        return value
    return str(value)


def _remove_thinking_tags(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"<think>.*?</think>",
        r"<\|thinking\|>.*?<\|end\|>",
        r"<\|thought\|>.*?<\|end\|>",
        r"<\|ankton\w*\|>.*?<\|/ankton\|>",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()


def sanitize_llm_output(text: str) -> str:
    return _remove_thinking_tags(text)


def _create_thinking_cleaner() -> RunnableLambda:
    def clean(value: Any) -> str:
        return _remove_thinking_tags(_extract_text(value))

    return RunnableLambda(clean)


def create_llm_chain(
    max_tokens: int = 500,
    temperature: float = 0,
    output_parser: Optional[Any] = None,
    timeout: float = TEXT_LLM_TIMEOUT_SECONDS,
):
    llm = _build_llm(max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    cleaner = _create_thinking_cleaner()

    if output_parser is not None:
        return llm | cleaner | output_parser

    return llm | cleaner | StrOutputParser()


def _build_messages(prompt: str, system: str = "") -> list:
    messages = []
    if system.strip():
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    return messages


def _build_openai_messages(prompt: str, system: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _fallback_completion(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    response = _build_openai_client(timeout=timeout).chat.completions.create(
        model=LLM_MODEL,
        messages=_build_openai_messages(prompt, system),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return sanitize_llm_output(response.choices[0].message.content or "")


def _extract_json_payload(text: str) -> Optional[dict | list]:
    cleaned = sanitize_llm_output(text)
    if not cleaned:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL):
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 500,
    temperature: float = 0,
    timeout: float = TEXT_LLM_TIMEOUT_SECONDS,
) -> str:
    try:
        if ChatOpenAI is not None:
            chain = create_llm_chain(
                max_tokens=max_tokens,
                temperature=temperature,
                output_parser=StrOutputParser(),
                timeout=timeout,
            )
            response = chain.invoke(_build_messages(prompt=prompt, system=system))
            return response if isinstance(response, str) else str(response)

        return _fallback_completion(prompt, system, max_tokens, temperature, timeout)
    except Exception as exc:
        logger.exception("LLM call failed")
        logger.warning("LLM text request failed: %s", exc)
        return ""


def call_llm_json(
    prompt: str,
    system: str = "",
    max_tokens: int = 500,
    temperature: float = 0,
    timeout: float = JSON_LLM_TIMEOUT_SECONDS,
) -> Optional[dict | list]:
    json_system = (
        "你是只输出 JSON 的助手。\n"
        "只返回合法 JSON。\n"
        "不要包含解释、Markdown 代码块或额外文本。"
    )
    if system.strip():
        json_system = system.strip() + "\n\n" + json_system

    try:
        if ChatOpenAI is not None:
            chain = create_llm_chain(
                max_tokens=max_tokens,
                temperature=temperature,
                output_parser=StrOutputParser(),
                timeout=timeout,
            )
            response = chain.invoke(_build_messages(prompt=prompt, system=json_system))
            return _extract_json_payload(_extract_text(response))

        return _extract_json_payload(
            _fallback_completion(prompt, json_system, max_tokens, temperature, timeout)
        )
    except Exception as exc:
        logger.exception("LLM JSON call failed")
        logger.warning("LLM JSON request failed: %s", exc)
        return None

"""
Document Processor Module

支持上传并处理多种类型的文档，使用 Milvus 进行向量存储。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .processor import DocumentProcessor

__all__ = ["DocumentProcessor"]


def __getattr__(name: str) -> Any:
    if name != "DocumentProcessor":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("codesage.document_processor.processor"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

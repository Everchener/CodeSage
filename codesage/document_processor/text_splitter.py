"""
文本分割模块 - 使用 RecursiveCharacterTextSplitter 进行文本分块

保留中英文分隔符支持。
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """创建文本分割器

    Args:
        chunk_size: 块大小（字符数）
        chunk_overlap: 块重叠大小
        separators: 分隔符列表，默认包含中英文分隔符

    Returns:
        RecursiveCharacterTextSplitter 实例
    """
    if separators is None:
        # 默认中英文分隔符
        separators = [
            "\n\n",  # 段落分隔
            "\n",    # 换行
            "。",    # 中文句子
            "，",    # 中文逗号
            "；",    # 中文分号
            "：",    # 中文冒号
            " ",     # 英文空格
            ""       # 字符级别
        ]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """分割文本为块

    Args:
        text: 待分割文本
        chunk_size: 块大小
        chunk_overlap: 块重叠

    Returns:
        文本块列表
    """
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def split_documents(documents: List[str], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """分割多个文档

    Args:
        documents: 文档列表
        chunk_size: 块大小
        chunk_overlap: 块重叠

    Returns:
        所有文本块列表
    """
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_texts(documents)

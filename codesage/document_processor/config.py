"""
文档处理模块的配置。

复用主项目配置。
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Milvus 配置，复用主项目设置。
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("DOCS_COLLECTION_NAME", "codesage_documents")

# 文本切分配置。
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 40

# 支持的文件扩展名。
SUPPORTED_EXTENSIONS = [
    ".pdf",   # PDF 文档
    ".txt",   # 纯文本
    ".md",    # Markdown 文档
    ".docx",  # Word 文档
    ".xlsx",  # Excel xlsx 表格
    ".xls",   # Excel xls 表格
    ".pptx",  # PowerPoint 演示文稿
    ".html",  # HTML 文档
    ".csv",   # CSV 文件
]

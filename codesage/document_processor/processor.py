"""
Document processor main workflow.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .chunks import build_document_chunk_records
from .document_loader import extract_text, is_supported, resolve_loader_name
from .vector_store import MilvusVectorStore
from .config import (
    COLLECTION_NAME,
    MILVUS_HOST,
    MILVUS_PORT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Coordinates document extraction, chunking and storage."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        vector_store: Optional[MilvusVectorStore] = None,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = vector_store or MilvusVectorStore(
            collection_name=collection_name,
            host=MILVUS_HOST,
            port=MILVUS_PORT,
        )

    def _build_result(
        self,
        *,
        status: str,
        message: str,
        records: list[dict[str, Any]] | None = None,
        vector_ids: list[int] | None = None,
        filename: str = "",
        documents: int = 0,
    ) -> Dict[str, Any]:
        normalized_records = records or []
        doc_types = sorted(
            {
                str(record.get("doc_type", "")).strip()
                for record in normalized_records
                if str(record.get("doc_type", "")).strip()
            }
        )
        titles_detected = sorted(
            {
                str(record.get("title", "")).strip()
                for record in normalized_records
                if str(record.get("title", "")).strip()
            }
        )
        payload: Dict[str, Any] = {
            "status": status,
            "message": message,
            "documents": documents,
            "chunks": len(normalized_records),
            "vector_ids": vector_ids or [],
            "doc_types": doc_types,
            "titles_detected": titles_detected,
        }
        if filename:
            payload["filename"] = filename
        return payload

    def process_file(
        self,
        filepath: str,
        verbose: bool = True,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            return self._build_result(
                status="error",
                message=f"文件不存在: {filepath}",
            )

        filename = source_name or os.path.basename(filepath)
        if not is_supported(filepath):
            return self._build_result(
                status="error",
                message=f"不支持的文件格式: {filename}",
                filename=filename,
            )

        try:
            if verbose:
                logger.info("正在提取文本: %s", filename)
                logger.info("正在使用文档提取器：%s", resolve_loader_name(filepath))
            text = extract_text(filepath)
            if not text or not text.strip():
                return self._build_result(
                    status="error",
                    message=f"无法提取文本或文档为空: {filename}",
                    filename=filename,
                )

            if verbose:
                logger.info("正在结构化文档: %s", filename)
            records = build_document_chunk_records(
                text,
                source=filename,
                source_path=filepath,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if not records:
                return self._build_result(
                    status="error",
                    message=f"文本切块失败: {filename}",
                    filename=filename,
                )

            if verbose:
                logger.info("正在存储向量: %s", filename)
            ids = self.vector_store.add_texts(records)
            serializable_ids = [int(vector_id) for vector_id in ids]
            if verbose:
                logger.info("完成处理 %s: %s 个文本块", filename, len(records))
            return self._build_result(
                status="success",
                message=f"成功处理 {filename}",
                records=records,
                vector_ids=serializable_ids,
                filename=filename,
                documents=1,
            )
        except Exception as exc:
            error_msg = str(exc)
            logger.error("处理文件失败 %s: %s", filename, error_msg)
            return self._build_result(
                status="error",
                message=f"处理失败: {error_msg}",
                filename=filename,
            )

    def process_files(
        self,
        filepaths: List[str],
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        results = []
        if verbose:
            logger.info("开始处理 %s 个文件", len(filepaths))

        for idx, filepath in enumerate(filepaths, 1):
            if verbose:
                logger.info("处理进度: %s/%s", idx, len(filepaths))
            results.append(self.process_file(filepath, verbose=verbose))

        success_count = sum(1 for result in results if result["status"] == "success")
        total_chunks = sum(int(result.get("chunks", 0) or 0) for result in results)
        if verbose:
            logger.info("处理完成: 成功 %s/%s, 共 %s 个文本块", success_count, len(filepaths), total_chunks)
        return results

    def process_documents(
        self,
        documents: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        all_records: list[dict[str, Any]] = []
        document_count = 0

        for doc in documents:
            text = ""
            source = str(doc.get("source", "unknown") or "unknown")
            source_path = str(doc.get("source_path", "") or "")
            if "filepath" in doc:
                filepath = str(doc["filepath"])
                text = extract_text(filepath)
                source = os.path.basename(filepath) if filepath else source
                source_path = filepath or source_path
            elif "text" in doc:
                text = str(doc["text"])
            else:
                continue

            if not text or not text.strip():
                continue

            records = build_document_chunk_records(
                text,
                source=source,
                source_path=source_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if not records:
                continue
            all_records.extend(records)
            document_count += 1

        if not all_records:
            return self._build_result(
                status="error",
                message="没有有效的文档内容",
            )

        ids = self.vector_store.add_texts(all_records)
        serializable_ids = [int(vector_id) for vector_id in ids]
        return self._build_result(
            status="success",
            message=f"成功处理 {document_count} 个文档",
            records=all_records,
            vector_ids=serializable_ids,
            documents=document_count,
        )

    def search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, top_k, filters=filters)

    def clear(self):
        self.vector_store.delete_all()

    def count(self) -> int:
        return self.vector_store.count()

    def close(self):
        self.vector_store.close()


def create_processor(
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> DocumentProcessor:
    return DocumentProcessor(
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

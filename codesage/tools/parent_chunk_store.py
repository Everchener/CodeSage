"""父级分块文档存储（用于自动合并检索器）。"""

import json
from pathlib import Path
from typing import Dict, List

from codesage.tools.file_io import write_json_file


class ParentChunkStore:
    """基于本地 JSON 的父级分块存储。"""

    def __init__(self, store_path: Path | None = None):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.store_path = store_path or (base_dir / "data" / "parent_chunks_apidocs.json")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, dict]:
        if not self.store_path.exists():
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        write_json_file(self.store_path, data)

    def upsert_documents(self, docs: List[dict]) -> int:
        """写入或更新父级分块，返回写入条数。"""
        if not docs:
            return 0
        store = self._load()
        upserted = 0
        for doc in docs:
            chunk_id = (doc.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            store[chunk_id] = {
                "text": doc.get("text", ""),
                "filename": doc.get("filename", ""),
                "file_type": doc.get("file_type", ""),
                "file_path": doc.get("file_path", ""),
                "page_number": doc.get("page_number", 0),
                "chunk_id": chunk_id,
                "parent_chunk_id": doc.get("parent_chunk_id", ""),
                "root_chunk_id": doc.get("root_chunk_id", ""),
                "chunk_level": int(doc.get("chunk_level", 0) or 0),
                "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
            }
            upserted += 1
        self._save(store)
        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []
        store = self._load()
        return [store[item] for item in chunk_ids if item in store]

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除父级分块，返回删除条数。"""
        if not filename:
            return 0
        store = self._load()
        before = len(store)
        filtered = {k: v for k, v in store.items() if v.get("filename") != filename}
        deleted = before - len(filtered)
        if deleted > 0:
            self._save(filtered)
        return deleted

    def delete_by_source(self, source: str) -> int:
        """按 `source`（文件路径）删除父级分块，返回删除条数。"""
        if not source:
            return 0
        store = self._load()
        before = len(store)
        # 注意：存储中使用 `file_path` 字段保存 `source` 信息。
        filtered = {k: v for k, v in store.items() if v.get("file_path") != source}
        deleted = before - len(filtered)
        if deleted > 0:
            self._save(filtered)
        return deleted

    def list_sources(self) -> List[str]:
        """列出所有唯一的 `source`（文件路径）。"""
        store = self._load()
        sources = set(v.get("file_path", "") for v in store.values())
        return sorted(s for s in sources if s)


# 全局单例。
_parent_chunk_store: ParentChunkStore | None = None


def get_parent_chunk_store() -> ParentChunkStore:
    global _parent_chunk_store
    if _parent_chunk_store is None:
        _parent_chunk_store = ParentChunkStore()
    return _parent_chunk_store

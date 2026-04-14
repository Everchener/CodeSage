"""Milvus 数据管理工具 - 提供 collection 审查和选择性删除功能"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymilvus import MilvusClient, connections, utility

from codesage.core.config import MILVUS_HOST, MILVUS_PORT


class MilvusManager:
    """Milvus 数据管理器 - 支持审查、选择删除和数据导出"""

    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        self.host = host
        self.port = port
        self.uri = f"http://{host}:{port}"
        self._client: Optional[MilvusClient] = None

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self.uri)
        return self._client

    def _ensure_connection(self):
        """确保旧版连接可用（用于某些操作）"""
        if not connections.has_connection("default"):
            connections.connect(host=self.host, port=str(self.port))

    # ========== 审查方法 ==========

    def list_collections(self) -> List[str]:
        """列出所有 collections"""
        return self.client.list_collections()

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取 collection 详细信息"""
        if not self.client.has_collection(collection_name):
            return {"exists": False}

        self._ensure_connection()
        col = Collection(collection_name)

        return {
            "exists": True,
            "name": collection_name,
            "row_count": col.num_entities,
            "description": col.description or "",
        }

    def count_entities(
        self, collection_name: str, filter_expr: Optional[str] = None
    ) -> int:
        """统计 collection 中的实体数量"""
        self._ensure_connection()
        col = Collection(collection_name)
        col.load()

        if filter_expr:
            results = col.query(expr=filter_expr, output_fields=["id"])
            return len(results)
        else:
            stats = col.get_stats()
            return stats.get("row_count", 0)

    def list_sources(self, collection_name: str) -> List[str]:
        """列出 collection 中所有唯一的 source"""
        self._ensure_connection()
        col = Collection(collection_name)
        col.load()

        try:
            results = col.query(
                expr="source != ''",
                output_fields=["source"],
                limit=10000,
            )
            sources = set(r.get("source", "") for r in results if r.get("source"))
            return sorted(s for s in sources if s)
        except Exception:
            # 某些 collection 可能没有 source 字段
            return []

    def list_doc_types(self, collection_name: str) -> List[str]:
        """列出 collection 中所有唯一的 doc_type"""
        self._ensure_connection()
        col = Collection(collection_name)
        col.load()

        try:
            results = col.query(
                expr="doc_type != ''",
                output_fields=["doc_type"],
                limit=10000,
            )
            doc_types = set(r.get("doc_type", "") for r in results if r.get("doc_type"))
            return sorted(d for d in doc_types if d)
        except Exception:
            return []

    def sample_data(
        self,
        collection_name: str,
        limit: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """采样查看 collection 中的数据"""
        self._ensure_connection()
        col = Collection(collection_name)
        col.load()

        try:
            # 获取所有字段
            fields = self.get_schema_fields(collection_name)
            results = col.query(
                expr=filter_expr if filter_expr else "id >= 0",
                output_fields=fields,
                limit=limit,
            )
            return results
        except Exception as e:
            return [{"error": str(e)}]

    def get_schema_fields(self, collection_name: str) -> List[str]:
        """获取 collection 的 schema 字段"""
        if not self.client.has_collection(collection_name):
            return []
        self._ensure_connection()
        schema = Collection(collection_name).schema
        return [field.name for field in schema.fields]

    # ========== 删除方法 ==========

    def _get_preview(
        self, collection_name: str, filter_expr: str, limit: int = 5
    ) -> List[Dict]:
        """获取删除预览数据"""
        self._ensure_connection()
        col = Collection(collection_name)
        col.load()

        try:
            return col.query(expr=filter_expr, limit=limit)
        except Exception:
            return []

    def _get_unique_sources_to_delete(
        self, collection_name: str, filter_expr: str
    ) -> List[str]:
        """获取要删除的数据中涉及的所有 source"""
        self._ensure_connection()
        col = Collection(collection_name)
        col.load()

        try:
            results = col.query(expr=filter_expr, output_fields=["source"], limit=10000)
            sources = set(r.get("source", "") for r in results if r.get("source"))
            return sorted(s for s in sources if s)
        except Exception:
            return []

    def delete_by_expression(
        self,
        collection_name: str,
        expr: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """按表达式删除数据

        Args:
            collection_name: collection 名称
            expr: Milvus 表达式，如 "source == 'httpx'"
            dry_run: 是否预览（不执行删除）

        Returns:
            包含操作结果的字典
        """
        if not self.client.has_collection(collection_name):
            return {"success": False, "error": f"Collection '{collection_name}' 不存在"}

        # 统计将删除的数量
        count = self.count_entities(collection_name, expr)

        # 获取预览数据
        preview = self._get_preview(collection_name, expr, limit=5)

        # 获取涉及的 sources（用于父级分块清理）
        sources = self._get_unique_sources_to_delete(collection_name, expr)

        result = {
            "success": True,
            "dry_run": dry_run,
            "collection": collection_name,
            "filter_expr": expr,
            "will_delete_count": count,
            "preview": preview[:5],
            "sources": sources,
        }

        if dry_run:
            result["message"] = f"[DRY-RUN] 预览: 将删除 {count} 条数据"
            return result

        # 执行删除
        self._ensure_connection()
        col = Collection(collection_name)
        col.load()
        delete_result = col.delete(expr=expr)
        col.flush()

        result["deleted_count"] = count
        result["message"] = f"已删除 {count} 条数据"

        # 尝试清理父级分块（仅对 v2 collection 生效）
        parent_chunks_deleted = self._cleanup_parent_chunks(sources)
        if parent_chunks_deleted > 0:
            result["parent_chunks_deleted"] = parent_chunks_deleted
            result["message"] += f"，清理了 {parent_chunks_deleted} 个父级分块"

        return result

    def delete_by_source(
        self,
        collection_name: str,
        source: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """按 source 删除数据"""
        expr = f"source == '{source}'"
        return self.delete_by_expression(collection_name, expr, dry_run)

    def delete_by_doc_type(
        self,
        collection_name: str,
        doc_type: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """按 doc_type 删除数据"""
        expr = f"doc_type == '{doc_type}'"
        return self.delete_by_expression(collection_name, expr, dry_run)

    def delete_all(
        self, collection_name: str, confirm: bool = False
    ) -> Dict[str, Any]:
        """删除整个 collection

        Args:
            collection_name: collection 名称
            confirm: 必须为 True 才能执行删除

        Returns:
            包含操作结果的字典
        """
        if not confirm:
            return {
                "success": False,
                "error": "需要 --confirm 参数才能执行删除整个 collection",
            }

        if not self.client.has_collection(collection_name):
            return {"success": False, "error": f"Collection '{collection_name}' 不存在"}

        # 获取 collection 信息
        info = self.get_collection_info(collection_name)
        count = info.get("row_count", 0)

        # 删除 collection
        self.client.drop_collection(collection_name)

        return {
            "success": True,
            "collection": collection_name,
            "deleted_count": count,
            "message": f"已删除 collection '{collection_name}'（共 {count} 条数据）",
        }

    # ========== 工具方法 ==========

    def export_data(
        self,
        collection_name: str,
        output_path: str,
        filter_expr: Optional[str] = None,
        limit: int = 10000,
    ) -> Dict[str, Any]:
        """导出数据到 JSON 文件

        Args:
            collection_name: collection 名称
            output_path: 输出文件路径
            filter_expr: 可选的过滤表达式
            limit: 最大导出条数

        Returns:
            包含操作结果的字典
        """
        if not self.client.has_collection(collection_name):
            return {"success": False, "error": f"Collection '{collection_name}' 不存在"}

        self._ensure_connection()
        col = Collection(collection_name)
        col.load()

        expr = filter_expr if filter_expr else "id >= 0"
        results = col.query(expr=expr, limit=limit)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "collection": collection_name,
            "exported_count": len(results),
            "output_path": str(output_file),
            "message": f"已导出 {len(results)} 条数据到 {output_path}",
        }

    def _cleanup_parent_chunks(self, sources: List[str]) -> int:
        """清理父级分块存储中对应的 source"""
        if not sources:
            return 0

        try:
            from codesage.tools.parent_chunk_store import get_parent_chunk_store

            store = get_parent_chunk_store()
            total_deleted = 0

            for source in sources:
                deleted = store.delete_by_source(source)
                total_deleted += deleted

            return total_deleted
        except Exception:
            return 0


# 兼容旧代码
def connect_milvus():
    """连接 Milvus（兼容旧代码）"""
    if not connections.has_connection("default"):
        connections.connect(host=MILVUS_HOST, port=str(MILVUS_PORT))


# 为 import Collection 提供兼容
from pymilvus import Collection  # noqa: E402

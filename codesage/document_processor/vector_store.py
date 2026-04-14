"""Milvus-backed document vector storage."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from ..core.config import MILVUS_HOST, MILVUS_PORT, get_embedding, get_embedding_dim, get_query_embedding
from ..tools.bm25_embedding import append_bm25_corpus, delete_bm25_artifacts, get_bm25_service, load_bm25_corpus
from ..indexing.vector_utils import ensure_collection_vector_dim
from .chunks import DocumentChunkRecord, build_document_filter_expr, deserialize_tags, serialize_tags
from .config import COLLECTION_NAME

logger = logging.getLogger(__name__)

DOCUMENT_OUTPUT_FIELDS = [
    "text",
    "source",
    "chunk_index",
    "doc_type",
    "title",
    "section_path",
    "source_path",
    "file_ext",
    "page_or_slide",
    "tags",
]
DOCUMENT_REQUIRED_FIELDS = set(DOCUMENT_OUTPUT_FIELDS) | {"embedding", "id"}
SCHEMA_UPGRADE_MESSAGE = (
    "Document collection schema is outdated. Clear and rebuild the documents index before indexing or searching again."
)


class MilvusVectorStore:
    """Stores structured document chunks in Milvus."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        dim: int | None = None,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dim = int(dim if dim is not None else get_embedding_dim())
        self._collection = None
        self._connect()

    def _connect(self):
        try:
            aliases = connections.list_connections()
            if self.host not in aliases:
                connections.connect(host=self.host, port=self.port)
            logger.info("已连接到 Milvus: %s:%s", self.host, self.port)
        except Exception as exc:
            logger.error("Milvus 连接失败: %s", exc)
            raise

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            ensure_collection_vector_dim(collection, self.dim, collection_name=self.collection_name)
            _ensure_document_collection_schema(collection)
            collection.load()
            return collection

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="file_ext", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="page_or_slide", dtype=DataType.INT64),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description="Structured document chunks with embeddings")
        collection = Collection(self.collection_name, schema)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        collection.create_index("embedding", index_params)
        collection.load()
        logger.info("创建 Collection: %s", self.collection_name)
        return collection

    def add_texts(self, records: list[DocumentChunkRecord]) -> list[int]:
        if not records:
            return []

        embeddings = get_embedding([str(record.get("search_text", "") or record.get("text", "")) for record in records])
        data = [
            [str(record.get("text", ""))[:65535] for record in records],
            [str(record.get("source", ""))[:512] for record in records],
            [int(record.get("chunk_index", 0) or 0) for record in records],
            [str(record.get("doc_type", ""))[:64] for record in records],
            [str(record.get("title", ""))[:512] for record in records],
            [str(record.get("section_path", ""))[:2048] for record in records],
            [str(record.get("source_path", ""))[:2048] for record in records],
            [str(record.get("file_ext", ""))[:16] for record in records],
            [int(record.get("page_or_slide", 0) or 0) for record in records],
            [serialize_tags(list(record.get("tags", [])))[:1024] for record in records],
            embeddings,
        ]

        collection = self._get_or_create_collection()
        result = collection.insert(data)
        collection.flush()

        corpus_entries = []
        for record in records:
            corpus_entries.append(
                {
                    "text": str(record.get("text", "")),
                    "search_text": str(record.get("search_text", "") or record.get("text", "")),
                    "source": str(record.get("source", "")),
                    "chunk_index": int(record.get("chunk_index", 0) or 0),
                    "doc_type": str(record.get("doc_type", "")),
                    "title": str(record.get("title", "")),
                    "section_path": str(record.get("section_path", "")),
                    "source_path": str(record.get("source_path", "")),
                    "file_ext": str(record.get("file_ext", "")),
                    "page_or_slide": int(record.get("page_or_slide", 0) or 0),
                    "tags": list(record.get("tags", [])),
                }
            )
        append_bm25_corpus(self.collection_name, corpus_entries)
        corpus = load_bm25_corpus(self.collection_name)
        service = get_bm25_service(self.collection_name)
        service.fit_corpus([str(item.get("search_text", item.get("text", ""))) for item in corpus])

        logger.info("插入 %s 个文档块到 Milvus", len(records))
        return result.primary_keys

    def search(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[Dict[str, Any]]:
        query_embedding = get_query_embedding([query])[0]
        collection = self._get_or_create_collection()
        search_kwargs: dict[str, Any] = {
            "data": [query_embedding],
            "anns_field": "embedding",
            "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
            "limit": top_k,
            "output_fields": DOCUMENT_OUTPUT_FIELDS,
        }
        expr = build_document_filter_expr(filters)
        if expr is not None:
            search_kwargs["expr"] = expr
        results = collection.search(**search_kwargs)

        output = []
        for hits in results:
            for hit in hits:
                output.append(
                    {
                        "id": hit.id,
                        "text": hit.entity.get("text"),
                        "source": hit.entity.get("source"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "doc_type": hit.entity.get("doc_type"),
                        "title": hit.entity.get("title"),
                        "section_path": hit.entity.get("section_path"),
                        "source_path": hit.entity.get("source_path"),
                        "file_ext": hit.entity.get("file_ext"),
                        "page_or_slide": hit.entity.get("page_or_slide"),
                        "tags": deserialize_tags(hit.entity.get("tags")),
                        "distance": hit.distance,
                    }
                )
        return output

    def delete_all(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info("已删除 Collection: %s", self.collection_name)
            self._collection = None
        delete_bm25_artifacts(self.collection_name)

    def count(self) -> int:
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            return collection.num_entities
        return 0

    def close(self):
        try:
            connections.disconnect(self.host)
        except Exception:
            pass


_vector_store: Optional[MilvusVectorStore] = None


def get_vector_store() -> MilvusVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = MilvusVectorStore()
    return _vector_store


def _ensure_document_collection_schema(collection: Collection) -> None:
    field_names = {field.name for field in getattr(collection.schema, "fields", [])}
    missing = DOCUMENT_REQUIRED_FIELDS - field_names
    if missing:
        raise RuntimeError(SCHEMA_UPGRADE_MESSAGE)

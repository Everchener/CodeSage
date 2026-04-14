import time
import re
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from codesage.core.config import (
    COLLECTION_NAME,
    MEMORY_COLLECTION_NAME,
    MILVUS_HOST,
    MILVUS_PORT,
    get_embedding,
    get_embedding_dim,
    get_query_embedding,
)
from codesage.document_processor.chunks import (
    build_document_filter_expr,
    deserialize_tags,
    matches_document_filters,
)
from codesage.document_processor.config import COLLECTION_NAME as DOCS_COLLECTION_NAME
from codesage.document_processor.vector_store import DOCUMENT_OUTPUT_FIELDS
from codesage.tools.bm25_embedding import get_bm25_service, load_bm25_corpus
from codesage.indexing.vector_utils import ensure_collection_vector_dim


APIDOCS_COLLECTION_NAME = "codesage_apidocs"
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IDENTIFIER_STOPWORDS = {"def", "class", "return", "where", "what", "does", "how", "why", "the", "and", "or", "in", "on", "at", "it"}
BM25_SCORE_WEIGHT = 0.75
RRF_K = 60.0
MULTI_SOURCE_TOP_K = 8
FUSION_CANDIDATE_LIMIT = 12
FUSION_MIN_PER_SOURCE = 2


def connect_milvus() -> bool:
    if not connections.has_connection("default"):
        connections.connect(host=MILVUS_HOST, port=str(MILVUS_PORT))
    return True


def _ensure_connection():
    if not connections.has_connection("default"):
        connections.connect(host=MILVUS_HOST, port=str(MILVUS_PORT))


def _search_collection(
    collection_name: str,
    vector: list[float],
    top_k: int,
    metric_type: str,
    search_params: dict,
    output_fields: list[str],
    expr: str | None = None,
) -> list[dict]:
    try:
        _ensure_connection()
        if not utility.has_collection(collection_name):
            return []

        collection = Collection(collection_name)
        ensure_collection_vector_dim(collection, len(vector), collection_name=collection_name)
        collection.load()
        search_kwargs = {
            "data": [vector],
            "anns_field": "embedding",
            "param": {"metric_type": metric_type, "params": search_params},
            "limit": top_k,
            "output_fields": output_fields,
        }
        if expr is not None:
            search_kwargs["expr"] = expr
        results = collection.search(**search_kwargs)
    except Exception:
        return []

    hits = []
    for hit in results[0]:
        payload = {field: hit.entity.get(field) for field in output_fields}
        payload["id"] = int(hit.id) if getattr(hit, "id", None) is not None else None
        payload["score"] = getattr(hit, "score", None)
        hits.append(payload)
    return hits


def _query_exact_code_symbols(identifiers: set[str]) -> list[dict]:
    filtered = [
        identifier
        for identifier in identifiers
        if identifier and identifier not in IDENTIFIER_STOPWORDS
    ]
    if not filtered:
        return []

    try:
        _ensure_connection()
        if not utility.has_collection(COLLECTION_NAME):
            return []

        collection = Collection(COLLECTION_NAME)
        collection.load()
        expr_values = ", ".join(f'"{_escape_expr_value(identifier)}"' for identifier in filtered)
        rows = collection.query(
            expr=f"func_name in [{expr_values}]",
            output_fields=["file_path", "func_name", "code"],
            limit=max(20, len(filtered) * 3),
        )
    except Exception:
        return []

    hits = []
    for row in rows:
        payload = dict(row)
        payload["score"] = 1.0
        hits.append(payload)
    return hits


def _match_exact_code_symbols_from_corpus(identifiers: set[str]) -> list[dict]:
    filtered = {
        identifier
        for identifier in identifiers
        if identifier and identifier not in IDENTIFIER_STOPWORDS
    }
    if not filtered:
        return []

    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in load_bm25_corpus(COLLECTION_NAME):
        func_name = str(entry.get("func_name", "") or "").strip()
        normalized_func_name = func_name.lower()
        if not func_name or normalized_func_name not in filtered:
            continue
        key = (
            str(entry.get("file_path", "") or "").strip(),
            func_name,
        )
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "file_path": key[0],
                "func_name": func_name,
                "code": str(entry.get("code", "") or ""),
                "score": 1.0,
            }
        )
    return hits


def _search_bm25_corpus(
    namespace: str,
    query: str,
    *,
    top_k: int,
    text_getter,
) -> list[dict]:
    entries = load_bm25_corpus(namespace)
    if not entries:
        return []

    texts = [str(text_getter(entry) or "") for entry in entries]
    if not any(texts):
        return []

    service = get_bm25_service(namespace)
    if getattr(service, "_total_docs", 0) != len(texts):
        service.fit_corpus(texts)

    scores = service.score_texts(query, texts)
    ranked = [
        (dict(entry), float(score))
        for entry, score in zip(entries, scores)
        if float(score) > 0.0
    ]
    if not ranked:
        return []

    ranked.sort(key=lambda item: item[1], reverse=True)
    max_score = ranked[0][1] or 1.0
    hits = []
    for entry, score in ranked[: max(1, top_k)]:
        entry["score"] = float(entry.get("score") or 0.0)
        entry["bm25_score"] = score / max_score
        hits.append(entry)
    return hits


def _merge_hits(hits: list[dict], extra_hits: list[dict], *, key_fields: tuple[str, ...]) -> list[dict]:
    merged: dict[tuple[str, ...], dict] = {}

    def key_for(hit: dict) -> tuple[str, ...]:
        return tuple(str(hit.get(field, "")) for field in key_fields)

    for hit in hits:
        payload = dict(hit)
        payload["score"] = float(payload.get("score") or 0.0)
        payload["bm25_score"] = float(payload.get("bm25_score") or 0.0)
        merged[key_for(payload)] = payload

    for hit in extra_hits:
        payload = dict(hit)
        payload["score"] = float(payload.get("score") or 0.0)
        payload["bm25_score"] = float(payload.get("bm25_score") or 0.0)
        key = key_for(payload)
        existing = merged.get(key)
        if existing is None:
            merged[key] = payload
            continue
        existing["score"] = max(float(existing.get("score") or 0.0), payload["score"])
        existing["bm25_score"] = max(float(existing.get("bm25_score") or 0.0), payload["bm25_score"])

    return list(merged.values())


def _knowledge_hit_key(hit: dict) -> tuple[Any, ...]:
    source_type = str(hit.get("source_type", "") or "").strip().lower()

    if source_type == "codebase" or "func_name" in hit or "file_path" in hit:
        return (
            "codebase",
            str(hit.get("file_path", "")).strip(),
            str(hit.get("func_name", "")).strip(),
        )

    if source_type == "documents" or "chunk_index" in hit or "chunk_idx" in hit:
        return (
            "documents",
            str(hit.get("source", "")).strip(),
            int(hit.get("chunk_index", hit.get("chunk_idx", -1)) or -1),
        )

    if source_type == "apidocs" or "doc_id" in hit or "title" in hit:
        return (
            "apidocs",
            str(hit.get("doc_id", "")).strip(),
            str(hit.get("source", "")).strip(),
            str(hit.get("title", "")).strip(),
        )

    return (
        str(hit.get("source_type", "")).strip(),
        str(hit.get("source_label", "")).strip(),
        str(hit.get("file_path", "")).strip(),
        str(hit.get("func_name", "")).strip(),
        str(hit.get("source", "")).strip(),
        str(hit.get("title", "")).strip(),
    )


def _build_source_label(hit: dict, source_type: str) -> str:
    if source_type == "codebase":
        file_path = str(hit.get("file_path", "")).strip()
        func_name = str(hit.get("func_name", "")).strip()
        return f"{file_path}::{func_name}" if file_path and func_name else (file_path or func_name or "unknown")
    if source_type == "documents":
        source = str(hit.get("source", "")).strip()
        chunk_index = hit.get("chunk_index", hit.get("chunk_idx", ""))
        return f"{source}#{chunk_index}" if source and chunk_index not in ("", None) else (source or "unknown")
    if source_type == "apidocs":
        title = str(hit.get("title", "")).strip()
        source = str(hit.get("source", "")).strip()
        return f"{source}::{title}" if source and title else (title or source or "unknown")
    return "unknown"


def _annotate_source_metadata(hit: dict, source_type: str, source_rank: int) -> dict:
    payload = dict(hit)
    raw_score = float(payload.get("score") or 0.0)
    bm25_score = float(payload.get("bm25_score") or 0.0)
    payload["source_type"] = source_type
    payload["source_label"] = _build_source_label(payload, source_type)
    payload["source_rank"] = int(source_rank)
    payload["raw_score"] = raw_score
    payload["bm25_score"] = bm25_score
    payload.setdefault("rrf_score", 0.0)
    payload.setdefault("fusion_score", raw_score + (bm25_score * BM25_SCORE_WEIGHT))
    payload.setdefault("is_fusion_backup", False)
    return payload


def _annotate_knowledge_source_hits(source_results: dict[str, list[dict]]) -> dict[str, list[dict]]:
    annotated: dict[str, list[dict]] = {}
    for source_type, hits in source_results.items():
        annotated[source_type] = [
            _annotate_source_metadata(hit, source_type, rank)
            for rank, hit in enumerate(hits, start=1)
        ]
    return annotated


def _search_general_knowledge_sources(query: str, per_source_top_k: int = MULTI_SOURCE_TOP_K) -> dict[str, list[dict]]:
    source_results = {
        "documents": search_documents(query, top_k=max(1, per_source_top_k)),
        "apidocs": hybrid_search_apidocs(query, top_k=max(1, per_source_top_k)),
    }
    return _annotate_knowledge_source_hits(source_results)


def _search_all_knowledge_sources(query: str, per_source_top_k: int = MULTI_SOURCE_TOP_K) -> dict[str, list[dict]]:
    source_results = {
        "codebase": search_codebase(query, top_k=max(1, per_source_top_k)),
        "documents": search_documents(query, top_k=max(1, per_source_top_k)),
        "apidocs": hybrid_search_apidocs(query, top_k=max(1, per_source_top_k)),
    }
    return _annotate_knowledge_source_hits(source_results)


def _query_prefers_codebase_source(query: str) -> bool:
    normalized = str(query or "").strip().lower()
    if not normalized:
        return False
    has_symbol = bool(IDENTIFIER_PATTERN.search(normalized))
    code_lookup_cues = (
        "在哪里定义",
        "在哪定义",
        "where is",
        "defined",
        "definition",
        "函数",
        "方法",
        "类",
        "symbol",
        "route_request",
        "search_knowledge_base",
    )
    return has_symbol and any(cue in normalized for cue in code_lookup_cues)


def _fuse_ranked_hits_rrf(source_hits: dict[str, list[dict]]) -> list[dict]:
    fused: dict[tuple[Any, ...], dict] = {}
    for source_type, hits in source_hits.items():
        for rank, hit in enumerate(hits, start=1):
            key = _knowledge_hit_key(hit)
            score = 1.0 / (RRF_K + float(rank))
            existing = fused.get(key)
            if existing is None:
                payload = dict(hit)
                payload["source_type"] = source_type
                payload["source_rank"] = min(int(hit.get("source_rank", rank) or rank), rank)
                payload["rrf_score"] = score
                payload["fusion_score"] = score
                fused[key] = payload
                continue
            existing["rrf_score"] = float(existing.get("rrf_score") or 0.0) + score
            existing["fusion_score"] = float(existing.get("rrf_score") or 0.0)
            existing["raw_score"] = max(float(existing.get("raw_score") or 0.0), float(hit.get("raw_score") or 0.0))
            existing["bm25_score"] = max(float(existing.get("bm25_score") or 0.0), float(hit.get("bm25_score") or 0.0))
            existing["source_rank"] = min(int(existing.get("source_rank") or rank), rank)
    ranked = list(fused.values())
    ranked.sort(
        key=lambda item: (
            float(item.get("fusion_score") or 0.0),
            -int(item.get("source_rank") or 999),
            float(item.get("raw_score") or 0.0),
            float(item.get("bm25_score") or 0.0),
        ),
        reverse=True,
    )
    return ranked


def _retain_diverse_candidates(
    fused_hits: list[dict],
    *,
    limit: int = FUSION_CANDIDATE_LIMIT,
    min_per_source: int = FUSION_MIN_PER_SOURCE,
) -> list[dict]:
    if not fused_hits:
        return []

    naive_top_keys = {_knowledge_hit_key(hit) for hit in fused_hits[: max(1, limit)]}
    grouped: dict[str, list[dict]] = {}
    source_order: list[str] = []
    for hit in fused_hits:
        source_type = str(hit.get("source_type", "")).strip().lower()
        if source_type not in grouped:
            grouped[source_type] = []
            source_order.append(source_type)
        grouped.setdefault(source_type, []).append(hit)

    selected: list[dict] = []
    selected_keys: set[tuple[Any, ...]] = set()

    for source_type in source_order:
        candidates = grouped.get(source_type, [])
        retain_count = min(len(candidates), max(0, min_per_source))
        for hit in candidates[:retain_count]:
            key = _knowledge_hit_key(hit)
            if key in selected_keys:
                continue
            payload = dict(hit)
            payload["is_fusion_backup"] = key not in naive_top_keys
            selected.append(payload)
            selected_keys.add(key)

    for hit in fused_hits:
        if len(selected) >= max(1, limit):
            break
        key = _knowledge_hit_key(hit)
        if key in selected_keys:
            continue
        payload = dict(hit)
        payload["is_fusion_backup"] = False
        selected.append(payload)
        selected_keys.add(key)

    selected.sort(
        key=lambda item: (
            float(item.get("fusion_score") or 0.0),
            -int(item.get("source_rank") or 999),
            float(item.get("raw_score") or 0.0),
            float(item.get("bm25_score") or 0.0),
        ),
        reverse=True,
    )
    return selected[: max(1, limit)]


def search_codebase(query: str, top_k: int = 5) -> list[dict]:
    identifiers = {match.group(0).lower() for match in IDENTIFIER_PATTERN.finditer(query or "")}
    search_variants = [query]
    search_variants.extend(
        identifier
        for identifier in sorted(identifiers)
        if identifier not in {query.lower(), *IDENTIFIER_STOPWORDS}
    )

    merged_hits: dict[tuple[str, str], dict] = {}
    for hit in _query_exact_code_symbols(identifiers):
        key = (
            str(hit.get("file_path", "")),
            str(hit.get("func_name", "")),
        )
        merged_hits[key] = hit
    for hit in _match_exact_code_symbols_from_corpus(identifiers):
        key = (
            str(hit.get("file_path", "")),
            str(hit.get("func_name", "")),
        )
        existing = merged_hits.get(key)
        if existing is None or float(hit.get("score") or 0.0) > float(existing.get("score") or 0.0):
            merged_hits[key] = hit

    for variant in search_variants:
        vector = get_query_embedding([variant])[0]
        for hit in _search_collection(
            collection_name=COLLECTION_NAME,
            vector=vector,
            top_k=max(top_k * 4, 20),
            metric_type="COSINE",
            search_params={"nprobe": 10},
            output_fields=["file_path", "func_name", "code"],
        ):
            key = (
                str(hit.get("file_path", "")),
                str(hit.get("func_name", "")),
            )
            existing = merged_hits.get(key)
            if existing is None or float(hit.get("score") or 0.0) > float(existing.get("score") or 0.0):
                merged_hits[key] = hit

    hits = list(merged_hits.values())
    bm25_hits = _search_bm25_corpus(
        COLLECTION_NAME,
        query,
        top_k=max(top_k * 4, 20),
        text_getter=lambda entry: entry.get("text", ""),
    )
    hits = _merge_hits(
        hits,
        bm25_hits,
        key_fields=("file_path", "func_name"),
    )

    def rerank_score(hit: dict) -> float:
        base_score = float(hit.get("score") or 0.0)
        bm25_score = float(hit.get("bm25_score") or 0.0)
        func_name = str(hit.get("func_name", "")).strip().lower()
        file_path = str(hit.get("file_path", "")).strip().lower()
        code = str(hit.get("code", "")).strip().lower()
        bonus = 0.0

        if func_name and func_name in identifiers:
            bonus += 1.0

        for identifier in identifiers:
            if identifier and identifier in file_path:
                bonus += 0.15
            if identifier and identifier in code:
                bonus += 0.1

        return base_score + (bm25_score * BM25_SCORE_WEIGHT) + bonus

    hits.sort(key=rerank_score, reverse=True)
    return hits[: max(1, top_k)]


def search_documents(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict]:
    vector = get_query_embedding([query])[0]
    expr = build_document_filter_expr(filters)
    hits = _search_collection(
        collection_name=DOCS_COLLECTION_NAME,
        vector=vector,
        top_k=top_k,
        metric_type="COSINE",
        search_params={"nprobe": 10},
        output_fields=DOCUMENT_OUTPUT_FIELDS,
        expr=expr,
    )
    for hit in hits:
        hit["tags"] = deserialize_tags(hit.get("tags"))
    if filters:
        hits = [hit for hit in hits if matches_document_filters(hit, filters)]
    bm25_hits = _search_bm25_corpus(
        DOCS_COLLECTION_NAME,
        query,
        top_k=top_k,
        text_getter=lambda entry: entry.get("search_text") or entry.get("text", ""),
    )
    if filters:
        bm25_hits = [hit for hit in bm25_hits if matches_document_filters(hit, filters)]
    hits = _merge_hits(
        hits,
        bm25_hits,
        key_fields=("source", "chunk_index"),
    )
    for hit in hits:
        hit["tags"] = deserialize_tags(hit.get("tags"))
    hits.sort(
        key=lambda hit: float(hit.get("score") or 0.0) + float(hit.get("bm25_score") or 0.0) * BM25_SCORE_WEIGHT,
        reverse=True,
    )
    return hits[: max(1, top_k)]


def hybrid_search_apidocs(query: str, top_k: int = 5) -> list[dict]:
    return search_apidocs(query, top_k=top_k)


def search_apidocs(query: str, top_k: int = 5) -> list[dict]:
    vector = get_query_embedding([query])[0]
    hits = _search_collection(
        collection_name=APIDOCS_COLLECTION_NAME,
        vector=vector,
        top_k=top_k,
        metric_type="COSINE",
        search_params={"ef": top_k * 4},
        output_fields=["doc_id", "source", "doc_type", "title", "content"],
    )
    bm25_hits = _search_bm25_corpus(
        APIDOCS_COLLECTION_NAME,
        query,
        top_k=top_k,
        text_getter=lambda entry: entry.get("content", ""),
    )
    hits = _merge_hits(
        hits,
        bm25_hits,
        key_fields=("doc_id", "source", "title"),
    )
    hits.sort(
        key=lambda hit: float(hit.get("score") or 0.0) + float(hit.get("bm25_score") or 0.0) * BM25_SCORE_WEIGHT,
        reverse=True,
    )
    return hits[: max(1, top_k)]


def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    if _query_prefers_codebase_source(query):
        source_hits = _search_all_knowledge_sources(query, per_source_top_k=MULTI_SOURCE_TOP_K)
    else:
        source_hits = _search_general_knowledge_sources(query, per_source_top_k=MULTI_SOURCE_TOP_K)
    fused_hits = _fuse_ranked_hits_rrf(source_hits)
    retained_hits = _retain_diverse_candidates(
        fused_hits,
        limit=max(FUSION_CANDIDATE_LIMIT, int(top_k or 0), 1),
        min_per_source=FUSION_MIN_PER_SOURCE,
    )
    return retained_hits[: max(1, int(top_k or 0))]


def _escape_expr_value(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def ensure_memory_collection(collection_name: str = MEMORY_COLLECTION_NAME) -> bool:
    _ensure_connection()
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        ensure_collection_vector_dim(collection, get_embedding_dim(), collection_name=collection_name)
        return True

    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("memory_id", DataType.VARCHAR, max_length=64),
        FieldSchema("project_id", DataType.VARCHAR, max_length=64),
        FieldSchema("thread_id", DataType.VARCHAR, max_length=128),
        FieldSchema("scope", DataType.VARCHAR, max_length=32),
        FieldSchema("memory_type", DataType.VARCHAR, max_length=64),
        FieldSchema("content", DataType.VARCHAR, max_length=4096),
        FieldSchema("summary", DataType.VARCHAR, max_length=512),
        FieldSchema("status", DataType.VARCHAR, max_length=32),
        FieldSchema("confidence", DataType.FLOAT),
        FieldSchema("importance", DataType.FLOAT),
        FieldSchema("created_at", DataType.INT64),
        FieldSchema("updated_at", DataType.INT64),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=get_embedding_dim()),
    ]
    schema = CollectionSchema(fields, description="CodeSage long-term chat memory")
    collection = Collection(collection_name, schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    collection.load()
    return True


def search_memory_items(
    thread_id: str,
    project_id: str,
    query: str,
    top_k: int = 3,
    collection_name: str = MEMORY_COLLECTION_NAME,
    policy: str = "default",
) -> list[dict]:
    if not query:
        return []

    vector = get_query_embedding([query])[0]
    clauses = ['status == "active"']
    if project_id:
        clauses.append(f'project_id == "{_escape_expr_value(project_id)}"')
    if policy == "review":
        if thread_id:
            clauses.append(
                f'(scope != "thread" or thread_id == "{_escape_expr_value(thread_id)}")'
            )
    elif thread_id:
        clauses.append(
            f'(scope != "thread" or thread_id == "{_escape_expr_value(thread_id)}")'
        )
    expr = " and ".join(clauses)
    hits = _search_collection(
        collection_name=collection_name,
        vector=vector,
        top_k=max(1, top_k),
        metric_type="COSINE",
        search_params={"nprobe": max(10, top_k * 2)},
        output_fields=[
            "memory_id",
            "project_id",
            "thread_id",
            "scope",
            "memory_type",
            "content",
            "summary",
            "status",
            "confidence",
            "importance",
            "created_at",
            "updated_at",
        ],
        expr=expr,
    )
    return hits


def _insert_memory_record(
    collection: Collection,
    memory_id: str,
    project_id: str,
    thread_id: str,
    scope: str,
    memory_type: str,
    content: str,
    summary: str,
    status: str,
    confidence: float,
    importance: float,
    created_at: int,
    updated_at: int,
    vector: list[float],
) -> None:
    collection.insert(
        [
            [memory_id[:64]],
            [project_id[:64]],
            [thread_id],
            [scope[:32]],
            [memory_type],
            [content[:4096]],
            [summary[:512]],
            [status[:32]],
            [float(confidence)],
            [float(importance)],
            [int(created_at)],
            [int(updated_at)],
            [vector],
        ]
    )


def upsert_memory_items(
    thread_id: str,
    project_id: str,
    items: list[dict[str, Any]],
    collection_name: str = MEMORY_COLLECTION_NAME,
) -> dict[str, int]:
    if not thread_id or not items:
        return {"inserted": 0, "updated": 0}

    try:
        ensure_memory_collection(collection_name=collection_name)
        collection = Collection(collection_name)
        collection.load()
    except Exception:
        return {"inserted": 0, "updated": 0}

    inserted = 0
    updated = 0
    safe_thread_id = thread_id[:128]
    safe_project_id = project_id[:64]

    for item in items:
        content = str(item.get("content", "")).strip()
        memory_type = str(item.get("memory_type", "")).strip()[:64]
        memory_id = str(item.get("memory_id", "")).strip()[:64]
        scope = str(item.get("scope", "thread") or "thread").strip()[:32]
        summary = str(item.get("summary", "") or content).strip()[:512]
        status = str(item.get("status", "active") or "active").strip()[:32]
        if not content or not memory_type:
            continue

        confidence = float(item.get("confidence", 0.0) or 0.0)
        importance = float(item.get("importance", 0.0) or 0.0)
        now = int(item.get("updated_at", int(time.time())) or int(time.time()))
        created_at = int(item.get("created_at", now) or now)
        vector = get_embedding([content])[0]
        if memory_id:
            expr = f'memory_id == "{_escape_expr_value(memory_id)}"'
        else:
            expr = (
                f'thread_id == "{_escape_expr_value(safe_thread_id)}" '
                f'and memory_type == "{_escape_expr_value(memory_type)}"'
            )

        try:
            results = collection.query(
                expr=expr,
                output_fields=[
                    "memory_id",
                    "project_id",
                    "thread_id",
                    "scope",
                    "memory_type",
                    "content",
                    "summary",
                    "status",
                    "confidence",
                    "importance",
                    "created_at",
                    "updated_at",
                ],
            )
        except Exception:
            results = []

        if results:
            try:
                collection.delete(expr=expr)
                updated += 1
            except Exception:
                pass

        try:
            _insert_memory_record(
                collection=collection,
                memory_id=memory_id or f"{safe_thread_id}:{memory_type}",
                project_id=safe_project_id,
                thread_id=safe_thread_id,
                scope=scope,
                memory_type=memory_type,
                content=content,
                summary=summary,
                status=status,
                confidence=confidence,
                importance=importance,
                created_at=created_at,
                updated_at=now,
                vector=vector,
            )
            if not results:
                inserted += 1
        except Exception:
            continue

    try:
        collection.flush()
    except Exception:
        pass
    return {"inserted": inserted, "updated": updated}

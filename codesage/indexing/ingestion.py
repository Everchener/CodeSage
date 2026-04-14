import ast
import logging
import os
from pathlib import Path
from typing import Literal

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from codesage.core.config import COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT, get_embedding, get_embedding_dim
from codesage.tools.bm25_embedding import (
    delete_bm25_artifacts,
    get_bm25_service,
    load_bm25_corpus,
    save_bm25_corpus,
)
from codesage.indexing.vector_utils import ensure_collection_vector_dim

logger = logging.getLogger(__name__)

INDEX_BATCH_SIZE = 50
INDEX_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".pytest_cache",
    ".codesage",
    ".chrome-headless",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "deepagents-main",
    "mem0-main",
    "ragflow-main",
}
INDEX_SKIP_DIR_PREFIXES = (
    ".tmp",
    ".pytest-tmp",
    ".skill-selfcheck",
    "pytest-cache-files-",
    "skill-test-",
    "codesage-rag-sample-",
    "pytest_run_",
)


def _truncate_utf8(text: str, max_bytes: int) -> str:
    value = text or ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""


def _build_embedding_text(chunk: dict) -> str:
    parts = [
        f"file_path: {chunk['file_path']}",
        f"func_name: {chunk['func_name']}",
        chunk["code"][:1000],
    ]
    return "\n".join(part for part in parts if part.strip())


def _get_functions(source: str, file_path: str) -> list[dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"func_name": "<module>", "file_path": file_path, "code": source[:500]}]
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            code = ast.get_source_segment(source, node) or ""
            chunks.append({"func_name": node.name, "file_path": file_path, "code": code})
    return chunks or [{"func_name": "<module>", "file_path": file_path, "code": source[:500]}]


def _ensure_collection():
    embedding_dim = get_embedding_dim()
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        ensure_collection_vector_dim(collection, embedding_dim, collection_name=COLLECTION_NAME)
        return collection
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("file_path", DataType.VARCHAR, max_length=512),
        FieldSchema("func_name", DataType.VARCHAR, max_length=256),
        FieldSchema("code", DataType.VARCHAR, max_length=4096),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]
    schema = CollectionSchema(fields)
    col = Collection(COLLECTION_NAME, schema)
    col.create_index("embedding", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    return col


def _resolve_index_roots(repo_path: str, include_paths: list[str] | None = None) -> list[Path]:
    root = Path(repo_path).resolve()
    if not include_paths:
        return [root]

    resolved_roots: list[Path] = []
    for item in include_paths:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate.exists():
            resolved_roots.append(candidate)
    return resolved_roots or [root]


def _should_skip_dir(name: str) -> bool:
    normalized = (name or "").strip()
    if not normalized:
        return False
    if normalized in INDEX_SKIP_DIR_NAMES:
        return True
    return any(normalized.startswith(prefix) for prefix in INDEX_SKIP_DIR_PREFIXES)


def _iter_python_files(repo_path: str, include_paths: list[str] | None = None) -> list[Path]:
    roots = _resolve_index_roots(repo_path, include_paths=include_paths)
    files: list[Path] = []
    seen: set[Path] = set()

    def _ignore_error(_exc):
        return None

    for root in roots:
        for current_root, dirnames, filenames in os.walk(root, topdown=True, onerror=_ignore_error):
            dirnames[:] = [name for name in sorted(dirnames) if not _should_skip_dir(name)]
            current_path = Path(current_root)
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = (current_path / filename).resolve()
                if path in seen:
                    continue
                seen.add(path)
                files.append(path)
    return files


def _connect_milvus() -> None:
    if not connections.has_connection("default"):
        connections.connect(alias="default", host=MILVUS_HOST, port=str(MILVUS_PORT))


def _escape_expr_value(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def _delete_entries_for_files(collection: Collection, file_paths: set[str]) -> None:
    sorted_paths = sorted(path for path in file_paths if path)
    if sorted_paths:
        collection.load()
    for offset in range(0, len(sorted_paths), INDEX_BATCH_SIZE):
        batch = sorted_paths[offset:offset + INDEX_BATCH_SIZE]
        if not batch:
            continue
        expr_values = ", ".join(f'"{_escape_expr_value(path)}"' for path in batch)
        collection.delete(expr=f"file_path in [{expr_values}]")


def _belongs_to_roots(file_path: str, roots: list[Path]) -> bool:
    try:
        resolved = Path(file_path).resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _build_chunk_entries(py_files: list[Path]) -> list[dict]:
    chunks: list[dict] = []
    for file_path in py_files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Skipping unreadable file during indexing: %s (%s)", file_path, exc)
            continue
        chunks.extend(_get_functions(source, str(file_path)))
    return chunks


def _build_corpus_entries(chunks: list[dict]) -> list[dict]:
    entries: list[dict] = []
    for chunk in chunks:
        embedding_text = _build_embedding_text(chunk)
        entries.append(
            {
                "file_path": _truncate_utf8(chunk["file_path"], 512),
                "func_name": _truncate_utf8(chunk["func_name"], 256),
                "code": _truncate_utf8(chunk["code"], 4096),
                "text": embedding_text,
            }
        )
    return entries


def _merge_corpus_entries(
    existing_entries: list[dict],
    new_entries: list[dict],
    refreshed_file_paths: set[str],
) -> list[dict]:
    retained_entries = [
        entry
        for entry in existing_entries
        if str(entry.get("file_path", "")) not in refreshed_file_paths
    ]
    return retained_entries + new_entries


def _reset_index_artifacts() -> None:
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    delete_bm25_artifacts(COLLECTION_NAME)


def index_repository(
    repo_path: str,
    include_paths: list[str] | None = None,
    mode: Literal["incremental", "rebuild"] = "incremental",
):
    normalized_mode: Literal["incremental", "rebuild"] = "rebuild" if mode == "rebuild" else "incremental"
    _connect_milvus()
    if normalized_mode == "rebuild":
        _reset_index_artifacts()
    col = _ensure_collection()

    roots = _resolve_index_roots(repo_path, include_paths=include_paths)
    py_files = _iter_python_files(repo_path, include_paths=include_paths)
    indexed_file_paths = {str(path) for path in py_files}
    existing_corpus = [] if normalized_mode == "rebuild" else load_bm25_corpus(COLLECTION_NAME)
    stale_file_paths = {
        str(entry.get("file_path", ""))
        for entry in existing_corpus
        if _belongs_to_roots(str(entry.get("file_path", "")), roots)
        and str(entry.get("file_path", "")) not in indexed_file_paths
    }
    refreshed_file_paths = indexed_file_paths | stale_file_paths
    if normalized_mode == "incremental" and refreshed_file_paths:
        _delete_entries_for_files(col, refreshed_file_paths)

    all_chunks = _build_chunk_entries(py_files)
    for i in range(0, len(all_chunks), INDEX_BATCH_SIZE):
        batch = all_chunks[i:i + INDEX_BATCH_SIZE]
        texts = [_build_embedding_text(c) for c in batch]
        vectors = get_embedding(texts)
        col.insert(
            [
                [_truncate_utf8(c["file_path"], 512) for c in batch],
                [_truncate_utf8(c["func_name"], 256) for c in batch],
                [_truncate_utf8(c["code"], 4096) for c in batch],
                vectors,
            ]
        )
    col.flush()

    new_corpus_entries = _build_corpus_entries(all_chunks)
    if normalized_mode == "rebuild":
        corpus_entries = new_corpus_entries
    else:
        corpus_entries = _merge_corpus_entries(existing_corpus, new_corpus_entries, refreshed_file_paths)
    corpus_texts = [str(entry.get("text", "")) for entry in corpus_entries]
    service = get_bm25_service(COLLECTION_NAME)
    service.fit_corpus(corpus_texts)
    save_bm25_corpus(COLLECTION_NAME, corpus_entries)
    result = {
        "status": "completed",
        "mode": normalized_mode,
        "files_indexed": len(py_files),
        "chunks_indexed": len(all_chunks),
        "files_refreshed": len(refreshed_file_paths),
    }
    logger.info(
        "Indexed %s chunks from %s files using %s mode.",
        result["chunks_indexed"],
        result["files_indexed"],
        normalized_mode,
    )
    return result

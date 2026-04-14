import hashlib
import logging
import math
import os
import re
from functools import lru_cache
from typing import Any, Protocol

from dotenv import load_dotenv

from codesage.core.error_handling import read_env_float, read_env_int


logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_load_dotenv() -> bool:
    if _env_flag("CODESAGE_SKIP_DOTENV"):
        return False
    # pytest 会在测试中重新加载配置；这里跳过 dotenv 可避免
    # 开发者本地 .env 泄漏到单元测试预期中。
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return True


if _should_load_dotenv():
    load_dotenv()


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GITHUB_HTTP_TIMEOUT_SECONDS = read_env_float(
    "GITHUB_HTTP_TIMEOUT_SECONDS",
    10.0,
    logger=logger,
    module=__name__,
)
PR_REVIEW_MAX_DIFF_BYTES = read_env_int(
    "PR_REVIEW_MAX_DIFF_BYTES",
    262144,
    logger=logger,
    module=__name__,
)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = read_env_int("MILVUS_PORT", 19530, logger=logger, module=__name__)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "codesage")

LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("MINIMAX_API_KEY", "")).strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("MINIMAX_CHAT_BASE_URL", "")).strip()
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("MINIMAX_CHAT_MODEL", "")).strip()

# 为旧的品牌专用导入保留向后兼容别名。
MINIMAX_API_KEY = LLM_API_KEY
MINIMAX_CHAT_BASE_URL = LLM_BASE_URL
MINIMAX_CHAT_MODEL = LLM_MODEL

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hash").strip().lower() or "hash"
EMBEDDING_DIM = read_env_int("EMBEDDING_DIM", 512, logger=logger, module=__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MEMORY_ENABLED = _env_bool("MEMORY_ENABLED", True)
MEMORY_SHORT_WINDOW_TURNS = read_env_int(
    "MEMORY_SHORT_WINDOW_TURNS",
    8,
    logger=logger,
    module=__name__,
)
MEMORY_SHORT_TTL_MINUTES = read_env_int(
    "MEMORY_SHORT_TTL_MINUTES",
    120,
    logger=logger,
    module=__name__,
)
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "codesage_memory")
MEMORY_LONG_TOP_K = read_env_int("MEMORY_LONG_TOP_K", 3, logger=logger, module=__name__)
MEMORY_WRITE_MIN_CONFIDENCE = read_env_float(
    "MEMORY_WRITE_MIN_CONFIDENCE",
    0.75,
    logger=logger,
    module=__name__,
)
BGEM3_MODEL_NAME = os.getenv("BGEM3_MODEL_NAME", "BAAI/bge-m3").strip() or "BAAI/bge-m3"
BGEM3_DEVICE = os.getenv("BGEM3_DEVICE", "cpu").strip() or "cpu"
BGEM3_BATCH_SIZE = read_env_int("BGEM3_BATCH_SIZE", 16, logger=logger, module=__name__)
BGEM3_USE_FP16 = _env_bool("BGEM3_USE_FP16", False)


class EmbeddingProvider(Protocol):
    dim: int
    model_name: str

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbeddingFunction:
    """为轻量级本地启动提供确定性的后备嵌入实现。"""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.model_name = "hash"

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower()) or [text or ""]

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                chunk = digest[offset:offset + 4]
                index = int.from_bytes(chunk, "big") % self.dim
                sign = 1.0 if chunk[0] % 2 == 0 else -1.0
                vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self.encode_documents(texts)


def _to_vector_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


class DefaultEmbeddingProvider:
    def __init__(self):
        from pymilvus import model

        self._provider = model.DefaultEmbeddingFunction()
        self.dim = int(getattr(self._provider, "dim", EMBEDDING_DIM))
        self.model_name = getattr(self._provider, "model_name", "all-MiniLM-L6-v2")

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_to_vector_list(vector) for vector in self._provider.encode_documents(texts)]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encode_queries = getattr(self._provider, "encode_queries", self._provider.encode_documents)
        return [_to_vector_list(vector) for vector in encode_queries(texts)]


class BGEM3EmbeddingProvider:
    def __init__(self):
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        self._provider = BGEM3EmbeddingFunction(
            model_name=BGEM3_MODEL_NAME,
            device=BGEM3_DEVICE,
            use_fp16=BGEM3_USE_FP16,
            batch_size=BGEM3_BATCH_SIZE,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            normalize_embeddings=True,
        )
        provider_dim = getattr(self._provider, "dim", {})
        dense_dim = provider_dim.get("dense") if isinstance(provider_dim, dict) else provider_dim
        if dense_dim is None:
            raise ValueError("BGEM3EmbeddingFunction did not expose a dense embedding dimension.")
        self.dim = int(dense_dim)
        self.model_name = BGEM3_MODEL_NAME

    def _extract_dense_vectors(self, payload: Any) -> list[list[float]]:
        if not isinstance(payload, dict) or "dense" not in payload:
            raise ValueError("BGEM3EmbeddingFunction returned no dense embeddings.")
        return [_to_vector_list(vector) for vector in payload["dense"]]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._extract_dense_vectors(self._provider.encode_documents(texts))

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._extract_dense_vectors(self._provider.encode_queries(texts))


def _build_embedding_provider() -> tuple[EmbeddingProvider, str]:
    if EMBEDDING_BACKEND in {"milvus", "bgem3", "bge-m3"}:
        return BGEM3EmbeddingProvider(), "bgem3"
    if EMBEDDING_BACKEND == "defaultembeddingfunction":
        return DefaultEmbeddingProvider(), "defaultembeddingfunction"
    if EMBEDDING_BACKEND == "hash":
        return HashEmbeddingFunction(), "hash"
    raise ValueError(f"Unsupported embedding backend: {EMBEDDING_BACKEND}")


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    provider, _ = _build_embedding_provider()
    return provider


@lru_cache(maxsize=1)
def get_embedding_backend_name() -> str:
    _, backend_name = _build_embedding_provider()
    return backend_name


def get_embedding_status() -> dict:
    try:
        provider = get_embedding_provider()
        return {
            "backend": get_embedding_backend_name(),
            "configured_backend": EMBEDDING_BACKEND,
            "model": getattr(provider, "model_name", ""),
            "dim": getattr(provider, "dim", EMBEDDING_DIM),
            "healthy": True,
            "error": "",
        }
    except Exception as exc:
        return {
            "backend": "",
            "configured_backend": EMBEDDING_BACKEND,
            "model": BGEM3_MODEL_NAME if EMBEDDING_BACKEND in {"milvus", "bgem3", "bge-m3"} else "",
            "dim": None,
            "healthy": False,
            "error": str(exc),
        }


def get_embedding_dim() -> int:
    provider = get_embedding_provider()
    return int(getattr(provider, "dim", EMBEDDING_DIM))


def get_embedding(texts: list[str]) -> list[list[float]]:
    provider = get_embedding_provider()
    return provider.encode_documents(texts)


def get_query_embedding(texts: list[str]) -> list[list[float]]:
    provider = get_embedding_provider()
    return provider.encode_queries(texts)

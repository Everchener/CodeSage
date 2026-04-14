import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from codesage.tools.file_io import write_json_file

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _sanitize_namespace(namespace: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (namespace or "").strip())
    return normalized or "default"


def _state_path(namespace: str) -> Path:
    return _DATA_DIR / f"bm25_state_{_sanitize_namespace(namespace)}.json"


def _corpus_path(namespace: str) -> Path:
    return _DATA_DIR / f"bm25_corpus_{_sanitize_namespace(namespace)}.json"


class BM25EmbeddingService:
    """用于处理中英混合文本和代码标识符的稀疏 BM25 辅助类。"""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.k1 = 1.5
        self.b = 0.75
        self._vocab: Dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter = Counter()
        self._total_docs = 0
        self._avg_doc_len = 0.0
        self._load_state()

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        chinese_pat = re.compile(r"[\u4e00-\u9fff]")
        english_pat = re.compile(r"[a-zA-Z0-9_]+")

        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pat.match(char):
                tokens.append(char)
                i += 1
                continue

            match = english_pat.match(text[i:])
            if match:
                word = match.group()
                tokens.extend(self._split_identifier(word))
                i += len(word)
                continue

            i += 1
        return tokens

    @staticmethod
    def _split_identifier(word: str) -> List[str]:
        parts = word.split("_")
        result = []
        for part in parts:
            if not part:
                continue
            sub = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part).split()
            result.extend(segment.lower() for segment in sub)
        if len(result) > 1:
            result.append(word.lower())
        elif not result:
            result = [word.lower()]
        return result

    def fit_corpus(self, texts: List[str]) -> None:
        self._total_docs = len(texts)
        total_len = 0
        self._doc_freq = Counter()
        self._vocab = {}
        self._vocab_counter = 0

        for text in texts:
            tokens = self.tokenize(text)
            total_len += len(tokens)
            for token in set(tokens):
                self._doc_freq[token] += 1
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_counter
                    self._vocab_counter += 1

        self._avg_doc_len = total_len / self._total_docs if self._total_docs > 0 else 1.0
        self._save_state()

    def get_sparse_embedding(self, text: str) -> Dict[int, float]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse: Dict[int, float] = {}

        for token, freq in tf.items():
            if token not in self._vocab:
                continue

            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)
            if df == 0:
                idf = math.log((self._total_docs + 1) / 1)
            else:
                idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
            score = idf * numerator / denominator
            if score > 0:
                sparse[idx] = float(score)

        return sparse

    def get_sparse_embeddings(self, texts: List[str]) -> List[Dict[int, float]]:
        return [self.get_sparse_embedding(text) for text in texts]

    @staticmethod
    def _dot_sparse_vectors(left: Dict[int, float], right: Dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(index, 0.0) for index, value in left.items())

    def score_texts(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        query_vector = self.get_sparse_embedding(query)
        if not query_vector:
            return [0.0 for _ in texts]
        return [
            float(self._dot_sparse_vectors(query_vector, self.get_sparse_embedding(text)))
            for text in texts
        ]

    def _save_state(self) -> None:
        state = {
            "vocab": self._vocab,
            "vocab_counter": self._vocab_counter,
            "doc_freq": dict(self._doc_freq),
            "total_docs": self._total_docs,
            "avg_doc_len": self._avg_doc_len,
        }
        write_json_file(self.state_path, state)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            self._vocab = {key: int(value) for key, value in state.get("vocab", {}).items()}
            self._vocab_counter = int(state.get("vocab_counter", 0))
            self._doc_freq = Counter(state.get("doc_freq", {}))
            self._total_docs = int(state.get("total_docs", 0))
            self._avg_doc_len = float(state.get("avg_doc_len", 0.0))
        except Exception:
            pass


def save_bm25_corpus(namespace: str, entries: List[Dict[str, Any]]) -> None:
    path = _corpus_path(namespace)
    write_json_file(path, entries)


def append_bm25_corpus(namespace: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    payload = load_bm25_corpus(namespace)
    payload.extend(entries)
    save_bm25_corpus(namespace, payload)


def load_bm25_corpus(namespace: str) -> List[Dict[str, Any]]:
    path = _corpus_path(namespace)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def delete_bm25_artifacts(namespace: str) -> None:
    for path in (_state_path(namespace), _corpus_path(namespace)):
        try:
            path.unlink()
        except (FileNotFoundError, PermissionError):
            continue


_bm25_services: dict[str, BM25EmbeddingService] = {}


def get_bm25_service(namespace: str = "apidocs") -> BM25EmbeddingService:
    normalized = _sanitize_namespace(namespace)
    service = _bm25_services.get(normalized)
    if service is None:
        service = BM25EmbeddingService(state_path=_state_path(normalized))
        _bm25_services[normalized] = service
    return service

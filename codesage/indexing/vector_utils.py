from __future__ import annotations

from typing import Any


def _coerce_dim(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_collection_vector_dim(collection: Any, field_name: str = "embedding") -> int | None:
    schema = getattr(collection, "schema", None)
    fields = getattr(schema, "fields", None) or []
    for field in fields:
        if getattr(field, "name", "") != field_name:
            continue

        params = getattr(field, "params", None)
        if isinstance(params, dict):
            dim = _coerce_dim(params.get("dim"))
            if dim is not None:
                return dim

        to_dict = getattr(field, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                dim = _coerce_dim((payload.get("params") or {}).get("dim"))
                if dim is not None:
                    return dim

        dim = _coerce_dim(getattr(field, "dim", None))
        if dim is not None:
            return dim

    return None


def ensure_collection_vector_dim(
    collection: Any,
    expected_dim: int,
    *,
    collection_name: str,
    field_name: str = "embedding",
) -> None:
    actual_dim = get_collection_vector_dim(collection, field_name=field_name)
    if actual_dim is None:
        return
    if actual_dim != int(expected_dim):
        raise ValueError(
            f"Collection '{collection_name}' uses embedding dim {actual_dim}, "
            f"but the active provider outputs dim {expected_dim}. Rebuild the collection before querying or indexing."
        )

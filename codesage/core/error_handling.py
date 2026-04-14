from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ExceptionContext:
    operation: str
    module: str
    fallback: Any = None
    target: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExceptionResolution:
    fallback: Any
    log_level: int
    message: str
    include_traceback: bool = False


class ExceptionHandler(Protocol):
    def can_handle(self, exc: Exception, context: ExceptionContext) -> bool:
        ...

    def handle(self, exc: Exception, context: ExceptionContext) -> ExceptionResolution:
        ...


def _format_target(target: str) -> str:
    target_text = str(target or "").strip()
    if not target_text:
        return ""
    return f" target={target_text}"


class FileSystemExceptionHandler:
    def can_handle(self, exc: Exception, context: ExceptionContext) -> bool:
        return isinstance(exc, OSError)

    def handle(self, exc: Exception, context: ExceptionContext) -> ExceptionResolution:
        return ExceptionResolution(
            fallback=context.fallback,
            log_level=logging.WARNING,
            message=(
                f"Filesystem error while {context.operation} in {context.module}"
                f"{_format_target(context.target)}; using fallback."
            ),
        )


class DataParsingExceptionHandler:
    _SUPPORTED_KINDS = {"json", "timestamp", "number", "text"}

    def can_handle(self, exc: Exception, context: ExceptionContext) -> bool:
        if isinstance(exc, (json.JSONDecodeError, UnicodeError)):
            return True
        data_kind = str(context.metadata.get("kind", "")).strip().lower()
        return data_kind in self._SUPPORTED_KINDS and isinstance(
            exc,
            (TypeError, ValueError, OverflowError),
        )

    def handle(self, exc: Exception, context: ExceptionContext) -> ExceptionResolution:
        data_kind = str(context.metadata.get("kind", "data") or "data")
        return ExceptionResolution(
            fallback=context.fallback,
            log_level=logging.WARNING,
            message=(
                f"Failed to parse {data_kind} while {context.operation} in {context.module}"
                f"{_format_target(context.target)}; using fallback."
            ),
        )


class ConfigExceptionHandler:
    def can_handle(self, exc: Exception, context: ExceptionContext) -> bool:
        return (
            str(context.metadata.get("kind", "")).strip().lower() == "config"
            and isinstance(exc, (TypeError, ValueError, OverflowError))
        )

    def handle(self, exc: Exception, context: ExceptionContext) -> ExceptionResolution:
        env_name = str(context.metadata.get("env_name", context.target) or "").strip()
        raw_value = context.metadata.get("raw_value")
        default_value = context.metadata.get("default")
        return ExceptionResolution(
            fallback=context.fallback,
            log_level=logging.WARNING,
            message=(
                f"Invalid config value for {env_name or 'unknown_env'} in {context.module}"
                f" raw_value={raw_value!r} default={default_value!r}; using fallback."
            ),
        )


class UnexpectedExceptionHandler:
    def can_handle(self, exc: Exception, context: ExceptionContext) -> bool:
        return True

    def handle(self, exc: Exception, context: ExceptionContext) -> ExceptionResolution:
        return ExceptionResolution(
            fallback=context.fallback,
            log_level=logging.ERROR,
            message=(
                f"Unexpected error while {context.operation} in {context.module}"
                f"{_format_target(context.target)}; using fallback."
            ),
            include_traceback=True,
        )


class ExceptionManager:
    def __init__(self, handlers: list[ExceptionHandler] | None = None):
        self.handlers = list(
            handlers
            or [
                ConfigExceptionHandler(),
                FileSystemExceptionHandler(),
                DataParsingExceptionHandler(),
                UnexpectedExceptionHandler(),
            ]
        )

    def handle(
        self,
        exc: Exception,
        *,
        context: ExceptionContext,
        logger: logging.Logger | None = None,
    ) -> Any:
        active_logger = logger or logging.getLogger(context.module)
        resolution = self._resolve(exc, context)
        if resolution.include_traceback:
            active_logger.exception("%s error=%s", resolution.message, exc)
        else:
            active_logger.log(resolution.log_level, "%s error=%s", resolution.message, exc)
        return resolution.fallback

    def call(
        self,
        operation: Callable[[], T],
        *,
        context: ExceptionContext,
        logger: logging.Logger | None = None,
    ) -> T:
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - exercised via integration sites
            return self.handle(exc, context=context, logger=logger)

    def _resolve(self, exc: Exception, context: ExceptionContext) -> ExceptionResolution:
        for handler in self.handlers:
            if handler.can_handle(exc, context):
                return handler.handle(exc, context)
        return UnexpectedExceptionHandler().handle(exc, context)


DEFAULT_EXCEPTION_MANAGER = ExceptionManager()


def safe_call(
    operation: Callable[[], T],
    *,
    context: ExceptionContext,
    logger: logging.Logger | None = None,
    manager: ExceptionManager | None = None,
) -> T:
    active_manager = manager or DEFAULT_EXCEPTION_MANAGER
    return active_manager.call(operation, context=context, logger=logger)


def safe_read_text(
    path: Path,
    *,
    fallback: str,
    logger: logging.Logger | None = None,
    module: str,
    operation: str,
    encoding: str = "utf-8",
) -> str:
    return safe_call(
        lambda: path.read_text(encoding=encoding),
        context=ExceptionContext(
            operation=operation,
            module=module,
            target=str(path),
            fallback=fallback,
            metadata={"kind": "text"},
        ),
        logger=logger,
    )


def safe_write_text(
    path: Path,
    content: str,
    *,
    logger: logging.Logger | None = None,
    module: str,
    operation: str,
    encoding: str = "utf-8",
) -> bool:
    def _write() -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return True

    return safe_call(
        _write,
        context=ExceptionContext(
            operation=operation,
            module=module,
            target=str(path),
            fallback=False,
            metadata={"kind": "text"},
        ),
        logger=logger,
    )


def safe_append_text(
    path: Path,
    content: str,
    *,
    logger: logging.Logger | None = None,
    module: str,
    operation: str,
    encoding: str = "utf-8",
) -> bool:
    def _append() -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding=encoding) as handle:
            handle.write(content)
        return True

    return safe_call(
        _append,
        context=ExceptionContext(
            operation=operation,
            module=module,
            target=str(path),
            fallback=False,
            metadata={"kind": "text"},
        ),
        logger=logger,
    )


def safe_json_loads(
    value: str,
    *,
    fallback: Any,
    logger: logging.Logger | None = None,
    module: str,
    operation: str,
    target: str = "",
) -> Any:
    return safe_call(
        lambda: json.loads(value),
        context=ExceptionContext(
            operation=operation,
            module=module,
            target=target,
            fallback=fallback,
            metadata={"kind": "json"},
        ),
        logger=logger,
    )


def safe_coerce_float(
    value: Any,
    *,
    default: float,
    logger: logging.Logger | None = None,
    module: str,
    operation: str,
    target: str = "",
    kind: str = "number",
) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return safe_call(
        lambda: float(value),
        context=ExceptionContext(
            operation=operation,
            module=module,
            target=target,
            fallback=default,
            metadata={"kind": kind},
        ),
        logger=logger,
    )


def read_env_int(
    name: str,
    default: int,
    *,
    logger: logging.Logger | None = None,
    module: str,
) -> int:
    raw_value = os.getenv(name, str(default))
    return safe_call(
        lambda: int(raw_value),
        context=ExceptionContext(
            operation="read environment integer",
            module=module,
            target=name,
            fallback=default,
            metadata={
                "kind": "config",
                "env_name": name,
                "raw_value": raw_value,
                "default": default,
            },
        ),
        logger=logger,
    )


def read_env_float(
    name: str,
    default: float,
    *,
    logger: logging.Logger | None = None,
    module: str,
) -> float:
    raw_value = os.getenv(name, str(default))
    return safe_call(
        lambda: float(raw_value),
        context=ExceptionContext(
            operation="read environment float",
            module=module,
            target=name,
            fallback=default,
            metadata={
                "kind": "config",
                "env_name": name,
                "raw_value": raw_value,
                "default": default,
            },
        ),
        logger=logger,
    )

"""Process-wide logging: readable console output + rotating JSON-lines file.

Every record is tagged with the current ``run_id`` / ``thread_id`` (set via :func:`log_context`), so
``logs/app.jsonl`` can be filtered to a single research run. Structured fields passed through
``logger.info(..., extra={...})`` are kept as JSON keys rather than flattened into the message.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_LOG_NAME = "app.jsonl"
_HANDLER_MARK = "_deep_research_handler"
_STANDARD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "", None, None))) | {"message", "asctime"}

_log_context: ContextVar[dict[str, str]] = ContextVar("log_context", default={})  # noqa: B039 - read-only default


@contextmanager
def log_context(**fields: str) -> Iterator[None]:
    """Tag every log record emitted in this context (and child tasks) with ``fields``."""
    token = _log_context.set({**_log_context.get(), **{k: v for k, v in fields.items() if v}})
    try:
        yield
    finally:
        try:
            _log_context.reset(token)
        except ValueError:  # finalized from another context (e.g. async-generator cleanup)
            pass


def current_log_context() -> dict[str, str]:
    return dict(_log_context.get())


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _log_context.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message, run context and extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(log_dir: str | Path = "logs", level: str = "INFO", *, console: bool = True,
                      max_bytes: int = 5_000_000, backups: int = 5) -> Path:
    """Install the console + rotating JSON file handlers once (idempotent). Returns the log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / APP_LOG_NAME

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backups,
                                                        encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    handlers: list[logging.Handler] = [file_handler]
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                                       datefmt="%H:%M:%S"))
        handlers.append(console_handler)
    for handler in handlers:
        setattr(handler, _HANDLER_MARK, True)
        handler.addFilter(_ContextFilter())
        root.addHandler(handler)

    # Third-party libraries are chatty at INFO (one line per HTTP request); keep them at WARNING.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "trafilatura", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path

"""Provider-agnostic error classification.

Every failure from an LLM or search provider is mapped to an :class:`ErrorKind` that tells the
router *what to blame* and therefore *what to do next*:

==================  ===================================  ======================================
Kind                Typical cause                        Router action
==================  ===================================  ======================================
RATE_LIMIT          429, RPM/TPM exceeded                cool the key for this model, next key
QUOTA_EXHAUSTED     402, daily/monthly credits gone      cool the key for all models (long)
AUTH                401/403, invalid or revoked key      disable the key permanently
CONTEXT_TOO_LARGE   413, prompt over model/TPM cap       skip this model (not the key's fault)
MODEL_UNAVAILABLE   404, decommissioned/unknown model    mark model dead for a while
TOOL_CALL_FAILED    model emitted an invalid tool call   try a different model
TRANSIENT           5xx, timeout, connection, overload   short cooldown, next provider
INCOMPATIBLE        other 400/422 request rejections     try a different provider/model
FATAL               bugs in our code, unknown errors     re-raise immediately
==================  ===================================  ======================================
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Any

import httpx

MAX_RETRY_AFTER = 900.0


class ErrorKind(str, Enum):
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH = "auth"
    CONTEXT_TOO_LARGE = "context_too_large"
    MODEL_UNAVAILABLE = "model_unavailable"
    TOOL_CALL_FAILED = "tool_call_failed"
    TRANSIENT = "transient"
    INCOMPATIBLE = "incompatible"
    FATAL = "fatal"


@dataclass(frozen=True)
class ClassifiedError:
    kind: ErrorKind
    message: str
    status: int | None = None
    retry_after: float | None = None
    detail: str = ""
    """The provider's own human-readable error message (from the response body), if any."""

    @property
    def recoverable(self) -> bool:
        return self.kind is not ErrorKind.FATAL


_STATUS_PREFIX_RE = re.compile(r"^error code:\s*\d+\s*-?\s*", re.I)


def error_detail(exc: BaseException) -> str:
    """Extract the provider's error message from an SDK exception.

    Handles OpenAI-style ``{"error": {"message": ...}}`` bodies, Gemini's list-wrapped variant
    ``[{"error": {...}}]``, Mistral's flat ``{"message": ...}`` and plain strings; falls back to
    ``str(exc)`` without the ``Error code: NNN -`` prefix.

    >>> error_detail(type("E", (Exception,), {"body": [{"error": {"message": "missing  sig"}}]})())
    'missing sig'
    """
    body: Any = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    text = ""
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            text = str(err.get("message") or err.get("detail") or "")
        elif isinstance(err, str):
            text = err
        text = text or str(body.get("message") or body.get("detail") or "")
    elif isinstance(body, str):
        text = body
    if not text:
        text = _STATUS_PREFIX_RE.sub("", str(exc))
    return " ".join(text.split())


_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")
_MSG_RETRY_RE = re.compile(
    r"(?:try again in|retry in|retry after|retrydelay['\"]?:\s*['\"]?)\s*([\d.]+(?:ms|s|m|h)?(?:[\d.]+(?:ms|s))?)",
    re.I,
)


def parse_duration(text: str | None) -> float | None:
    """Parse ``"7.66s"``, ``"1m30.5s"``, ``"120ms"``, ``"2h"`` or a bare number of seconds.

    >>> parse_duration("1m30.5s"), parse_duration("120ms"), parse_duration("7"), parse_duration("x")
    (90.5, 0.12, 7.0, None)
    """
    if not text:
        return None
    text = text.strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    m = _DURATION_RE.match(text)
    if not m or not any(m.groups()):
        return None
    h, mins, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mins * 60 + s + ms / 1000


def _status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _headers(exc: BaseException) -> Any:
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or {}


def extract_retry_after(exc: BaseException) -> float | None:
    """Best-effort ``Retry-After`` from headers (standard + Groq/OpenAI style) or the message."""
    headers = _headers(exc)
    candidates: list[float] = []
    try:
        if ms := headers.get("retry-after-ms"):
            candidates.append(float(ms) / 1000)
        if ra := headers.get("retry-after"):
            parsed = parse_duration(ra)
            if parsed is None:
                try:
                    from datetime import datetime, timezone

                    parsed = (parsedate_to_datetime(ra) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError):
                    parsed = None
            if parsed is not None:
                candidates.append(parsed)
        for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            if (val := parse_duration(headers.get(name))) is not None:
                candidates.append(val)
    except Exception:  # noqa: BLE001 - header shapes vary across SDKs
        pass
    if not candidates and (m := _MSG_RETRY_RE.search(str(exc))):
        if (val := parse_duration(m.group(1))) is not None:
            candidates.append(val)
    positive = [c for c in candidates if c and c > 0]
    return min(max(positive), MAX_RETRY_AFTER) if positive else None


def _any(msg: str, *terms: str) -> bool:
    return any(t in msg for t in terms)


def classify_error(exc: BaseException) -> ClassifiedError:
    """Map an exception from any provider SDK to an :class:`ErrorKind`."""
    status = _status(exc)
    name = type(exc).__name__
    msg = f"{name}: {exc}".lower()
    message = str(exc) or name

    def make(kind: ErrorKind, retry: bool = False) -> ClassifiedError:
        return ClassifiedError(kind, message, status, extract_retry_after(exc) if retry else None,
                               detail=error_detail(exc))

    # --- Tavily SDK exceptions (no status code attached) ----------------------------------
    if name in {"UsageLimitExceededError", "TavilyKeylessLimitError"}:
        return make(ErrorKind.RATE_LIMIT, retry=True)
    if name == "ForbiddenError":  # Tavily raises this for 403 and plan limits (432/433)
        return make(ErrorKind.QUOTA_EXHAUSTED)
    if name in {"InvalidAPIKeyError", "MissingAPIKeyError"}:
        return make(ErrorKind.AUTH)

    # --- Order matters: "request too large ... tokens per minute" is a size problem ---------
    if status == 413 or _any(msg, "request too large", "context length", "context_length", "maximum context",
                              "too many tokens", "reduce the length", "prompt is too long", "input is too long"):
        return make(ErrorKind.CONTEXT_TOO_LARGE)
    if status == 402 or _any(msg, "insufficient_quota", "insufficient credits", "credit balance", "billing",
                              "exceeded your current quota", "payment required"):
        return make(ErrorKind.QUOTA_EXHAUSTED, retry=True)
    if status == 429 or _any(msg, "rate limit", "rate_limit", "too many requests", "tokens per minute",
                              "requests per minute", "resource_exhausted", "resource exhausted"):
        return make(ErrorKind.RATE_LIMIT, retry=True)
    if status in (401, 403) or _any(msg, "invalid api key", "invalid_api_key", "incorrect api key",
                                     "unauthorized", "permission denied", "api key not valid"):
        return make(ErrorKind.AUTH)
    if status == 404 or _any(msg, "model_not_found", "does not exist", "decommissioned", "no endpoints found",
                              "model not found", "unknown model", "is not a valid model"):
        return make(ErrorKind.MODEL_UNAVAILABLE)
    if _any(msg, "tool_use_failed", "failed to call a function", "failed_generation", "tool call validation",
            "invalid tool call", "could not parse tool"):
        return make(ErrorKind.TOOL_CALL_FAILED)
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, asyncio.TimeoutError, TimeoutError,
                        ConnectionError)) or name in {"APITimeoutError", "APIConnectionError"} \
            or status in (408, 409, 425, 500, 502, 503, 504, 520, 522, 524, 529) \
            or _any(msg, "overloaded", "service unavailable", "temporarily unavailable", "timed out", "timeout",
                    "capacity", "bad gateway", "connection error", "connection reset"):
        return make(ErrorKind.TRANSIENT, retry=True)
    if status in (400, 422):
        return make(ErrorKind.INCOMPATIBLE)
    return make(ErrorKind.FATAL)

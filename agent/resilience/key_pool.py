"""Multi-key pools with round-robin rotation, proactive rate budgets and failure quarantine.

A :class:`KeyPool` holds every API key configured for one provider. Each key is a
:class:`KeySlot` that tracks, *per scope* (normally the model name, because providers such as Groq
and Gemini enforce limits per model):

* **proactive budgets**: token buckets for requests-per-minute and tokens-per-minute. A request is
  only sent when the bucket says it fits, so we route *around* a 429 instead of eating one;
* **reactive quarantine**: cooldowns after 429 / quota / transient errors, honouring the
  provider's ``Retry-After`` when available and backing off exponentially otherwise;
* **health**: permanent disable on authentication failures, plus call/failure counters for the UI.

All state is mutated from a single asyncio event loop (no ``await`` between check and update), so
no locks are needed.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent.resilience.errors import ClassifiedError, ErrorKind

ALL_SCOPES = "*"


def mask_key(key: str) -> str:
    """Mask an API key for logs and UI, e.g. ``gsk_...3a4f``.

    >>> mask_key("gsk_1234567890abcdef")
    'gsk_...cdef'
    """
    if not key:
        return "<none>"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def parse_keys(env: Mapping[str, str], prefix: str) -> list[str]:
    """Collect every key for ``prefix`` from ``env``, deduplicated, in a stable order.

    Supported forms (any combination; unlimited numbered keys)::

        GROQ_API_KEYS=k1,k2,k3
        GROQ_API_KEY=k1
        GROQ_API_KEY_2=k2   GROQ_API_KEY_3=k3   GROQ_API_KEY_17=...

    >>> parse_keys({"X_API_KEYS": "a, b", "X_API_KEY": "a", "X_API_KEY_10": "d", "X_API_KEY_2": "c"}, "X")
    ['a', 'b', 'c', 'd']
    """
    keys: list[str] = []

    def add(value: str | None) -> None:
        for part in (value or "").split(","):
            part = part.strip()
            if part and part not in keys and not part.startswith(("your", "<")):
                keys.append(part)

    add(env.get(f"{prefix}_API_KEYS"))
    add(env.get(f"{prefix}_API_KEY"))
    numbered = []
    pattern = re.compile(rf"^{re.escape(prefix)}_API_KEY_(\d+)$")
    for name, value in env.items():
        if m := pattern.match(name):
            numbered.append((int(m.group(1)), value))
    for _, value in sorted(numbered):
        add(value)
    return keys


class TokenBucket:
    """Classic token bucket that starts full; ``rate_per_minute=None`` means unlimited.

    Used for both RPM (cost 1 per request) and TPM (cost = estimated tokens) budgets.
    """

    def __init__(self, rate_per_minute: float | None, capacity: float | None = None) -> None:
        self.rate = (rate_per_minute / 60.0) if rate_per_minute else None
        self.capacity = float(capacity if capacity is not None else (rate_per_minute or 0))
        self.tokens = self.capacity
        self._last = time.monotonic()

    @property
    def unlimited(self) -> bool:
        return self.rate is None

    def _refill(self) -> None:
        if self.rate is None:
            return
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def fits(self, cost: float) -> bool:
        """Whether ``cost`` can *ever* fit (cost larger than capacity never will)."""
        return self.rate is None or cost <= self.capacity

    def wait_time(self, cost: float = 1.0) -> float:
        """Seconds until ``cost`` tokens are available (``inf`` if it can never fit)."""
        if self.rate is None:
            return 0.0
        if not self.fits(cost):
            return math.inf
        self._refill()
        return 0.0 if self.tokens >= cost else (cost - self.tokens) / self.rate

    def try_acquire(self, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens if available right now."""
        if self.wait_time(cost) > 0:
            return False
        if self.rate is not None:
            self.tokens -= cost
        return True


@dataclass
class ScopeLimits:
    """Per-scope proactive budgets."""

    rpm: TokenBucket
    tpm: TokenBucket


@dataclass
class KeySlot:
    """One API key and its health, budgets and lazily created clients."""

    key: str
    rpm: float | None = None
    tpm: float | None = None
    model_tpm: dict[str, float | None] = field(default_factory=dict)
    model_rpm: dict[str, float | None] = field(default_factory=dict)
    """Per-scope (model) RPM overrides; falls back to ``rpm``."""
    """Per-scope (model) TPM overrides; falls back to ``tpm``."""
    masked: str = ""
    disabled_reason: str | None = None
    cooldowns: dict[str, float] = field(default_factory=dict)
    limits: dict[str, ScopeLimits] = field(default_factory=dict)
    clients: dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0
    calls: int = 0
    successes: int = 0
    rate_limited: int = 0
    failures: int = 0
    last_error: str | None = None

    def __post_init__(self) -> None:
        self.masked = self.masked or mask_key(self.key)

    # ------------------------------------------------------------------ budgets
    def _limits(self, scope: str) -> ScopeLimits:
        if scope not in self.limits:
            # RPM bucket allows a small burst (~10s worth); TPM bucket allows a full minute.
            rpm = self.model_rpm[scope] if scope in self.model_rpm else self.rpm
            burst = max(1.0, math.ceil(rpm / 6)) if rpm else None
            tpm = self.model_tpm[scope] if scope in self.model_tpm else self.tpm
            self.limits[scope] = ScopeLimits(rpm=TokenBucket(rpm, burst), tpm=TokenBucket(tpm))
        return self.limits[scope]

    def cooldown_remaining(self, scope: str) -> float:
        """Seconds until this key may be used for ``scope`` again (``inf`` if disabled)."""
        if self.disabled_reason:
            return math.inf
        now = time.monotonic()
        until = max(self.cooldowns.get(scope, 0.0), self.cooldowns.get(ALL_SCOPES, 0.0))
        return max(0.0, until - now)

    def can_ever_fit(self, scope: str, tokens: float) -> bool:
        """False if the request exceeds this key's TPM capacity (it would always 413/429)."""
        return self._limits(scope).tpm.fits(tokens)

    def wait_time(self, scope: str, tokens: float = 0.0) -> float:
        """Seconds until a request of ``tokens`` could be sent with this key for ``scope``."""
        lim = self._limits(scope)
        return max(self.cooldown_remaining(scope), lim.rpm.wait_time(1), lim.tpm.wait_time(tokens))

    def try_reserve(self, scope: str, tokens: float = 0.0) -> bool:
        """Atomically reserve one request + ``tokens`` if the key is ready right now."""
        if self.wait_time(scope, tokens) > 0:
            return False
        lim = self._limits(scope)
        lim.rpm.try_acquire(1)
        lim.tpm.try_acquire(tokens)
        return True

    # ------------------------------------------------------------------ outcomes
    def record_success(self) -> None:
        self.calls += 1
        self.successes += 1
        self.consecutive_failures = 0

    def cooldown(self, scope: str, seconds: float) -> None:
        self.cooldowns[scope] = max(self.cooldowns.get(scope, 0.0), time.monotonic() + seconds)

    def status(self, scope: str | None = None) -> str:
        if self.disabled_reason:
            return "disabled"
        waits = [self.cooldown_remaining(s) for s in ([scope] if scope else list(self.cooldowns) or [ALL_SCOPES])]
        longest = max(waits, default=0.0)
        return f"cooling {longest:.0f}s" if longest > 0 else "ready"


class KeyPool:
    """All keys for one provider, with round-robin ordering and a shared failure policy."""

    def __init__(
        self,
        provider: str,
        keys: list[str],
        *,
        rpm: float | None = None,
        tpm: float | None = None,
        model_tpm: dict[str, float | None] | None = None,
        model_rpm: dict[str, float | None] | None = None,
        cooldown_base: float = 20.0,
        cooldown_max: float = 300.0,
        quota_cooldown: float = 3600.0,
        transient_cooldown: float = 5.0,
    ) -> None:
        self.provider = provider
        self.slots = [KeySlot(key=k, rpm=rpm, tpm=tpm, model_tpm=dict(model_tpm or {}),
                              model_rpm=dict(model_rpm or {})) for k in keys if k]
        self.cooldown_base = cooldown_base
        self.cooldown_max = cooldown_max
        self.quota_cooldown = quota_cooldown
        self.transient_cooldown = transient_cooldown
        self._index = 0

    def __len__(self) -> int:
        return len(self.slots)

    @property
    def available(self) -> bool:
        return any(not s.disabled_reason for s in self.slots)

    def rotation(self) -> list[KeySlot]:
        """All slots in round-robin order; the starting key advances on every call.

        This spreads concurrent requests (e.g. parallel sub-agents) evenly across keys.
        """
        n = len(self.slots)
        if not n:
            return []
        ordered = [self.slots[(self._index + i) % n] for i in range(n)]
        self._index = (self._index + 1) % n
        return ordered

    def pick(self, scope: str, *, tokens: float = 0.0, max_wait: float = 30.0) -> tuple[KeySlot | None, float]:
        """Best key for ``scope``: a ready key (reserved immediately), else the soonest within ``max_wait``.

        Returns ``(slot, wait_seconds)``. When ``wait_seconds > 0`` the caller must sleep, then call
        :meth:`KeySlot.try_reserve` (or simply proceed; budgets are best-effort).
        """
        order = self.rotation()
        for slot in order:
            if slot.try_reserve(scope, tokens):
                return slot, 0.0
        waits = sorted(((s.wait_time(scope, tokens), i, s) for i, s in enumerate(order)), key=lambda t: t[:2])
        if waits and waits[0][0] <= max_wait:
            return waits[0][2], waits[0][0]
        return None, math.inf

    def backoff(self, slot: KeySlot) -> float:
        """Exponential cooldown: base, 2×base, 4×base … capped at ``cooldown_max``."""
        return min(self.cooldown_base * (2 ** max(slot.consecutive_failures - 1, 0)), self.cooldown_max)

    def apply_failure(self, slot: KeySlot, scope: str, err: ClassifiedError) -> float:
        """Update key health after a failure; return the cooldown applied (0 if none)."""
        slot.calls += 1
        slot.failures += 1
        slot.last_error = f"{err.kind.value}: {err.message[:160]}"
        if err.kind is ErrorKind.RATE_LIMIT:
            slot.rate_limited += 1
            slot.consecutive_failures += 1
            seconds = err.retry_after if err.retry_after else self.backoff(slot)
            slot.cooldown(scope, seconds)
            return seconds
        if err.kind is ErrorKind.QUOTA_EXHAUSTED:
            slot.rate_limited += 1
            slot.cooldown(scope if err.model_scoped else ALL_SCOPES, err.retry_after or self.quota_cooldown)
            return err.retry_after or self.quota_cooldown
        if err.kind is ErrorKind.AUTH:
            slot.disabled_reason = err.message[:160] or "authentication failed"
            return math.inf
        if err.kind is ErrorKind.TRANSIENT:
            slot.cooldown(scope, self.transient_cooldown)
            return self.transient_cooldown
        return 0.0

    def snapshot(self) -> list[dict[str, Any]]:
        """Health table for the UI (keys are masked)."""
        return [
            {
                "key": s.masked,
                "status": s.status(),
                "calls": s.calls,
                "ok": s.successes,
                "rate_limited": s.rate_limited,
                "last_error": s.last_error or "",
            }
            for s in self.slots
        ]

"""Tests for key pools, token buckets, error classification and provider/chain configuration."""

from __future__ import annotations

import math

import httpx
import openai
import pytest

from agent.resilience import key_pool as kp
from agent.resilience.errors import ErrorKind, classify_error, extract_retry_after, parse_duration
from agent.resilience.key_pool import KeyPool, TokenBucket, mask_key, parse_keys
from agent.resilience.providers import default_chain, load_provider_configs, parse_chain


class Clock:
    """Controllable replacement for time.monotonic."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(kp.time, "monotonic", c)
    return c


def api_error(status: int, message: str = "error", headers: dict | None = None) -> openai.APIStatusError:
    """Build a real openai SDK exception for a given HTTP status."""
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://api.test/v1/chat"))
    cls = {429: openai.RateLimitError, 401: openai.AuthenticationError, 404: openai.NotFoundError,
           400: openai.BadRequestError, 500: openai.InternalServerError}.get(status, openai.APIStatusError)
    return cls(message, response=response, body=None)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
def test_mask_and_parse_keys() -> None:
    assert mask_key("gsk_1234567890abcdef") == "gsk_...cdef"
    assert mask_key("") == "<none>" and mask_key("short") == "***"
    env = {"GROQ_API_KEYS": "k1, k2", "GROQ_API_KEY": "k1", "GROQ_API_KEY_3": "k4", "GROQ_API_KEY_2": "k3",
           "GROQ_API_KEY_12": "k5", "GEMINI_API_KEY": "your_gemini_key_here"}
    assert parse_keys(env, "GROQ") == ["k1", "k2", "k3", "k4", "k5"]
    assert parse_keys(env, "GEMINI") == []  # placeholder values are ignored


def test_token_bucket_starts_full_and_refills(clock: Clock) -> None:
    b = TokenBucket(60, capacity=2)  # 1 token / second
    assert b.try_acquire() and b.try_acquire()
    assert not b.try_acquire()
    assert b.wait_time() == pytest.approx(1.0)
    clock.t += 1.0
    assert b.try_acquire()
    assert TokenBucket(None).try_acquire(10**9)  # unlimited
    assert TokenBucket(6000).wait_time(10_000) == math.inf  # can never fit


def test_round_robin_rotation_spreads_load() -> None:
    pool = KeyPool("groq", ["a", "b", "c"])
    starts = [pool.rotation()[0].key for _ in range(4)]
    assert starts == ["a", "b", "c", "a"]


def test_rate_limit_cooldown_is_scoped_per_model_and_backs_off(clock: Clock) -> None:
    pool = KeyPool("groq", ["a"], cooldown_base=10, cooldown_max=25)
    slot = pool.slots[0]
    err = classify_error(api_error(429, "Rate limit reached"))
    assert pool.apply_failure(slot, "model-x", err) == 10
    assert slot.cooldown_remaining("model-x") == 10
    assert slot.cooldown_remaining("model-y") == 0  # other models on the same key stay usable
    assert pool.apply_failure(slot, "model-x", err) == 20  # exponential backoff
    assert pool.apply_failure(slot, "model-x", err) == 25  # capped
    slot.record_success()
    assert slot.consecutive_failures == 0


def test_retry_after_header_wins_over_backoff(clock: Clock) -> None:
    pool = KeyPool("groq", ["a"], cooldown_base=60)
    err = classify_error(api_error(429, "slow down", headers={"retry-after": "7"}))
    assert err.retry_after == 7
    assert pool.apply_failure(pool.slots[0], "m", err) == 7


def test_auth_disables_and_quota_parks_whole_key(clock: Clock) -> None:
    pool = KeyPool("x", ["a", "b"], quota_cooldown=3600)
    a, b = pool.slots
    pool.apply_failure(a, "m", classify_error(api_error(401, "Invalid API key")))
    assert a.status() == "disabled" and a.cooldown_remaining("m") == math.inf
    pool.apply_failure(b, "m1", classify_error(api_error(402, "insufficient_quota")))
    assert b.cooldown_remaining("m2") == 3600  # quota is per key, not per model
    assert pool.available  # b is parked, not disabled


def test_pick_prefers_ready_key_then_soonest(clock: Clock) -> None:
    pool = KeyPool("tavily", ["a", "b"], rpm=6)  # burst capacity 1 per key
    first, w1 = pool.pick("s")
    second, w2 = pool.pick("s")
    assert {first.key, second.key} == {"a", "b"} and w1 == w2 == 0
    third, w3 = pool.pick("s", max_wait=30)
    assert third is not None and w3 == pytest.approx(10.0)  # 6 rpm -> next token in 10s
    assert pool.pick("s", max_wait=1) == (None, math.inf)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (api_error(429, "Rate limit reached for model"), ErrorKind.RATE_LIMIT),
        (api_error(413, "Request too large for model in organization on tokens per minute (TPM): Limit 12000"),
         ErrorKind.CONTEXT_TOO_LARGE),
        (api_error(401, "Invalid API Key"), ErrorKind.AUTH),
        (api_error(404, "The model `x` does not exist"), ErrorKind.MODEL_UNAVAILABLE),
        (api_error(400, "The model `llama2-70b` has been decommissioned"), ErrorKind.MODEL_UNAVAILABLE),
        (api_error(400, "tool_use_failed: Failed to call a function"), ErrorKind.TOOL_CALL_FAILED),
        (api_error(400, "Invalid schema for function"), ErrorKind.INCOMPATIBLE),
        (api_error(500, "internal error"), ErrorKind.TRANSIENT),
        (api_error(402, "Payment required"), ErrorKind.QUOTA_EXHAUSTED),
        (httpx.ReadTimeout("read timed out"), ErrorKind.TRANSIENT),
        (KeyError("bug in our code"), ErrorKind.FATAL),
    ],
)
def test_classify_error(exc: Exception, kind: ErrorKind) -> None:
    assert classify_error(exc).kind is kind


def test_tavily_errors_are_classified() -> None:
    from tavily.errors import ForbiddenError, InvalidAPIKeyError, UsageLimitExceededError

    assert classify_error(UsageLimitExceededError("limit")).kind is ErrorKind.RATE_LIMIT
    assert classify_error(ForbiddenError("plan limit")).kind is ErrorKind.QUOTA_EXHAUSTED
    assert classify_error(InvalidAPIKeyError("bad")).kind is ErrorKind.AUTH


def test_retry_after_parsing() -> None:
    assert parse_duration("1m30.5s") == 90.5 and parse_duration("250ms") == 0.25
    groq = api_error(429, "x", headers={"x-ratelimit-reset-tokens": "7.66s"})
    assert extract_retry_after(groq) == pytest.approx(7.66)
    gemini = api_error(429, "Quota exceeded. Please retry in 31.2s.")
    assert extract_retry_after(gemini) == pytest.approx(31.2)


# ---------------------------------------------------------------------------
# Providers & chains
# ---------------------------------------------------------------------------
def test_provider_discovery_tier_order_and_chains() -> None:
    env = {"OPENROUTER_API_KEY": "or1", "GROQ_API_KEYS": "g1,g2", "GEMINI_API_KEY": "ge1", "GROQ_RPM": "15",
           "GEMINI_TPM": "off"}
    configs = load_provider_configs(env, custom_base_url="http://localhost:11434/v1", custom_models=("qwen", ""))
    assert list(configs) == ["groq", "gemini", "openrouter", "custom"]
    assert configs["groq"].keys == ["g1", "g2"] and configs["groq"].rpm == 15
    assert configs["gemini"].tpm is None and configs["custom"].keys == ["not-needed"]

    main = [t.label for t in default_chain(configs, "main")]
    assert main[0].startswith("groq:") and main[-1] == "custom:qwen"
    fast = default_chain(configs, "fast")
    assert fast[0].provider == "groq"
    assert set(default_chain(configs, "main")) <= set(fast)  # main models are the fast chain's last resort


def test_parse_chain_validation() -> None:
    assert [t.label for t in parse_chain("groq:a, groq:a, openrouter:x/y:free")] == ["groq:a", "openrouter:x/y:free"]
    with pytest.raises(ValueError):
        parse_chain("groq")

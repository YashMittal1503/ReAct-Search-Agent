"""Resilient multi-tier, multi-key, cross-provider model router.

:class:`ModelRouter` owns one :class:`KeyPool` per provider and an ordered fallback chain of
:class:`ModelTarget`\\ s per *role* (``main`` for research and synthesis, ``fast`` for
clarify/plan/evaluate). For every LLM call it builds an **attempt plan**:

1. **Ready pass.** Walk the chain in tier order. For each target, try its keys in round-robin
   order and pick the first one whose proactive budgets (RPM/TPM token buckets) and cooldowns
   say "send now". No waiting: if Groq is saturated we go straight to the next key, then the
   next model, then the next provider.
2. **Patience pass.** Only if *nothing* is ready anywhere, wait for whichever key/target frees up
   soonest, bounded by ``max_wait``.

Failures are classified (:mod:`agent.resilience.errors`) so the right thing gets blamed: a 429
cools *that key for that model*; a 413 skips the *model* without punishing the key; an auth error
disables the key; a decommissioned model is remembered and skipped. SDK-level retries are turned
off (``max_retries=0``) so no hidden exponential backoff ever stalls the research loop: the router
fails over in milliseconds instead.

:class:`ResilientChatModel` exposes the router as an ordinary LangChain ``BaseChatModel``, so
deepagents, ``create_agent`` and ``with_structured_output`` use it transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManager, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from agent.observability.telemetry import LLMCallRecord, record_llm_call
from agent.observability.tracing import attempt_config
from agent.resilience.errors import ErrorKind, classify_error
from agent.resilience.key_pool import KeyPool, KeySlot
from agent.resilience.providers import (
    ROLES,
    ModelTarget,
    ProviderConfig,
    ProviderSpec,
    default_chain,
    load_env,
    load_provider_configs,
    parse_chain,
)

logger = logging.getLogger(__name__)

ClientFactory = Callable[[ProviderSpec, str, str], BaseChatModel]
"""``(provider_spec, api_key, model) -> chat model`` used to talk to one key/model."""

#: Completion headroom used in budget estimates when a call sets no explicit ``max_tokens``.
COMPLETION_TOKEN_ALLOWANCE = 800
#: Default completion cap per role. Sent to the provider *and* used for TPM budgeting, so the
#: estimate matches what providers like Groq count (prompt + requested completion).
ROLE_MAX_TOKENS: dict[str, int] = {"main": 2_048, "fast": 1_024}
#: Characters of the provider's error message appended to failover trace events.
DETAIL_CHARS = 150


# ---------------------------------------------------------------------------
# Events (surfaced in the UI trace via a context-local listener)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RouterEvent:
    kind: str  # "failover" | "error"
    message: str
    data: dict[str, Any] = field(default_factory=dict)


_listener: ContextVar[Callable[[RouterEvent], None] | None] = ContextVar("router_listener", default=None)


@contextmanager
def router_events(listener: Callable[[RouterEvent], None] | None) -> Iterator[None]:
    """Route :class:`RouterEvent`\\ s raised in this context (incl. child tasks) to ``listener``."""
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


#: Per-run routing affinity: role -> the target that last served that role in this context.
_affinity: ContextVar[dict[str, ModelTarget] | None] = ContextVar("router_affinity", default=None)


@contextmanager
def routing_session() -> Iterator[dict[str, ModelTarget]]:
    """Scope for **sticky routing**: calls inside it prefer the target that last succeeded.

    One graph-node run (e.g. a whole ReAct research loop, including sub-agent tasks, which inherit
    the context) is one session. Sticking to one provider keeps a conversation from mixing
    providers mid-stream, which avoids cross-provider incompatibilities and keeps prompt caches warm.
    Stickiness only *reorders* the attempt plan: if the preferred target isn't ready, the normal
    tier order applies.
    """
    state: dict[str, ModelTarget] = {}
    token = _affinity.set(state)
    try:
        yield state
    finally:
        _affinity.reset(token)


class AllProvidersExhaustedError(RuntimeError):
    """Every key of every provider in the chain failed or is unavailable."""


@dataclass
class TargetStats:
    calls: int = 0
    ok: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


def estimate_tokens(messages: list[BaseMessage], tools: Any = None, completion: int | None = None) -> int:
    """Cheap prompt-size estimate (chars/4) plus tool schemas plus the completion cap."""
    tokens = count_tokens_approximately(messages)
    if tools:
        tokens += len(json.dumps(tools, default=str)) // 4
    return tokens + (completion or COMPLETION_TOKEN_ALLOWANCE)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class ModelRouter:
    """Process-wide router; share one instance so key health persists across requests."""

    def __init__(
        self,
        providers: Mapping[str, ProviderConfig],
        chains: Mapping[str, list[ModelTarget]] | None = None,
        *,
        client_factory: ClientFactory | None = None,
        max_wait: float = 45.0,
        cooldown_base: float = 20.0,
        cooldown_max: float = 300.0,
        dead_model_ttl: float = 900.0,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> None:
        self.providers = dict(providers)
        self.pools: dict[str, KeyPool] = {
            name: KeyPool(name, cfg.keys, rpm=cfg.rpm, tpm=cfg.tpm, model_tpm=cfg.model_tpm,
                          model_rpm=cfg.model_rpm,
                          cooldown_base=cooldown_base, cooldown_max=cooldown_max)
            for name, cfg in self.providers.items()
        }
        self.chains: dict[str, list[ModelTarget]] = {}
        for role in ROLES:
            raw = list((chains or {}).get(role) or default_chain(self.providers, role))
            missing = sorted({t.provider for t in raw if t.provider not in self.pools})
            if missing:
                logger.warning("Chain '%s' references providers without keys (skipped): %s", role, missing)
            self.chains[role] = [t for t in raw if t.provider in self.pools]
        self.client_factory = client_factory or self._default_client
        self.max_wait = max_wait
        self.dead_model_ttl = dead_model_ttl
        self.temperature = temperature
        self.timeout = timeout
        self._dead: dict[ModelTarget, tuple[float, str]] = {}
        self.stats: dict[str, TargetStats] = {}
        self.failovers = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_settings(cls, settings: Any, env: Mapping[str, str] | None = None,
                      client_factory: ClientFactory | None = None) -> "ModelRouter":
        """Build from :class:`agent.config.Settings` plus provider keys found in ``.env`` / env."""
        env = env if env is not None else load_env()
        configs = load_provider_configs(
            env,
            custom_base_url=settings.llm_base_url,
            custom_models=(settings.llm_model, settings.llm_fast_model or settings.llm_model),
        )
        chains = {role: parse_chain(spec) for role, spec in
                  (("main", settings.llm_chain_main), ("fast", settings.llm_chain_fast)) if spec}
        return cls(configs, chains, client_factory=client_factory, max_wait=settings.llm_max_wait_seconds,
                   cooldown_base=settings.llm_cooldown_base_seconds, cooldown_max=settings.llm_cooldown_max_seconds,
                   temperature=settings.llm_temperature, timeout=settings.llm_timeout_seconds)

    def chat_model(self, role: str = "main", *, max_tokens: int | None = None) -> "ResilientChatModel":
        """A LangChain chat model bound to this router and a role's chain.

        Args:
            role: ``main`` (research/synthesis) or ``fast`` (control steps).
            max_tokens: Completion cap; defaults to :data:`ROLE_MAX_TOKENS` for the role.
        """
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
        return ResilientChatModel(router=self, role=role, max_tokens=max_tokens or ROLE_MAX_TOKENS.get(role))

    @property
    def configured(self) -> bool:
        return bool(self.chains.get("main"))

    def _default_client(self, spec: ProviderSpec, key: str, model: str) -> BaseChatModel:
        """OpenAI-compatible client whose payload is adapted to the provider's quirks."""
        from agent.resilience.adapters import ADAPTERS, AdaptedChatOpenAI

        return AdaptedChatOpenAI(
            model=model,
            api_key=key,
            base_url=spec.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=0,  # the router handles retries/failover; no hidden SDK backoff sleeps
            default_headers=dict(spec.headers) or None,
            payload_adapter=ADAPTERS.get(spec.name),
        )

    def _client(self, target: ModelTarget, slot: KeySlot) -> BaseChatModel:
        if target.model not in slot.clients:
            slot.clients[target.model] = self.client_factory(self.providers[target.provider].spec, slot.key, target.model)
        return slot.clients[target.model]

    # ------------------------------------------------------------------ helpers
    def _is_dead(self, target: ModelTarget) -> bool:
        entry = self._dead.get(target)
        if entry and entry[0] > time.monotonic():
            return True
        self._dead.pop(target, None)
        return False

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        logger.info("[llm-router] %s", message)
        listener = _listener.get()
        if listener is not None:
            try:
                listener(RouterEvent(kind, message, data))
            except Exception:  # noqa: BLE001 - observability must never break a call
                logger.debug("router listener failed", exc_info=True)

    def _stats(self, target: ModelTarget) -> TargetStats:
        return self.stats.setdefault(target.label, TargetStats())

    # ------------------------------------------------------------------ attempt planning
    def _attempts(self, role: str, tokens: int, skip: set[ModelTarget],
                  preferred: ModelTarget | None = None) -> Iterator[tuple[ModelTarget, KeySlot, float]]:
        """Yield ``(target, key_slot, wait_seconds)`` in the order they should be tried.

        ``skip`` is shared with the caller, which adds targets that must not be retried during
        this call (oversized request, transient provider outage, bad tool call …).
        """
        chain = self.chains.get(role) or []
        if not chain:
            raise AllProvidersExhaustedError(
                "No LLM providers configured. Set at least one of GROQ_API_KEY, CEREBRAS_API_KEY, "
                "MISTRAL_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY or LLM_BASE_URL in .env."
            )
        live = [t for t in chain if not self._is_dead(t)]
        if preferred in live:  # sticky routing: try the last successful target first
            live = [preferred, *(t for t in live if t != preferred)]
        tried: set[tuple[ModelTarget, str]] = set()

        # Context windows: skip targets that can never accept a request this large (no wasted call).
        for target in live:
            limit = self.providers[target.provider].context_for(target.model)
            if limit and tokens > limit and target not in skip:
                skip.add(target)
                self._emit("failover", f"~{tokens:,} tokens exceeds {target.label}'s {limit:,}-token request limit "
                                       f"→ skipping", model=target.label)

        # Pass 1: anything that can be sent *right now*, in tier order, keys round-robin.
        for target in live:
            for slot in self.pools[target.provider].rotation():
                if target in skip:
                    break
                if not slot.can_ever_fit(target.model, tokens):
                    skip.add(target)
                    self._emit("failover", f"~{tokens:,} tokens exceeds {target.label}'s TPM budget "
                                           f"→ routing to a higher-limit model", model=target.label)
                    break
                if slot.try_reserve(target.model, tokens):
                    tried.add((target, slot.key))
                    yield target, slot, 0.0

        # Pass 2: everything is busy; wait for whichever key/target frees up first.
        pending: list[tuple[float, int, ModelTarget, KeySlot]] = []
        for order, target in enumerate(live):
            if target in skip:
                continue
            for slot in self.pools[target.provider].slots:
                if (target, slot.key) not in tried:
                    wait = slot.wait_time(target.model, tokens)
                    if wait != math.inf:
                        pending.append((wait, order, target, slot))
        for _, _, target, slot in sorted(pending, key=lambda p: (p[0], p[1])):
            if target in skip or (target, slot.key) in tried:
                continue
            wait = slot.wait_time(target.model, tokens)
            if wait <= self.max_wait:
                tried.add((target, slot.key))
                yield target, slot, wait

    # ------------------------------------------------------------------ outcome handling
    def _on_failure(self, target: ModelTarget, slot: KeySlot, exc: Exception, skip: set[ModelTarget]) -> bool:
        """Apply the failure policy; return ``False`` if the error must be re-raised."""
        err = classify_error(exc)
        stats = self._stats(target)
        stats.calls += 1
        stats.failures += 1
        if err.kind is ErrorKind.FATAL:
            logger.error("Unrecoverable error on %s (key %s): %s", target.label, slot.masked, exc)
            return False

        self.failovers += 1
        cooldown = self.pools[target.provider].apply_failure(slot, target.model, err)
        where = f"{target.label} [key {slot.masked}]"
        if err.kind is ErrorKind.RATE_LIMIT:
            msg = f"{where} rate-limited → cooling {cooldown:.0f}s, rotating to next key/provider"
        elif err.kind is ErrorKind.QUOTA_EXHAUSTED:
            msg = f"{where} quota exhausted → key parked for {cooldown / 60:.0f} min"
        elif err.kind is ErrorKind.AUTH:
            msg = f"{where} rejected the API key → key disabled"
        elif err.kind is ErrorKind.MODEL_UNAVAILABLE:
            self._dead[target] = (time.monotonic() + self.dead_model_ttl, err.message[:200])
            skip.add(target)
            msg = f"{target.label} unavailable ({err.status or 'model error'}) → skipped for {self.dead_model_ttl / 60:.0f} min"
        elif err.kind is ErrorKind.CONTEXT_TOO_LARGE:
            skip.add(target)
            msg = f"{target.label} can't take a request this large → trying a higher-limit model"
        elif err.kind is ErrorKind.TOOL_CALL_FAILED:
            skip.add(target)
            msg = f"{target.label} produced an invalid tool call → retrying on another model"
        elif err.kind is ErrorKind.TRANSIENT and "timeout" in type(exc).__name__.lower():
            # A hung request is usually a one-off (measured: Gemini flash-lite answers the same
            # request in ~4s on retry), so only this key cools down; the model's other keys stay
            # in play for this request. Each (model, key) pair is still tried at most once.
            msg = f"{where} timed out → trying the next key"
        elif err.kind is ErrorKind.TRANSIENT:
            skip.add(target)
            msg = f"{target.label} transient error ({err.status or type(exc).__name__}) → failing over"
        else:  # INCOMPATIBLE
            skip.add(target)
            msg = f"{target.label} rejected the request ({err.status}) → trying another provider"
        if err.detail:
            msg += f" — {err.detail[:DETAIL_CHARS]}{'…' if len(err.detail) > DETAIL_CHARS else ''}"
        self._emit("failover", msg, provider=target.provider, model=target.model, key=slot.masked,
                   error=err.kind.value, detail=err.detail[:500])
        return True

    def _finish(self, target: ModelTarget, slot: KeySlot, message: AIMessage, started: float,
                first: ModelTarget | None) -> ChatResult:
        slot.record_success()
        stats = self._stats(target)
        stats.calls += 1
        stats.ok += 1
        stats.latency_s += time.monotonic() - started
        usage = getattr(message, "usage_metadata", None) or {}
        stats.input_tokens += int(usage.get("input_tokens", 0) or 0)
        stats.output_tokens += int(usage.get("output_tokens", 0) or 0)
        message.response_metadata = {**(message.response_metadata or {}),
                                     "router": {"provider": target.provider, "model": target.model, "key": slot.masked}}
        if first is not None and target != first:
            self._emit("failover", f"Served by {target.label} after failover", model=target.label)
        return ChatResult(generations=[ChatGeneration(message=message)],
                          llm_output={"model_name": target.model, "provider": target.provider})

    @staticmethod
    def _record(role: str, target: ModelTarget, slot: KeySlot, attempt: int, started: float, tokens: int, *,
                message: AIMessage | None = None, error: BaseException | None = None) -> None:
        """Emit one :class:`LLMCallRecord` for this attempt (logs + current run's collector)."""
        usage = (getattr(message, "usage_metadata", None) or {}) if message is not None else {}
        record_llm_call(LLMCallRecord(
            role=role, provider=target.provider, model=target.model, key=slot.masked,
            outcome="ok" if error is None else classify_error(error).kind.value,
            latency_ms=round((time.monotonic() - started) * 1000, 1), attempt=attempt, est_tokens=tokens,
            input_tokens=int(usage.get("input_tokens", 0) or 0), output_tokens=int(usage.get("output_tokens", 0) or 0),
            error="" if error is None else f"{type(error).__name__}: {str(error)[:300]}",
        ))

    def _exhausted(self, role: str, last: BaseException | None) -> AllProvidersExhaustedError:
        detail = f"{type(last).__name__}: {str(last)[:300]}" if last else "all keys cooling down or disabled"
        self._emit("error", f"All providers exhausted for '{role}' chain ({detail})")
        return AllProvidersExhaustedError(f"All LLM providers/keys in the '{role}' chain failed. Last error: {detail}")

    # ------------------------------------------------------------------ execution
    async def agenerate(self, role: str, messages: list[BaseMessage], stop: list[str] | None = None,
                        *, callbacks: Any = None, **kwargs: Any) -> ChatResult:
        """Async call with key rotation and cross-provider failover."""
        kwargs.pop("ls_structured_output_format", None)
        tokens = estimate_tokens(messages, kwargs.get("tools"),
                                 kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"))
        skip: set[ModelTarget] = set()
        first: ModelTarget | None = None
        last: Exception | None = None
        attempt = 0
        affinity = _affinity.get()
        for target, slot, wait in self._attempts(role, tokens, skip, affinity.get(role) if affinity is not None else None):
            if wait > 0:
                self._emit("failover", f"All providers at capacity → waiting {wait:.1f}s for {target.label}")
                await asyncio.sleep(wait)
                slot.try_reserve(target.model, tokens)
            first = first or target
            attempt += 1
            started = time.monotonic()
            try:
                message = await self._client(target, slot).ainvoke(
                    messages, config=attempt_config(role=role, provider=target.provider, model=target.model,
                                                    key=slot.masked, attempt=attempt, callbacks=callbacks),
                    stop=stop, **kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                self._record(role, target, slot, attempt, started, tokens, error=exc)
                if not self._on_failure(target, slot, exc, skip):
                    raise
                continue
            if affinity is not None:
                affinity[role] = target
            self._record(role, target, slot, attempt, started, tokens, message=message)
            return self._finish(target, slot, message, started, first)
        raise self._exhausted(role, last) from last

    def generate(self, role: str, messages: list[BaseMessage], stop: list[str] | None = None,
                 *, callbacks: Any = None, **kwargs: Any) -> ChatResult:
        """Synchronous twin of :meth:`agenerate`."""
        kwargs.pop("ls_structured_output_format", None)
        tokens = estimate_tokens(messages, kwargs.get("tools"),
                                 kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"))
        skip: set[ModelTarget] = set()
        first: ModelTarget | None = None
        last: Exception | None = None
        attempt = 0
        affinity = _affinity.get()
        for target, slot, wait in self._attempts(role, tokens, skip, affinity.get(role) if affinity is not None else None):
            if wait > 0:
                self._emit("failover", f"All providers at capacity → waiting {wait:.1f}s for {target.label}")
                time.sleep(wait)
                slot.try_reserve(target.model, tokens)
            first = first or target
            attempt += 1
            started = time.monotonic()
            try:
                message = self._client(target, slot).invoke(
                    messages, config=attempt_config(role=role, provider=target.provider, model=target.model,
                                                    key=slot.masked, attempt=attempt, callbacks=callbacks),
                    stop=stop, **kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                self._record(role, target, slot, attempt, started, tokens, error=exc)
                if not self._on_failure(target, slot, exc, skip):
                    raise
                continue
            if affinity is not None:
                affinity[role] = target
            self._record(role, target, slot, attempt, started, tokens, message=message)
            return self._finish(target, slot, message, started, first)
        raise self._exhausted(role, last) from last

    # ------------------------------------------------------------------ observability
    def health(self) -> dict[str, Any]:
        """Snapshot for dashboards: chains, per-key status, dead models, per-model stats."""
        return {
            "chains": {role: [{"target": t.label, "dead": self._is_dead(t)} for t in chain]
                       for role, chain in self.chains.items()},
            "providers": [
                {"name": name, "label": self.providers[name].spec.label, "rpm": self.providers[name].rpm,
                 "tpm": self.providers[name].tpm, "model_tpm": dict(self.providers[name].model_tpm),
                 "model_rpm": dict(self.providers[name].model_rpm),
                 "context": self.providers[name].context, "keys": pool.snapshot()}
                for name, pool in self.pools.items()
            ],
            "dead_models": {t.label: reason for t, (until, reason) in self._dead.items() if until > time.monotonic()},
            "stats": {label: vars(s) for label, s in self.stats.items()},
            "failovers": self.failovers,
        }


# ---------------------------------------------------------------------------
# LangChain facade
# ---------------------------------------------------------------------------
def _child_callbacks(run_manager: Any) -> CallbackManager | None:
    """Callback manager for runs nested under ``run_manager``'s LLM run.

    LLM run managers have no ``get_child()`` (only chain run managers do), so this mirrors its
    implementation: inherit handlers, tags and metadata, and set the parent run id. Router attempts
    then show up as children of the ``ResilientChatModel`` run in LangSmith.
    """
    if run_manager is None:
        return None
    manager = CallbackManager(handlers=[], parent_run_id=run_manager.run_id)
    manager.set_handlers(run_manager.inheritable_handlers)
    manager.add_tags(run_manager.inheritable_tags)
    manager.add_metadata(run_manager.inheritable_metadata)
    return manager


def _normalize_tool_choice(tool_choice: Any, formatted: list[dict[str, Any]]) -> Any:
    """Translate LangChain tool_choice conventions into the OpenAI wire format."""
    if tool_choice is None or tool_choice is False:
        return None
    if tool_choice is True:
        return {"type": "function", "function": {"name": formatted[0]["function"]["name"]}}
    if tool_choice == "any":
        return "required"
    if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
        return {"type": "function", "function": {"name": tool_choice}}
    return tool_choice


class ResilientChatModel(BaseChatModel):
    """``BaseChatModel`` facade over :class:`ModelRouter` for one role (``main`` / ``fast``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    router: Any = Field(exclude=True)
    role: str = "main"
    model_name: str = "resilient-router"
    max_tokens: int | None = None
    """Completion cap forwarded to every provider call (unless the caller sets one)."""

    def _with_cap(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.max_tokens and "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    @property
    def _llm_type(self) -> str:
        return "resilient-router"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"role": self.role, "chain": [t.label for t in self.router.chains.get(self.role, [])]}

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, strict: bool | None = None,
                   parallel_tool_calls: bool | None = None, **kwargs: Any) -> Any:
        """Bind tools in OpenAI format (understood by every provider in the registry).

        ``parallel_tool_calls`` is accepted but not forwarded: several providers reject it.
        """
        formatted = [convert_to_openai_tool(t, strict=strict) for t in tools]
        choice = _normalize_tool_choice(tool_choice, formatted)
        if choice is not None:
            kwargs["tool_choice"] = choice
        return self.bind(tools=formatted, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return await self.router.agenerate(self.role, messages, stop=stop,
                                           callbacks=_child_callbacks(run_manager),
                                           **self._with_cap(kwargs))

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return self.router.generate(self.role, messages, stop=stop,
                                    callbacks=_child_callbacks(run_manager),
                                    **self._with_cap(kwargs))

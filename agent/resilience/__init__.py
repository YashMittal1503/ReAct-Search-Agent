"""Resilient multi-tier, cross-provider orchestration with multi-key pools.

* :mod:`~agent.resilience.key_pool`: per-provider key pools (round-robin, RPM/TPM budgets, cooldowns)
* :mod:`~agent.resilience.errors`: provider-agnostic error classification + Retry-After parsing
* :mod:`~agent.resilience.providers`: provider registry, tier order, model chains
* :mod:`~agent.resilience.router`: :class:`ModelRouter` + LangChain :class:`ResilientChatModel`
"""

from agent.resilience.errors import ClassifiedError, ErrorKind, classify_error
from agent.resilience.key_pool import KeyPool, KeySlot, TokenBucket, mask_key, parse_keys
from agent.resilience.providers import PROVIDERS, ModelTarget, ProviderConfig, load_env, parse_chain
from agent.resilience.router import (
    AllProvidersExhaustedError,
    ModelRouter,
    ResilientChatModel,
    RouterEvent,
    router_events,
)

__all__ = [
    "PROVIDERS",
    "AllProvidersExhaustedError",
    "ClassifiedError",
    "ErrorKind",
    "KeyPool",
    "KeySlot",
    "ModelRouter",
    "ModelTarget",
    "ProviderConfig",
    "ResilientChatModel",
    "RouterEvent",
    "TokenBucket",
    "classify_error",
    "load_env",
    "mask_key",
    "parse_chain",
    "parse_keys",
    "router_events",
]

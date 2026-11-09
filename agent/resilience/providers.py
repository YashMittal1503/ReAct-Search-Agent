"""Provider registry, tier ordering and model-chain configuration.

Every provider here speaks the OpenAI Chat Completions protocol, so one client class
(``langchain_openai.ChatOpenAI``) with a different ``base_url`` covers all of them.

Tier order (when no explicit chain is configured) follows "fastest first, most generous next,
universal fail-safe last":

    1. Groq        - LPU inference, lowest latency
    2. Cerebras    - wafer-scale inference, very fast, generous free tier
    3. Mistral     - high TPM limits
    4. Gemini      - large free quotas (OpenAI-compatible endpoint)
    5. OpenRouter  - universal fallback across hundreds of models
    6. OpenAI      - paid, highly reliable
    7. custom      - any OpenAI-compatible endpoint from LLM_BASE_URL (vLLM, Ollama, LM Studio …)

Model ids below are *defaults*. They're easy to override per role with ``LLM_CHAIN_MAIN`` /
``LLM_CHAIN_FAST``, and the router automatically skips any that a provider reports as
unknown or decommissioned, so stale defaults degrade gracefully instead of breaking a run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from agent.resilience.key_pool import parse_keys

ROLES = ("main", "fast")


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of an OpenAI-compatible provider."""

    name: str
    label: str
    base_url: str
    env_prefix: str
    main_models: tuple[str, ...]
    fast_models: tuple[str, ...]
    default_rpm: float | None = None
    default_tpm: float | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {
    spec.name: spec
    for spec in (
        ProviderSpec("groq", "Groq", "https://api.groq.com/openai/v1", "GROQ",
                     main_models=("openai/gpt-oss-120b", "llama-3.3-70b-versatile"),
                     fast_models=("llama-3.1-8b-instant", "qwen/qwen3-32b"),
                     default_rpm=30),
        ProviderSpec("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "CEREBRAS",
                     main_models=("gpt-oss-120b",), fast_models=("llama3.1-8b",), default_rpm=30),
        ProviderSpec("mistral", "Mistral", "https://api.mistral.ai/v1", "MISTRAL",
                     main_models=("mistral-small-latest",), fast_models=("ministral-8b-latest",), default_rpm=60),
        ProviderSpec("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI",
                     main_models=("gemini-flash-latest",), fast_models=("gemini-flash-lite-latest",), default_rpm=10),
        ProviderSpec("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER",
                     main_models=("meta-llama/llama-3.3-70b-instruct",),
                     fast_models=("meta-llama/llama-3.1-8b-instruct",), default_rpm=20,
                     headers={"HTTP-Referer": "https://github.com/YashMittal1503", "X-Title": "Deep Research Agent"}),
        ProviderSpec("openai", "OpenAI", "https://api.openai.com/v1", "OPENAI",
                     main_models=("gpt-4o-mini",), fast_models=("gpt-4o-mini",)),
    )
}
PRIORITY: tuple[str, ...] = (*PROVIDERS, "custom")


@dataclass(frozen=True)
class ModelTarget:
    """One step of a fallback chain: a provider and a model id."""

    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class ProviderConfig:
    """Runtime configuration for one provider: spec + keys + proactive limits."""

    spec: ProviderSpec
    keys: list[str]
    rpm: float | None
    tpm: float | None


def load_env(env_file: str | Path = ".env") -> dict[str, str]:
    """``.env`` values overlaid with the real environment (real env wins)."""
    values = {k: v for k, v in dotenv_values(env_file).items() if v is not None} if Path(env_file).exists() else {}
    return {**values, **os.environ}


def _limit(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    """Read ``<PREFIX>_RPM`` / ``_TPM``; ``0`` or ``off`` disables the proactive budget."""
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"0", "off", "none", "unlimited"}:
        return None
    return float(raw)


def load_provider_configs(
    env: Mapping[str, str],
    *,
    custom_base_url: str | None = None,
    custom_models: tuple[str, str] | None = None,
) -> dict[str, ProviderConfig]:
    """Discover every provider that has at least one key, in tier order."""
    configs: dict[str, ProviderConfig] = {}
    for name, spec in PROVIDERS.items():
        keys = parse_keys(env, spec.env_prefix)
        if keys:
            configs[name] = ProviderConfig(
                spec, keys,
                rpm=_limit(env, f"{spec.env_prefix}_RPM", spec.default_rpm),
                tpm=_limit(env, f"{spec.env_prefix}_TPM", spec.default_tpm),
            )
    if custom_base_url:
        keys = parse_keys(env, "LLM") or ["not-needed"]  # local servers (Ollama, vLLM) accept any key
        main, fast = custom_models or ("", "")
        spec = ProviderSpec("custom", "Custom endpoint", custom_base_url, "LLM",
                            main_models=(main,) if main else (), fast_models=(fast or main,) if (fast or main) else ())
        configs["custom"] = ProviderConfig(spec, keys, rpm=_limit(env, "LLM_RPM", None), tpm=_limit(env, "LLM_TPM", None))
    return configs


def parse_chain(spec: str) -> list[ModelTarget]:
    """Parse ``"groq:openai/gpt-oss-120b, gemini:gemini-flash-latest"`` into targets.

    Only the first ``:`` separates provider from model, so OpenRouter ids like
    ``meta-llama/llama-3.3-70b-instruct:free`` work.

    >>> [t.label for t in parse_chain("groq:a, openrouter:x/y:free")]
    ['groq:a', 'openrouter:x/y:free']
    """
    targets: list[ModelTarget] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        provider, sep, model = item.partition(":")
        if not sep or not model:
            raise ValueError(f"Invalid chain entry {item!r}; expected 'provider:model'")
        target = ModelTarget(provider.strip().lower(), model.strip())
        if target not in targets:
            targets.append(target)
    return targets


def default_chain(configs: Mapping[str, ProviderConfig], role: str) -> list[ModelTarget]:
    """Tier-ordered chain from every configured provider's default models.

    The fast chain ends with the main models as a last resort: a slower answer beats no answer.
    """
    ordered = [configs[p] for p in PRIORITY if p in configs]
    main = [ModelTarget(c.spec.name, m) for c in ordered for m in c.spec.main_models]
    if role == "main":
        return main
    fast = [ModelTarget(c.spec.name, m) for c in ordered for m in c.spec.fast_models]
    return fast + [t for t in main if t not in fast]

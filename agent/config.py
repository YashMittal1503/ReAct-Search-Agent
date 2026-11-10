"""Typed runtime configuration loaded from environment variables / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables in one place. Every field maps to an upper-case env var of the same name."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM routing (multi-tier, multi-key, cross-provider) -----------------------------------
    # Provider keys are discovered from the environment by agent.resilience.providers:
    #   GROQ_API_KEY / GROQ_API_KEY_2 … / GROQ_API_KEYS=a,b,c (same for CEREBRAS, MISTRAL, GEMINI,
    #   OPENROUTER, OPENAI). Optional proactive budgets: <PROVIDER>_RPM, <PROVIDER>_TPM.
    llm_chain_main: str | None = Field(
        default=None, description="Explicit main chain, e.g. 'groq:openai/gpt-oss-120b,gemini:gemini-flash-latest'.")
    llm_chain_fast: str | None = Field(default=None, description="Explicit fast chain (clarify/plan/evaluate).")
    llm_max_wait_seconds: float = Field(default=45.0, ge=0, description="Max wait when every key is saturated.")
    llm_cooldown_base_seconds: float = Field(default=20.0, gt=0, description="First 429 cooldown w/o Retry-After.")
    llm_cooldown_max_seconds: float = Field(default=300.0, gt=0)
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 120.0
    llm_structured_method: Literal["function_calling", "json_schema", "json_mode"] = "function_calling"

    # Optional extra tier: any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, Together …).
    # Keys: LLM_API_KEY / LLM_API_KEY_2 … / LLM_API_KEYS.
    llm_base_url: str | None = None
    llm_model: str = ""
    llm_fast_model: str | None = None

    # --- Search -------------------------------------------------------------------------------
    # Keys: TAVILY_API_KEY / TAVILY_API_KEY_2 … / TAVILY_API_KEYS (rotated like LLM keys).
    tavily_rpm: float | None = Field(default=100, description="Proactive per-key Tavily budget (None = off).")
    tavily_search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"
    search_results_per_query: int = Field(default=5, ge=1, le=20)

    # --- Agent behaviour ----------------------------------------------------------------------
    research_engine: Literal["deep", "react"] = "deep"
    max_iterations: int = Field(default=4, ge=1, le=8)
    max_clarification_rounds: int = Field(default=2, ge=0, le=5)
    min_query_words: int = Field(default=4, ge=1, description="Shorter prompts trigger clarification.")
    max_scrapes_per_iteration: int = Field(default=4, ge=0, le=20)
    scrape_relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    scrape_timeout_seconds: float = 15.0
    scrape_max_chars: int = 12_000
    scrape_concurrency: int = 4
    research_recursion_limit: int = 60
    max_report_sources: int = 12

    # --- Observability ------------------------------------------------------------------------
    log_dir: str = Field(default="logs", description="app.jsonl (rotating) and runs/ live here.")
    log_level: str = "INFO"
    run_logs: bool = Field(default=True, description="Persist every run to logs/runs/<date>/<run_id>.jsonl.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()

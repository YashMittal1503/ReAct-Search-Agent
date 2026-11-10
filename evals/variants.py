"""Named pipeline variants compared by the harness (settings overrides on top of ``.env``).

Add an entry here to evaluate a change. The ``experiment/rag`` branch, for example, adds
``"rag": {"synthesis_mode": "rag"}``. Keep ``baseline`` equal to the production defaults.
"""

from __future__ import annotations

from typing import Any

from agent.config import Settings, get_settings

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},                                   # production defaults (two-step synthesis)
    "single_step": {"synthesis_mode": "single_step"},  # legacy synthesis from raw excerpts (#3 A/B)
    "react_engine": {"research_engine": "react"},      # lighter ReAct engine (meaningful in e2e/snapshot)
}


def settings_for(variant: str, base: Settings | None = None) -> Settings:
    if variant not in VARIANTS:
        raise SystemExit(f"Unknown variant {variant!r}. Known: {', '.join(VARIANTS)}")
    return (base or get_settings()).model_copy(update=VARIANTS[variant])

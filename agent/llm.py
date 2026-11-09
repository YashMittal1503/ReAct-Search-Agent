"""LLM access layer.

Chat models come from :class:`agent.resilience.ModelRouter`, which exposes every configured
provider/key/model as a single LangChain ``BaseChatModel`` per role (``main`` / ``fast``) with
multi-key rotation and cross-provider failover. This module adds the structured-output helper
used by the control nodes.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from agent.config import Settings
from agent.resilience.router import ModelRouter, ResilientChatModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def build_router(settings: Settings) -> ModelRouter:
    """Create the process-wide router from settings and the provider keys in ``.env``."""
    return ModelRouter.from_settings(settings)


def build_chat_model(router: ModelRouter, *, fast: bool = False) -> ResilientChatModel:
    """Chat model for the ``fast`` (control steps) or ``main`` (research/synthesis) chain."""
    return router.chat_model("fast" if fast else "main")


async def structured_call(
    model: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    method: str = "function_calling",
    attempts: int = 2,
) -> T:
    """Invoke ``model`` and parse its output into ``schema`` (Pydantic v2).

    Open-weight models occasionally emit malformed tool arguments. On a parse/validation failure
    the error is fed back to the model once ("self-correction") before giving up. (Provider-level
    failures such as 429s are handled underneath by the router.)

    Raises:
        ValueError: if no valid object could be produced after ``attempts`` tries.
    """
    runnable = model.with_structured_output(schema, method=method)
    convo = list(messages)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = await runnable.ainvoke(convo)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            raise ValueError(f"model returned no structured output ({type(result).__name__})")
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            last_error = exc
            logger.warning("Structured output for %s failed (attempt %d): %s", schema.__name__, attempt + 1, exc)
            convo = [*messages, HumanMessage(
                content=f"Your previous answer was invalid: {str(exc)[:500]}\n"
                f"Respond again, strictly matching the {schema.__name__} schema."
            )]
    raise ValueError(f"Could not obtain valid {schema.__name__}: {last_error}")

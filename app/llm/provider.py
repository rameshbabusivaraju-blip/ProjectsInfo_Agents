"""LLM provider layer — ADR-004.

Exposes get_llm(tier) so agent nodes never hard-code a specific
model. The provider for each tier comes from .env, so switching
OpenAI / Anthropic / Gemini is a config change, not a code change.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# tier -> provider -> model name. Candidates fixed by ADR-004.
_MODELS = {
    "fast": {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
    },
    "strong": {
        "anthropic": "claude-sonnet-5",  # ADR-004 says claude-sonnet-4-5; that name's stale
        "openai": "gpt-4o",
        "gemini": "gemini-2.5-pro",
    },
}


def get_llm(tier: str):
    """Return a chat model for the given tier.

    tier: "fast" (classification, routing, SQL generation) or
    "strong" (synthesis). Reads {TIER}_LLM_PROVIDER from .env to
    pick the provider, e.g. FAST_LLM_PROVIDER=anthropic.
    """
    provider = os.environ[f"{tier.upper()}_LLM_PROVIDER"]

    if provider not in _MODELS[tier]:
        raise ValueError(f"Unknown provider '{provider}' for tier '{tier}'")
    model = _MODELS[tier][provider]

    if provider == "anthropic":
        kwargs = {"model": model, "max_tokens": 1024}  # required on Anthropic
        if model.startswith("claude-haiku"):
            kwargs["temperature"] = 0  # rejected on Sonnet 5
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0)

    return ChatGoogleGenerativeAI(model=model, temperature=0)
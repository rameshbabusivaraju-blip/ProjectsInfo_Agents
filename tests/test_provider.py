"""Tests for the LLM provider layer — ADR-004 tier/provider selection.

Fully offline: the LangChain chat-model classes are swapped for a small
recorder so no API key or network call is needed to check that get_llm()
builds the right model name and kwargs for each tier/provider.
"""

import pytest

from app.llm import provider


class _RecordingChatModel:
    """Stands in for a LangChain chat-model class; records the kwargs it was built with."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_fast_anthropic_uses_haiku_with_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """fast/anthropic must build Haiku 4.5 with temperature=0 and max_tokens set."""
    monkeypatch.setenv("FAST_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(provider, "ChatAnthropic", _RecordingChatModel)

    model = provider.get_llm("fast")

    assert isinstance(model, _RecordingChatModel)
    assert model.kwargs["model"] == "claude-haiku-4-5-20251001"
    assert model.kwargs["temperature"] == 0
    assert model.kwargs["max_tokens"] == 1024


def test_strong_anthropic_omits_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sonnet 5 rejects temperature, so strong/anthropic must not pass it."""
    monkeypatch.setenv("STRONG_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(provider, "ChatAnthropic", _RecordingChatModel)

    model = provider.get_llm("strong")

    assert isinstance(model, _RecordingChatModel)
    assert model.kwargs["model"] == "claude-sonnet-5"
    assert "temperature" not in model.kwargs


def test_fast_openai_uses_mini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """fast/openai must pick gpt-4o-mini."""
    monkeypatch.setenv("FAST_LLM_PROVIDER", "openai")
    monkeypatch.setattr(provider, "ChatOpenAI", _RecordingChatModel)

    model = provider.get_llm("fast")

    assert isinstance(model, _RecordingChatModel)
    assert model.kwargs["model"] == "gpt-4o-mini"
    assert model.kwargs["temperature"] == 0


def test_strong_gemini_uses_pro_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """strong/gemini must pick gemini-2.5-pro."""
    monkeypatch.setenv("STRONG_LLM_PROVIDER", "gemini")
    monkeypatch.setattr(provider, "ChatGoogleGenerativeAI", _RecordingChatModel)

    model = provider.get_llm("strong")

    assert isinstance(model, _RecordingChatModel)
    assert model.kwargs["model"] == "gemini-2.5-pro"


def test_unknown_provider_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognized provider name must fail clearly, not silently pick one."""
    monkeypatch.setenv("FAST_LLM_PROVIDER", "made-up-vendor")

    with pytest.raises(ValueError, match="Unknown provider"):
        provider.get_llm("fast")

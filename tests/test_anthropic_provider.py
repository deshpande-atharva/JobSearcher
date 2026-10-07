"""Anthropic Claude provider: initialization, selection, failure semantics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm.base import LLMProvider, LLMSettings, NullLLMProvider, build_llm_provider
from src.models.config import AppConfig, Secrets, load_config

ROOT = Path(__file__).resolve().parents[1]


class Echo(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides) -> LLMSettings:
    data = dict(
        provider="anthropic",
        model="claude-sonnet-5",
        timeout_seconds=2,
        max_calls_per_run=20,
        circuit_breaker_failures=5,
        circuit_reset_seconds=0,
        retry_attempts=2,
        retry_backoff_seconds=0,
    )
    data.update(overrides)
    return LLMSettings(**data)


def _mock_anthropic_module():
    """Return a mock that looks like the anthropic package."""
    mod = MagicMock()
    mod.AsyncAnthropic = MagicMock()
    return mod


def _mock_response(text: str):
    """Build a mock Anthropic message response."""
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _mock_error(status: int, message: str):
    """Build a mock Anthropic API error."""

    class APIStatusError(Exception):
        def __init__(self, msg, status_code):
            super().__init__(msg)
            self.status_code = status_code

    return APIStatusError(message, status_code=status)


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


def test_anthropic_provider_initializes():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(_settings(), api_key="sk-ant-test-key")
        assert provider.name == "anthropic"
        assert provider.available is True


def test_anthropic_provider_rejects_empty_key():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        with pytest.raises(ValueError, match="API key is required"):
            AnthropicProvider(_settings(), api_key="")


# ---------------------------------------------------------------------------
# 2. Model configuration
# ---------------------------------------------------------------------------


def test_model_comes_from_settings():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            _settings(model="claude-opus-4"), api_key="sk-ant-test"
        )
        assert provider._model == "claude-opus-4"


# ---------------------------------------------------------------------------
# 3. API key from environment
# ---------------------------------------------------------------------------


def test_build_provider_uses_anthropic_api_key_from_secrets():
    config = load_config(ROOT / "config", send_email=False, env={
        "ANTHROPIC_API_KEY": "sk-ant-test-key-123",
        "LLM_PROVIDER": "anthropic",
    })
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        provider = build_llm_provider(config)
        assert provider.name == "anthropic"


def test_missing_anthropic_key_falls_back_to_null():
    config = load_config(ROOT / "config", send_email=False, env={
        "LLM_PROVIDER": "anthropic",
    })
    provider = build_llm_provider(config)
    assert isinstance(provider, NullLLMProvider)


# ---------------------------------------------------------------------------
# 4. Successful structured response
# ---------------------------------------------------------------------------


async def test_successful_structured_response():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(_settings(), api_key="sk-ant-test")
        provider._client.messages.create = AsyncMock(
            return_value=_mock_response('{"value": "hello"}')
        )
        result = await provider.structured(
            prompt="test", response_model=Echo, purpose="test"
        )
        assert result is not None
        assert result.value == "hello"
        assert provider.stats.successes == 1
        assert provider.circuit_state == "CLOSED"


# ---------------------------------------------------------------------------
# 5. Malformed response
# ---------------------------------------------------------------------------


async def test_malformed_response_returns_none():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(_settings(), api_key="sk-ant-test")
        provider._client.messages.create = AsyncMock(
            return_value=_mock_response("not json at all")
        )
        result = await provider.structured(
            prompt="test", response_model=Echo, purpose="test"
        )
        assert result is None
        assert provider.stats.fallbacks == 1
        assert provider.stats.failures_by_category.get("schema") == 1


# ---------------------------------------------------------------------------
# 6. Timeout
# ---------------------------------------------------------------------------


async def test_timeout_returns_none():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            _settings(retry_attempts=0), api_key="sk-ant-test"
        )

        async def slow(**kwargs):
            await asyncio.sleep(10)

        provider._client.messages.create = slow
        result = await provider.structured(
            prompt="test", response_model=Echo, purpose="test"
        )
        assert result is None
        assert provider.stats.failures_by_category.get("timeout") == 1


# ---------------------------------------------------------------------------
# 7. 429 rate limit
# ---------------------------------------------------------------------------


async def test_429_opens_circuit_after_two():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(_settings(), api_key="sk-ant-test")
        err = _mock_error(429, "429 rate_limit_error")
        provider._client.messages.create = AsyncMock(side_effect=err)
        result = await provider.structured(
            prompt="test", response_model=Echo, purpose="test"
        )
        assert result is None
        assert provider.stats.failures_by_category.get("429") == 2
        assert provider.circuit_state == "OPEN"


# ---------------------------------------------------------------------------
# 8. Authentication failure
# ---------------------------------------------------------------------------


async def test_auth_failure_opens_circuit():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            _settings(retry_attempts=0), api_key="sk-ant-test"
        )
        err = _mock_error(401, "401 authentication_error: invalid api key")
        provider._client.messages.create = AsyncMock(side_effect=err)
        result = await provider.structured(
            prompt="test", response_model=Echo, purpose="test"
        )
        assert result is None
        assert provider.circuit_state == "OPEN"


# ---------------------------------------------------------------------------
# 9. Deterministic fallback after failure
# ---------------------------------------------------------------------------


async def test_fallback_after_failure():
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            _settings(retry_attempts=0), api_key="sk-ant-test"
        )
        err = _mock_error(500, "500 internal_server_error")
        provider._client.messages.create = AsyncMock(side_effect=err)
        result = await provider.structured(
            prompt="test", response_model=Echo, purpose="test"
        )
        assert result is None
        assert provider.stats.fallbacks == 1
        # Provider is still usable (not a permanent error)
        provider._client.messages.create = AsyncMock(
            return_value=_mock_response('{"value": "recovered"}')
        )
        result2 = await provider.structured(
            prompt="test2", response_model=Echo, purpose="test"
        )
        assert result2 is not None
        assert result2.value == "recovered"


# ---------------------------------------------------------------------------
# 10. Provider selection
# ---------------------------------------------------------------------------


def test_provider_selection_anthropic():
    config = load_config(ROOT / "config", send_email=False, env={
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "LLM_PROVIDER": "anthropic",
    })
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        provider = build_llm_provider(config)
        assert provider.name == "anthropic"


def test_provider_selection_gemini():
    config = load_config(ROOT / "config", send_email=False, env={
        "GEMINI_API_KEY": "AIza-test",
        "LLM_PROVIDER": "gemini",
    })
    with patch("src.llm.gemini.GeminiProvider") as mock_cls:
        mock_cls.return_value = MagicMock(spec=LLMProvider)
        mock_cls.return_value.name = "gemini"
        provider = build_llm_provider(config)
        assert provider.name == "gemini"


def test_provider_selection_none():
    config = load_config(ROOT / "config", send_email=False, env={
        "LLM_PROVIDER": "none",
    })
    provider = build_llm_provider(config)
    assert isinstance(provider, NullLLMProvider)


def test_provider_selection_unknown():
    """Unknown providers are rejected at config validation."""
    from src.models.config import ConfigError

    with pytest.raises(ConfigError, match="literal_error"):
        load_config(ROOT / "config", send_email=False, env={
            "LLM_PROVIDER": "openai",
        })


# ---------------------------------------------------------------------------
# 11. Existing intelligence behaviour unchanged
# ---------------------------------------------------------------------------


async def test_intelligence_layer_uses_provider_from_resources(tmp_config):
    """The role agent reads llm from state.resources, not from a global."""
    from src.agents.role_agent import run_role_classification
    from src.llm.base import NullLLMProvider
    from src.models.state import PipelineState
    from tests.conftest import make_job

    provider = NullLLMProvider()
    state = PipelineState(
        config=tmp_config,
        jobs=[
            make_job(job_title="Software Engineer", job_id="1"),
            make_job(job_title="Marketing Manager", job_id="2"),
        ],
        resources={"llm": provider},
    )
    await run_role_classification(state)
    # Software Engineer passes deterministically; Marketing Manager is rejected.
    assert {job.job_id for job in state.jobs} == {"1"}


# ---------------------------------------------------------------------------
# 12. No Gemini call when provider is Anthropic
# ---------------------------------------------------------------------------


def test_no_gemini_import_when_anthropic():
    """When LLM_PROVIDER=anthropic, build_llm_provider must not import Gemini."""
    config = load_config(ROOT / "config", send_email=False, env={
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "LLM_PROVIDER": "anthropic",
    })
    with (
        patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}),
        patch("src.llm.base.log") as mock_log,
    ):
        provider = build_llm_provider(config)
        assert provider.name == "anthropic"
    # Ensure no "gemini" warning logged
    for call in mock_log.warning.call_args_list:
        assert "gemini" not in str(call).lower() or "unknown" not in str(call).lower()


# ---------------------------------------------------------------------------
# 13. Config model accepts anthropic provider
# ---------------------------------------------------------------------------


def test_llm_settings_accepts_anthropic():
    settings = LLMSettings(provider="anthropic", model="claude-sonnet-5")
    assert settings.provider == "anthropic"


def test_llm_settings_default_is_anthropic():
    settings = LLMSettings()
    assert settings.provider == "anthropic"
    assert settings.model == "claude-sonnet-5"


def test_secrets_reads_anthropic_env():
    secrets = Secrets.from_env({
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "ANTHROPIC_MODEL": "claude-opus-4",
    })
    assert secrets.anthropic_api_key == "sk-ant-test"
    assert secrets.anthropic_model == "claude-opus-4"


def test_env_override_anthropic_model():
    config = load_config(ROOT / "config", send_email=False, env={
        "ANTHROPIC_MODEL": "claude-haiku-4-5",
    })
    assert config.settings.llm.model == "claude-haiku-4-5"


def test_llm_enabled_anthropic():
    config = load_config(ROOT / "config", send_email=False, env={
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "LLM_PROVIDER": "anthropic",
    })
    assert config.llm_enabled is True


def test_llm_enabled_anthropic_no_key():
    config = load_config(ROOT / "config", send_email=False, env={
        "LLM_PROVIDER": "anthropic",
    })
    assert config.llm_enabled is False


# ---------------------------------------------------------------------------
# 14. Literal normalization for Claude responses
# ---------------------------------------------------------------------------


def test_normalize_literals_fixes_case():
    from src.llm.base import _normalize_literals
    from src.llm.schemas import RoleClassificationResult

    payload = {
        "is_software_engineering": True,
        "role_family": "Backend_Engineer",
        "confidence": 0.9,
        "reasoning": "test",
    }
    fixed = _normalize_literals(payload, RoleClassificationResult)
    assert fixed["role_family"] == "BACKEND_ENGINEER"


def test_normalize_literals_preserves_correct_values():
    from src.llm.base import _normalize_literals
    from src.llm.schemas import RoleClassificationResult

    payload = {
        "is_software_engineering": True,
        "role_family": "SOFTWARE_ENGINEER",
        "confidence": 0.9,
        "reasoning": "test",
    }
    fixed = _normalize_literals(payload, RoleClassificationResult)
    assert fixed["role_family"] == "SOFTWARE_ENGINEER"


def test_normalize_literals_handles_spaces():
    from src.llm.base import _normalize_literals
    from src.llm.schemas import SponsorshipLanguageResult

    payload = {"polarity": "positive", "confidence": 0.8, "quote": "", "reasoning": ""}
    fixed = _normalize_literals(payload, SponsorshipLanguageResult)
    assert fixed["polarity"] == "POSITIVE"


async def test_claude_response_with_lowercase_literals():
    """Claude may return lowercase enum values. Normalization should fix them."""
    with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
        from src.llm.anthropic import AnthropicProvider
        from src.llm.schemas import RoleClassificationResult

        provider = AnthropicProvider(_settings(), api_key="sk-ant-test")
        provider._client.messages.create = AsyncMock(
            return_value=_mock_response(json.dumps({
                "is_software_engineering": True,
                "role_family": "backend_engineer",
                "confidence": 0.95,
                "reasoning": "The posting describes backend development work.",
            }))
        )
        result = await provider.structured(
            prompt="test",
            response_model=RoleClassificationResult,
            purpose="role",
        )
        assert result is not None
        assert result.role_family == "BACKEND_ENGINEER"
        assert result.is_software_engineering is True

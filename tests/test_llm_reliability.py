"""Bounded retries, circuit recovery, and deterministic fallback."""

from __future__ import annotations

from pydantic import BaseModel

from src.agents.role_agent import run_role_classification
from src.agents.seniority_agent import run_seniority
from src.llm.base import LLMProvider, LLMSettings
from src.models.state import PipelineState
from src.services.discovery_orchestrator import orchestrate
from tests.conftest import make_job
from tests.test_discovery_orchestrator import _posting, _profile


class Echo(BaseModel):
    value: str


class ProviderError(Exception):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status_code = status


class ScriptLLM(LLMProvider):
    name = "script"

    def __init__(self, settings: LLMSettings, script: list) -> None:
        super().__init__(settings)
        self._script = list(script)
        self.generated = 0

    async def _generate(self, *, prompt: str, system: str | None, schema: dict) -> str:
        self.generated += 1
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _settings(**overrides) -> LLMSettings:
    data = dict(
        provider="gemini",
        model="test",
        timeout_seconds=2,
        max_calls_per_run=20,
        circuit_breaker_failures=5,
        circuit_reset_seconds=0,
        retry_attempts=2,
        retry_backoff_seconds=0,
    )
    data.update(overrides)
    return LLMSettings(**data)


def _provider(script: list, **overrides) -> ScriptLLM:
    return ScriptLLM(_settings(**overrides), script)


async def _ask(provider: ScriptLLM):
    return await provider.structured(prompt="classify", response_model=Echo, purpose="role")


async def test_successful_call_stays_closed() -> None:
    provider = _provider(['{"value": "ok"}'])
    parsed = await _ask(provider)
    assert parsed is not None and parsed.value == "ok"
    assert provider.stats.calls == 1
    assert provider.stats.successes == 1
    assert provider.stats.retries == 0
    assert provider.circuit_state == "CLOSED"


async def test_429_retries_once_then_can_succeed() -> None:
    provider = _provider([ProviderError(429, "429 RESOURCE_EXHAUSTED"), '{"value": "ok"}'])
    parsed = await _ask(provider)
    assert parsed is not None and parsed.value == "ok"
    assert provider.generated == 2
    assert provider.stats.retries == 1
    assert provider.stats.failures_by_category["429"] == 1
    assert provider.circuit_state == "CLOSED"


async def test_repeated_429_opens_without_a_third_attempt() -> None:
    provider = _provider(
        [ProviderError(429, "429 RESOURCE_EXHAUSTED"), ProviderError(429, "429 RESOURCE_EXHAUSTED"), '{"value": "nope"}']
    )
    parsed = await _ask(provider)
    assert parsed is None
    assert provider.generated == 2
    assert provider.stats.failures_by_category["429"] == 2
    assert provider.circuit_state == "OPEN"
    assert provider.stats.fallbacks == 1


async def test_503_retries_then_succeeds() -> None:
    provider = _provider(
        [ProviderError(503, "503 UNAVAILABLE"), ProviderError(503, "503 UNAVAILABLE"), '{"value": "ok"}']
    )
    parsed = await _ask(provider)
    assert parsed is not None and parsed.value == "ok"
    assert provider.generated == 3
    assert provider.stats.retries == 2
    assert provider.stats.failures_by_category["503"] == 2
    assert provider.circuit_state == "CLOSED"


async def test_timeout_retries() -> None:
    provider = _provider([TimeoutError("timed out"), '{"value": "ok"}'])
    parsed = await _ask(provider)
    assert parsed is not None and parsed.value == "ok"
    assert provider.stats.retries == 1
    assert provider.stats.failures_by_category["timeout"] == 1


async def test_non_retryable_error_is_not_retried() -> None:
    provider = _provider([ValueError("invalid prompt"), '{"value": "ok"}'])
    parsed = await _ask(provider)
    assert parsed is None
    assert provider.generated == 1
    assert provider.stats.retries == 0
    assert provider.stats.failures_by_category["other"] == 1
    assert provider.circuit_state == "CLOSED"


async def test_malformed_response_is_not_retried() -> None:
    provider = _provider(["not-json", '{"value": "ok"}'])
    parsed = await _ask(provider)
    assert parsed is None
    assert provider.generated == 1
    assert provider.stats.failures_by_category["schema"] == 1
    assert provider.stats.fallbacks == 1


async def test_circuit_opens_and_skips_the_next_call() -> None:
    provider = _provider(
        [ValueError("invalid prompt"), ValueError("invalid prompt"), '{"value": "ok"}'],
        circuit_breaker_failures=2,
        circuit_reset_seconds=999,
        retry_attempts=0,
    )
    assert await _ask(provider) is None
    assert await _ask(provider) is None
    assert provider.circuit_state == "OPEN"
    assert await _ask(provider) is None
    assert provider.generated == 2
    assert provider.stats.skipped_circuit_open == 1


async def test_successful_probe_closes_the_circuit() -> None:
    provider = _provider(
        [ValueError("invalid prompt"), ValueError("invalid prompt"), '{"value": "ok"}'],
        circuit_breaker_failures=2,
        circuit_reset_seconds=0,
        retry_attempts=0,
    )
    assert await _ask(provider) is None
    assert await _ask(provider) is None
    assert provider.circuit_state == "OPEN"
    parsed = await _ask(provider)
    assert parsed is not None and parsed.value == "ok"
    assert provider.stats.recovery_attempts == 1
    assert provider.circuit_state == "CLOSED"
    assert provider.stats.successes == 1


async def test_clear_titles_do_not_call_the_model(tmp_config) -> None:
    provider = _provider([])
    state = PipelineState(
        config=tmp_config,
        jobs=[
            make_job(job_title="Software Engineer", job_id="1"),
            make_job(job_title="Backend Engineer", job_id="2"),
            make_job(job_title="Engineering Manager", job_id="3"),
        ],
        resources={"llm": provider},
    )
    await run_role_classification(state)
    assert provider.generated == 0
    assert provider.stats.deterministic_avoided == 3
    assert {job.job_id for job in state.jobs} == {"1", "2"}


async def test_explicit_five_years_does_not_call_the_model(tmp_config) -> None:
    provider = _provider([])
    state = PipelineState(
        config=tmp_config,
        jobs=[
            make_job(
                job_title="Senior Software Engineer",
                job_id="5",
                description="Minimum Qualifications\n5+ years of experience.\nJava required.\n",
            )
        ],
        resources={"llm": provider},
    )
    await run_seniority(state)
    assert provider.generated == 0
    assert provider.stats.deterministic_avoided == 1
    assert state.jobs == []


async def test_llm_unavailable_for_the_whole_run_keeps_deterministic_jobs(tmp_config) -> None:
    provider = _provider([ValueError("down")])

    class Dead(ScriptLLM):
        @property
        def available(self) -> bool:
            return False

    dead = Dead(provider.settings, [ValueError("down")])

    async def greenhouse():
        return [_posting("greenhouse", "d1", title="Data Engineer", description="Analyze datasets.")]

    async def workday():
        return [_posting("workday", "w1", method="cxs")]

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse, "workday": workday},
        llm=dead,
        profile=_profile(),
    )
    assert dead.generated == 0
    assert [job.job_id for job in result.funnel.state.jobs] == ["w1"]
    assert result.sources["workday"].status == "SUCCESS"
    assert result.intelligence_evaluated == 1
    assert result.critic_reviewed == 1


async def test_greenhouse_llm_failure_does_not_stop_workday(tmp_config) -> None:
    provider = _provider(
        [ValueError("invalid prompt"), ValueError("invalid prompt"), ValueError("invalid prompt")],
        circuit_breaker_failures=2,
        circuit_reset_seconds=999,
        retry_attempts=0,
    )

    async def greenhouse():
        return [
            _posting("greenhouse", "d1", title="Data Engineer", description="Analyze datasets."),
            _posting("greenhouse", "d2", title="Data Engineer", description="Analyze datasets."),
            _posting("greenhouse", "d3", title="Data Engineer", description="Analyze datasets."),
        ]

    async def workday():
        return [_posting("workday", "w1", method="cxs")]

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse, "workday": workday},
        llm=provider,
        profile=_profile(),
    )
    assert result.sources["greenhouse"].status == "SUCCESS"
    assert result.sources["workday"].status == "SUCCESS"
    assert [job.job_id for job in result.funnel.state.jobs] == ["w1"]
    assert provider.generated == 2
    assert provider.circuit_state == "OPEN"
    assert result.llm_circuit_skips >= 1
    assert result.llm_deterministic_avoided >= 1

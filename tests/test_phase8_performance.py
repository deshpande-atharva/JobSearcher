"""Phase 8: discovery cost, registry stability, and freshness under caching."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agents.discovery_agent import run_discovery
from src.models.config import CompanyConfig, load_config
from src.models.job import DateSource, RawJobPosting
from src.models.state import PipelineState
from src.services.ats_registry import AtsRegistry
from src.services.freshness import is_fresh
from src.sources.ashby import AshbySource
from src.sources.base import DiscoverySource, HttpClient, SourceContext
from src.sources.company_career import CompanyCareerSource
from src.sources.greenhouse import GreenhouseSource
from src.sources.lever import LeverSource
from src.utils.dates import parse_datetime
from src.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)


class _Response:
    def __init__(self, status: int = 200, text: str = "ok") -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self.text = text
        self.url = "https://example.com/jobs"
        self.is_success = 200 <= status < 300


class _Client:
    def __init__(self, status: int = 200, text: str = "ok") -> None:
        self.calls = 0
        self.status = status
        self.text = text

    async def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls += 1
        return _Response(self.status, self.text)

    async def aclose(self) -> None:
        return None


def _http(config, client: _Client) -> HttpClient:
    client_http = HttpClient(config)
    client_http._client = client
    client_http._respect_robots = False
    client_http._retries = 0
    client_http._throttle._min_delay = 0
    return client_http


def _ctx(config, http: object) -> SourceContext:
    return SourceContext(config=config, http=http, logger=get_logger("phase8"))


class _CountingHttp:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[str] = []

    async def get_json(self, url: str, **kwargs: object) -> object:
        self.calls.append(url)
        return self.payload


def test_production_limits_and_static_registry() -> None:
    config = load_config(ROOT / "config", env={})
    run = config.settings.run
    discovery = config.settings.discovery
    assert run.freshness_hours == 24
    assert run.max_concurrency == 8
    assert run.company_timeout_seconds == 240
    assert run.request_timeout_seconds == 30
    assert config.settings.scraping.min_delay_seconds == 0.25
    assert discovery.persist_ats_registry is False
    assert discovery.sources.jobright.enabled is False
    assert discovery.sources.workday.browser_enabled is False
    assert discovery.sources.workday.max_concurrency == 2
    assert discovery.sources.lever.enabled is True
    assert discovery.sources.ashby.enabled is True
    workflow = (ROOT / ".github" / "workflows" / "daily_jobs.yml").read_text(encoding="utf-8")
    assert "git add data/current data/archive" in workflow
    assert "git add config/ats_registry.yaml" not in workflow
    assert "python -m src.main" in workflow


def test_repeated_failure_does_not_rewrite_the_registry(tmp_path: Path) -> None:
    path = tmp_path / "ats_registry.yaml"
    registry = AtsRegistry(path)
    registry.remember_failure("Acme", careers_url="https://acme.example/careers")
    registry.save()
    first = path.read_bytes()
    registry.remember_failure("Acme", careers_url="https://acme.example/careers")
    assert registry._dirty is False
    registry.save()
    assert path.read_bytes() == first


@pytest.mark.asyncio
async def test_get_cache_is_per_run_and_does_not_store_post_pages(tmp_config) -> None:
    client = _Client()
    http = _http(tmp_config, client)
    first = await http.get_text("https://example.com/jobs")
    second = await http.get_text("https://example.com/jobs")
    assert first.text == second.text == "ok"
    assert client.calls == 1
    assert http.cache_hits == 1
    await http.request("POST", "https://example.com/wday/cxs/t/s/jobs", json_body={"offset": 0})
    await http.request("POST", "https://example.com/wday/cxs/t/s/jobs", json_body={"offset": 20})
    assert client.calls == 3
    await http.aclose()


@pytest.mark.asyncio
async def test_transport_failure_is_not_cached(tmp_config) -> None:
    client = _Client(status=503, text="")
    http = _http(tmp_config, client)
    await http.get_text("https://example.com/jobs")
    await http.get_text("https://example.com/jobs")
    assert client.calls == 2
    assert http.cache_hits == 0
    await http.aclose()


@pytest.mark.asyncio
async def test_boards_below_the_cap_make_one_request(tmp_config) -> None:
    tmp_config.fixture_mode = False
    company = CompanyConfig(
        name="Example",
        ats={"type": "greenhouse", "identifier": "example", "discovery": "manual"},
    )
    greenhouse = _CountingHttp(
        {
            "jobs": [
                {
                    "id": 1,
                    "title": "Software Engineer",
                    "absolute_url": "https://boards.greenhouse.io/example/jobs/1",
                    "content": "Build software.",
                    "updated_at": "2026-09-26T00:00:00Z",
                    "location": {"name": "New York"},
                }
            ]
        }
    )
    await GreenhouseSource(_ctx(tmp_config, greenhouse)).discover(company)
    assert greenhouse.calls == ["https://boards-api.greenhouse.io/v1/boards/example/jobs"]

    lever_company = company.model_copy(
        update={"ats": {"type": "lever", "identifier": "example", "discovery": "manual"}}
    )
    lever = _CountingHttp([])
    await LeverSource(_ctx(tmp_config, lever)).discover(lever_company)
    assert len(lever.calls) == 1
    assert lever.calls[0].endswith("/example")

    ashby_company = company.model_copy(
        update={"ats": {"type": "ashby", "identifier": "example", "discovery": "manual"}}
    )
    ashby = _CountingHttp(
        {
            "jobs": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "title": "Software Engineer",
                    "jobUrl": "https://jobs.ashbyhq.com/example/11111111-1111-1111-1111-111111111111",
                    "descriptionHtml": "Build software.",
                    "publishedAt": "2026-09-26T00:00:00.000Z",
                    "isListed": True,
                }
            ]
        }
    )
    await AshbySource(_ctx(tmp_config, ashby)).discover(ashby_company)
    assert len(ashby.calls) == 1


@pytest.mark.asyncio
async def test_workday_host_does_not_open_a_browser(tmp_config) -> None:
    tmp_config.fixture_mode = False
    http = HttpClient(tmp_config)
    source = CompanyCareerSource(_ctx(tmp_config, http))
    html = "<html><body></body></html>"
    rendered = await source._maybe_render(
        "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
        html,
    )
    assert rendered is None
    assert http._renderer is None
    await http.aclose()


@pytest.mark.asyncio
async def test_successful_ats_skips_career_fallback(tmp_config, monkeypatch) -> None:
    calls: list[str] = []

    async def fake_result(self, company=None):
        calls.append(self.name)
        posting = RawJobPosting(
            source=self.name,
            company_name=getattr(company, "name", None),
            title="Software Engineer",
            job_id="1",
        )
        label = getattr(company, "name", None) or "*"
        from src.sources.base import SourceResult

        return SourceResult.ok(self.name, label, [posting], 0.01)

    monkeypatch.setattr(DiscoverySource, "discover_result", fake_result)
    monkeypatch.setattr(CompanyCareerSource, "discover_result", fake_result)
    state = PipelineState(config=tmp_config)
    state.resources["http"] = HttpClient(tmp_config)

    async def no_global(*_args, **_kwargs):
        return []

    monkeypatch.setattr("src.agents.discovery_agent._run_global_sources", no_global)
    await run_discovery(state)
    assert "company_career" not in calls
    assert "greenhouse" in calls
    await state.resources["http"].aclose()


@pytest.mark.asyncio
async def test_company_timeout_is_isolated(tmp_config, monkeypatch) -> None:
    tmp_config.settings = tmp_config.settings.model_copy(
        update={
            "run": tmp_config.settings.run.model_copy(update={"company_timeout_seconds": 0.05})
        }
    )
    state = PipelineState(config=tmp_config)

    async def no_global(*_args, **_kwargs):
        return []

    async def slow(*_args, **_kwargs):
        await asyncio.sleep(1)
        return [], []

    monkeypatch.setattr("src.agents.discovery_agent._run_global_sources", no_global)
    monkeypatch.setattr("src.agents.discovery_agent._discover_company", slow)
    monkeypatch.setattr(
        "src.agents.discovery_agent._context",
        lambda _state: SourceContext(
            config=tmp_config,
            http=HttpClient(tmp_config),
            logger=get_logger("phase8"),
        ),
    )
    await run_discovery(state)
    assert state.summary.companies_failed == len(tmp_config.target_companies())
    assert state.company_outcomes
    assert all("timed out" in (item.error or "") for item in state.company_outcomes)
    assert state.summary.discovery_profile.get("http_requests") == 0


def test_cached_copy_does_not_change_freshness() -> None:
    stale = RawJobPosting(
        source="greenhouse",
        company_name="Acme",
        title="Software Engineer",
        posted_at=datetime(2026, 9, 24, 12, tzinfo=timezone.utc),
        date_source=DateSource.POSTED_DATE,
    )
    cached = stale.model_copy(deep=True)
    assert is_fresh(cached, 24, now=NOW)[0] is False
    assert cached.posted_at == stale.posted_at
    assert cached.date_source == DateSource.POSTED_DATE

    fresh = stale.model_copy(
        update={"posted_at": datetime(2026, 9, 26, 1, tzinfo=timezone.utc)}
    )
    cached_fresh = fresh.model_copy(deep=True)
    assert is_fresh(cached_fresh, 24, now=NOW)[0] is True
    assert cached_fresh.posted_at == fresh.posted_at


def test_updated_missing_malformed_and_timezone_freshness() -> None:
    updated = RawJobPosting(
        source="ashby",
        company_name="Acme",
        title="Software Engineer",
        updated_at=datetime(2026, 9, 26, 10, tzinfo=timezone.utc),
        date_source=DateSource.UPDATED_DATE,
    )
    assert is_fresh(updated, 24, now=NOW, use_updated_when_posted_missing=True)[0] is True
    assert is_fresh(updated, 24, now=NOW, use_updated_when_posted_missing=False)[0] is False

    missing = RawJobPosting(source="lever", company_name="Acme", title="Software Engineer")
    assert is_fresh(missing, 24, now=NOW) == (False, None)
    assert missing.date_source == DateSource.UNKNOWN

    assert parse_datetime("not-a-timestamp") is None
    malformed = RawJobPosting(
        source="lever",
        company_name="Acme",
        title="Software Engineer",
        posted_at_raw="not-a-timestamp",
    )
    assert is_fresh(malformed, 24, now=NOW) == (False, None)

    zoned = RawJobPosting(
        source="greenhouse",
        company_name="Acme",
        title="Software Engineer",
        posted_at=parse_datetime("2026-09-26T07:30:00-04:00"),
        date_source=DateSource.POSTED_DATE,
    )
    assert is_fresh(zoned, 24, now=NOW)[0] is True
    older = zoned.model_copy(update={"posted_at": parse_datetime("2026-09-25T07:00:00-04:00")})
    assert is_fresh(older, 24, now=NOW)[0] is False

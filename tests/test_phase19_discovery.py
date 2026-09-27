"""Phase 19: bounded public Greenhouse and Ashby boards, plus overlap dedupe."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.discovery_agent import _dedupe_raw, run_discovery
from src.agents.extraction_agent import run_extraction
from src.agents.freshness_agent import run_freshness
from src.agents.location_agent import run_location_employment
from src.agents.role_agent import run_role_classification
from src.agents.seniority_agent import run_seniority
from src.llm.base import NullLLMProvider
from src.models.config import CompanyConfig
from src.models.job import DateSource, DecisionSource, RawJobPosting, RejectionReason
from src.models.state import PipelineState
from src.services.freshness import freshness_timestamp
from src.services.global_boards import collect_global_boards
from src.services.public_board_index import (
    archive_query_url,
    parse_cdx_tokens,
    prefix_for_day,
    prefixes_for_day,
    select_board_tokens,
    valid_board_token,
)
from src.services.public_board_index import PublicBoardIndex
from src.sources.ashby import AshbySource
from src.sources.base import FetchResult, SourceContext, SourceError
from src.sources.greenhouse import GreenhouseSource
from src.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 27, 16, 0, tzinfo=timezone.utc)
APPLIED_ID = "a837cbd6-9fe4-4d74-a2dc-84f602c40694"
APPLIED_URL = f"https://jobs.ashbyhq.com/applied/{APPLIED_ID}"
GH_JOBS = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
GH_BOARD = "https://boards-api.greenhouse.io/v1/boards/{token}"
ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board/{token}"
ASHBY_HTML = "https://jobs.ashbyhq.com/{token}"


class FakeHttp:
    def __init__(self, json_routes=None, text_routes=None, slow=()):
        self.json_routes = json_routes or {}
        self.text_routes = text_routes or {}
        self.slow = set(slow)
        self.calls: list[str] = []

    async def get_json(self, url, **kwargs):
        self.calls.append(url)
        if url in self.slow:
            await asyncio.sleep(5)
        if url not in self.json_routes:
            raise SourceError(f"GET {url} failed: 404", http_status=404)
        payload = self.json_routes[url]
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def request(self, method, url, **kwargs):
        self.calls.append(url)
        if url in self.text_routes:
            body = self.text_routes[url]
            status = 404 if body is None else 200
            return FetchResult(url=url, status=status, text="" if body is None else body)
        return FetchResult(url=url, status=404, text="")


def _live(config):
    notion = CompanyConfig(
        name="Notion",
        ats_type="ashby",
        ats_identifier="notion",
        careers_url="https://jobs.ashbyhq.com/notion",
    )
    universe = config.universe.model_copy(update={"companies": config.universe.companies + (notion,)})
    return config.model_copy(
        update={"universe": universe, "fixture_mode": False, "dry_run": True, "send_email": False}
    )


def _state(config, http):
    state = PipelineState(config=config, resources={"http": http, "llm": NullLLMProvider()})
    state.summary.run_started_at = NOW
    return state


def _ctx(config, http):
    return SourceContext(config=config, http=http, logger=get_logger("phase19"))


def _gh_job(job_id="501"):
    return {
        "id": int(job_id) if str(job_id).isdigit() else job_id,
        "title": "Software Engineer",
        "absolute_url": f"https://boards.greenhouse.io/phase19gh/jobs/{job_id}",
        "location": {"name": "Austin, TX"},
        "first_published": "2026-09-26T12:00:00Z",
        "content": "<p>Write production code.</p>",
    }


def _ashby_job(job_id, title, url, **extra):
    payload = {
        "id": job_id,
        "title": title,
        "jobUrl": url,
        "applyUrl": url + "/application",
        "isListed": True,
        "employmentType": "FullTime",
        "publishedAt": "2026-09-26T12:00:00Z",
        "location": "Austin, TX",
        "descriptionPlain": "Required Qualifications\n0-2 years of experience.\nWrite production code.",
    }
    payload.update(extra)
    return payload


@pytest.mark.asyncio
async def test_fixture_mode_does_not_query_the_public_index(tmp_config) -> None:
    http = FakeHttp()
    state = _state(tmp_config, http)
    found = await collect_global_boards(state, _ctx(tmp_config, http))
    assert found == []
    assert http.calls == []
    assert state.summary.discovery_profile["global_boards"]["status"] == "fixture_skipped"


def test_archive_parser_keeps_only_real_board_tokens() -> None:
    body = json.dumps(
        [
            ["original", "statuscode"],
            [APPLIED_URL, "200"],
            ["https://example.com/applied", "200"],
            ["https://jobs.ashbyhq.com/100vh", "200"],
            ["https://jobs.ashbyhq.com/embed/job", "200"],
            ["https://jobs.ashbyhq.com/Applied%20Compute/abc", "200"],
            ["https://jobs.ashbyhq.com/applied/other", "404"],
        ]
    )
    tokens, seen = parse_cdx_tokens(body, kind="ashby")
    assert tokens == ["applied"]
    assert seen == 5
    lines = "\n".join(
        [
            '{"url": "https://boards.greenhouse.io/airbnb/jobs/1", "status": "200"}',
            '{"url": "https://evil.example/airbnb", "status": "200"}',
            '{"url": "javascript:alert(1)", "status": "200"}',
        ]
    )
    greenhouse, _urls = parse_cdx_tokens(lines, kind="greenhouse")
    assert greenhouse == ["airbnb"]
    assert valid_board_token("../etc") is False
    assert prefix_for_day(NOW) == prefix_for_day(NOW)
    assert len(prefix_for_day(NOW)) == 1
    spaced = prefixes_for_day(NOW, 4)
    assert spaced == prefixes_for_day(NOW, 4)
    assert len(spaced) == 4
    assert len(set(spaced)) == 4
    query = archive_query_url("jobs.ashbyhq.com", prefix_for_day(NOW), limit=10)
    assert query.startswith("https://web.archive.org/cdx/search/cdx?")
    assert "matchType=prefix" in query
    with pytest.raises(ValueError):
        archive_query_url("evil.example", "a", limit=10)


def test_daily_board_slice_is_stable_and_bounded() -> None:
    tokens = [f"board{index}" for index in range(6)]
    first = select_board_tokens(tokens, limit=2, now=NOW)
    assert first == select_board_tokens(tokens, limit=2, now=NOW)
    assert len(first) == 2
    assert select_board_tokens(tokens, limit=0, now=NOW) == []


@pytest.mark.asyncio
async def test_greenhouse_boards_are_validated_before_ingestion(tmp_config) -> None:
    config = _live(tmp_config)
    good = "phase19gh"
    http = FakeHttp(
        json_routes={
            GH_BOARD.format(token=good): {"name": "Phase Nineteen"},
            GH_JOBS.format(token=good): {"jobs": [_gh_job()]},
            GH_BOARD.format(token="emptyboard"): {"name": "Empty Board"},
            GH_JOBS.format(token="emptyboard"): {"jobs": []},
            GH_BOARD.format(token="downboard"): {"name": "Down Board"},
            GH_JOBS.format(token="downboard"): SourceError("GET failed: 500", http_status=500),
        }
    )
    index = PublicBoardIndex(
        status="ok",
        coverage="partial",
        greenhouse=[good, good, "not a token", "acmerobotics", "missing-board", "emptyboard", "downboard", "100vh"],
        detail="test",
    )
    state = _state(config, http)
    found = await collect_global_boards(state, _ctx(config, http), index=index)
    profile = state.summary.discovery_profile["global_boards"]["greenhouse"]
    assert [posting.job_id for posting in found] == ["501"]
    assert found[0].company_name == "Phase Nineteen"
    assert found[0].provenance["board_origin"] == "global_index"
    assert profile["complete_boards"] == 1
    assert profile["empty_boards"] == 1
    assert profile["failed_boards"] >= 1
    assert profile["invalid_boards"] >= 1
    assert profile["duplicate_boards"] == 1
    assert profile["configured_overlap"] == 1
    assert profile["coverage"] == "partial"
    assert GH_JOBS.format(token="acmerobotics") not in http.calls
    assert GH_JOBS.format(token="missing-board") not in http.calls
    assert "Applied Intuition" not in (ROOT / "config" / "companies.yaml").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_one_greenhouse_timeout_does_not_drop_the_other_board(tmp_config) -> None:
    config = _live(tmp_config)
    boards = config.settings.discovery.global_boards.model_copy(update={"board_timeout_seconds": 0.2})
    discovery = config.settings.discovery.model_copy(update={"global_boards": boards})
    config = config.model_copy(update={"settings": config.settings.model_copy(update={"discovery": discovery})})
    http = FakeHttp(
        json_routes={
            GH_BOARD.format(token="fastboard"): {"name": "Fast"},
            GH_JOBS.format(token="fastboard"): {"jobs": [_gh_job("77")]},
            GH_BOARD.format(token="slowboard"): {"name": "Slow"},
            GH_JOBS.format(token="slowboard"): {"jobs": [_gh_job("78")]},
        },
        slow={GH_BOARD.format(token="slowboard")},
    )
    index = PublicBoardIndex(status="ok", greenhouse=["slowboard", "fastboard"])
    state = _state(config, http)
    found = await collect_global_boards(state, _ctx(config, http), index=index)
    assert [posting.job_id for posting in found] == ["77"]
    assert state.summary.discovery_profile["global_boards"]["greenhouse"]["failed_boards"] == 1


@pytest.mark.asyncio
async def test_ashby_global_sample_and_configured_notion_stay_separate(tmp_config) -> None:
    config = _live(tmp_config)
    notion = next(company for company in config.target_companies() if company.ats_identifier == "notion")
    notion_url = "https://jobs.ashbyhq.com/notion/bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    http = FakeHttp(
        json_routes={
            ASHBY_API.format(token="notion"): {"jobs": [_ashby_job("bbbbbbbb-cccc-dddd-eeee-ffffffffffff", "Software Engineer", notion_url)]},
            ASHBY_API.format(token="emptyash"): {"jobs": []},
            ASHBY_API.format(token="downash"): SourceError("GET failed: 500", http_status=500),
            ASHBY_API.format(token="phase19ash"): {
                "jobs": [_ashby_job("cccccccc-dddd-eeee-ffff-000000000001", "Software Engineer", "https://jobs.ashbyhq.com/phase19ash/cccccccc-dddd-eeee-ffff-000000000001")]
            },
        },
        text_routes={
            ASHBY_HTML.format(token="emptyash"): "<title>Empty Ash Jobs</title>",
            ASHBY_HTML.format(token="downash"): "<title>Down Ash Jobs</title>",
            ASHBY_HTML.format(token="phase19ash"): "<title>Phase Nineteen Ash Jobs</title>",
            ASHBY_HTML.format(token="missingash"): None,
        },
    )
    configured = await AshbySource(_ctx(config, http)).discover(notion)
    assert configured[0].job_id == "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    assert configured[0].apply_url == notion_url
    before = len(http.calls)
    index = PublicBoardIndex(
        status="ok",
        coverage="partial",
        ashby=["notion", "notion", "phase19ash", "missingash", "emptyash", "downash"],
    )
    state = _state(config, http)
    found = await collect_global_boards(state, _ctx(config, http), index=index)
    profile = state.summary.discovery_profile["global_boards"]["ashby"]
    assert [posting.job_id for posting in found] == ["cccccccc-dddd-eeee-ffff-000000000001"]
    assert found[0].company_name == "Phase Nineteen Ash"
    assert profile["configured_overlap"] == 1
    assert profile["duplicate_boards"] == 1
    assert profile["empty_boards"] == 1
    assert profile["failed_boards"] == 1
    assert profile["invalid_boards"] == 1
    assert profile["complete_boards"] == 1
    assert ASHBY_API.format(token="notion") not in http.calls[before:]


@pytest.mark.asyncio
async def test_ashby_cap_is_partial_not_empty(tmp_config) -> None:
    config = _live(tmp_config)
    ashby = config.settings.discovery.sources.ashby.model_copy(update={"max_jobs": 1})
    sources = config.settings.discovery.sources.model_copy(update={"ashby": ashby})
    discovery = config.settings.discovery.model_copy(
        update={"sources": sources, "global_boards": config.settings.discovery.global_boards}
    )
    config = config.model_copy(update={"settings": config.settings.model_copy(update={"discovery": discovery})})
    first = _ashby_job("dddddddd-eeee-ffff-0000-111111111111", "Software Engineer", "https://jobs.ashbyhq.com/capped/dddddddd-eeee-ffff-0000-111111111111")
    second = _ashby_job("dddddddd-eeee-ffff-0000-222222222222", "Backend Engineer", "https://jobs.ashbyhq.com/capped/dddddddd-eeee-ffff-0000-222222222222", publishedAt="2026-09-20T12:00:00Z")
    http = FakeHttp(
        json_routes={ASHBY_API.format(token="capped"): {"jobs": [first, second]}},
        text_routes={ASHBY_HTML.format(token="capped"): "<title>Capped Jobs</title>"},
    )
    index = PublicBoardIndex(status="ok", ashby=["capped"])
    state = _state(config, http)
    found = await collect_global_boards(state, _ctx(config, http), index=index)
    profile = state.summary.discovery_profile["global_boards"]["ashby"]
    assert len(found) == 1
    assert profile["partial_boards"] == 1
    assert profile["empty_boards"] == 0
    assert profile["failed_boards"] == 0


def test_configured_and_global_copies_collapse_to_one_job() -> None:
    url = "https://boards.greenhouse.io/phase19gh/jobs/501"
    configured = RawJobPosting(
        source="greenhouse",
        company_name="Phase Nineteen",
        title="Software Engineer",
        job_id="501",
        apply_url=url,
        provenance={"discovered_from": ["greenhouse"]},
    )
    global_copy = configured.model_copy(
        update={"provenance": {"discovered_from": ["global_index"], "board_origin": "global_index"}}
    )
    unique = _dedupe_raw([configured, global_copy])
    assert len(unique) == 1
    assert unique[0].job_id == "501"
    assert "global_index" in unique[0].provenance["discovered_from"]


@pytest.mark.asyncio
async def test_applied_intuition_is_discovered_without_a_companies_yaml_entry(tmp_config) -> None:
    yaml_text = (ROOT / "config" / "companies.yaml").read_text(encoding="utf-8")
    assert "Applied Intuition" not in yaml_text
    assert "a837cbd6-9fe4-4d74-a2dc-84f602c40694" not in yaml_text
    config = _live(tmp_config)
    http = FakeHttp(
        json_routes={
            ASHBY_API.format(token="applied"): {
                "jobs": [
                    _ashby_job(
                        APPLIED_ID,
                        "Software Engineer - New Grad (December 2026)",
                        APPLIED_URL,
                        location="Sunnyvale",
                        secondaryLocations=[
                            {
                                "location": "Ann Arbor",
                                "address": {
                                    "postalAddress": {
                                        "addressLocality": "Ann Arbor",
                                        "addressRegion": "Michigan",
                                        "addressCountry": "United States",
                                    }
                                },
                            }
                        ],
                        address={
                            "postalAddress": {
                                "addressLocality": "Sunnyvale",
                                "addressRegion": "California",
                                "addressCountry": "United States",
                            }
                        },
                        publishedAt="2026-08-14T21:57:14.117+00:00",
                        updatedAt=None,
                        employmentType="FullTime",
                    )
                ]
            }
        },
        text_routes={ASHBY_HTML.format(token="applied"): "<title>Applied Intuition Jobs</title>"},
    )
    index = PublicBoardIndex(status="ok", coverage="partial", ashby=["applied"], detail="injected test index")
    state = _state(config, http)
    found = await collect_global_boards(state, _ctx(config, http), index=index)
    assert len(found) == 1
    posting = found[0]
    assert posting.company_name == "Applied Intuition"
    assert posting.job_id == APPLIED_ID
    assert posting.apply_url == APPLIED_URL
    assert "California" in (posting.location_raw or "")
    assert posting.posted_at == datetime(2026, 8, 14, 21, 57, 14, 117000, tzinfo=timezone.utc)
    assert posting.updated_at is None
    assert posting.date_source is DateSource.POSTED_DATE

    state.raw_postings = found
    state.resources["freshness_now"] = NOW
    await run_extraction(state)
    await run_role_classification(state)
    await run_seniority(state)
    await run_location_employment(state)
    assert [job.job_id for job in state.jobs] == [APPLIED_ID]
    job = state.jobs[0]
    assert job.direct_application_url == APPLIED_URL
    assert job.employment_type.value == "Full-time"
    assert any(decision.agent == "role" and decision.passed and decision.decided_by is DecisionSource.DETERMINISTIC for decision in job.decisions)
    assert any(decision.agent == "location" and decision.passed for decision in job.decisions)
    job.first_seen_at = NOW
    posted = job.posted_at
    await run_freshness(state)
    assert state.jobs == []
    assert job.posted_at == posted
    assert job.first_seen_at == NOW
    assert job.first_seen_at != job.posted_at
    assert freshness_timestamp(job) == job.posted_at
    assert job.freshness_tier == "OLD"
    assert state.rejected[-1].reason is RejectionReason.FRESHNESS
    assert all(item.reason is not RejectionReason.ROLE for item in state.rejected)
    assert "OLD" in (state.rejected[-1].detail or "")


@pytest.mark.asyncio
async def test_a_global_index_crash_does_not_abort_configured_discovery(tmp_config, monkeypatch) -> None:
    async def boom(*_args, **_kwargs):
        raise RuntimeError("index down")

    monkeypatch.setattr("src.services.global_boards.collect_global_boards", boom)
    config = tmp_config.model_copy(update={"company_filter": "Acme Robotics", "fixture_mode": True, "dry_run": True})
    http = FakeHttp()
    state = PipelineState(config=config, resources={"http": http, "llm": NullLLMProvider()})
    await run_discovery(state)
    assert any("global board discovery failed" in note for note in state.summary.warnings)
    assert state.summary.companies_attempted == 1


def test_configured_greenhouse_collector_still_reads_a_board(tmp_config) -> None:
    source = GreenhouseSource
    assert source.name == "greenhouse"
    assert any(company.ats_identifier == "acmerobotics" for company in tmp_config.target_companies())

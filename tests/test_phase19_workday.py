"""Phase 19.1: bounded public Workday boards reuse the existing CXS collector."""

from __future__ import annotations

import asyncio
import base64
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from src.agents.dedup_agent import run_dedup
from src.agents.discovery_agent import _record_discovery_profile
from src.graph.pipeline import run_pipeline
from src.agents.extraction_agent import run_extraction
from src.agents.freshness_agent import run_freshness
from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
from src.agents.intelligence_agent import run_job_intelligence
from src.agents.location_agent import run_location_employment
from src.agents.output_agent import run_output
from src.agents.qc_agent import run_quality_control
from src.agents.role_agent import run_role_classification
from src.agents.seniority_agent import run_seniority
from src.agents.url_agent import run_url_verification
from src.llm.base import NullLLMProvider
from src.models.config import CompanyConfig
from src.models.job import DecisionSource, RejectionReason
from src.models.state import PipelineState
from src.services.global_boards import collect_global_boards
from src.services.production_report import render_pipeline_health
from src.services.public_board_index import (
    PublicBoardIndex,
    parse_cdx_workday,
    parse_public_workday_board,
    select_workday_boards,
    workday_archive_query_url,
    workday_board_identity,
    workday_clusters_for_run,
)
from src.services.xlsx import ALL_COLUMNS
from src.sources.base import FetchResult, SourceContext
from src.sources.workday import CXS_PAGE_SIZE, cxs_request_body
from src.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 27, 16, 0, tzinfo=timezone.utc)
HOST = "phase19labs.wd5.myworkdayjobs.com"
SITE = "External"
CAREER = f"https://{HOST}/{SITE}"
CXS = f"https://{HOST}/wday/cxs/phase19labs/{SITE}/jobs"
INDEX_URL = f"https://{HOST}/en-US/{SITE}/job/Austin/Software-Engineer_JR100"
CONFIGURED = "https://configured.wd5.myworkdayjobs.com/en-US/External"
CONFIGURED_CXS = "https://configured.wd5.myworkdayjobs.com/wday/cxs/configured/External/jobs"


def _posting(job_id: str, title: str, posted: str) -> dict:
    return {
        "title": title,
        "externalPath": f"/job/Austin/{title.replace(' ', '-')}_{job_id}",
        "id": job_id,
        "locationsText": "Austin, Texas, United States",
        "postedOn": posted,
    }


class WorkdayHttp:
    def __init__(self, hosts: dict[str, str] | None = None):
        self.hosts = hosts or {}
        self.bodies: list[dict] = []
        self.calls: list[tuple[str, str]] = []
        self.workday_post_count = 0

    async def head(self, url, **kwargs):
        return await self.request("HEAD", url, **kwargs)

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        host = urlparse(url).netloc.lower()
        mode = self.hosts.get(host, "ok")
        if method != "POST":
            if mode == "timeout":
                await asyncio.sleep(1)
            title = "Phase Nineteen Labs Careers"
            return FetchResult(url=url, status=200, text=f"<title>{title}</title>")
        body = dict(kwargs.get("json_body") or {})
        self.bodies.append(body)
        if mode == "timeout":
            await asyncio.sleep(1)
        limit = body.get("limit", 20)
        if isinstance(limit, int) and limit > CXS_PAGE_SIZE:
            return FetchResult(url=url, status=400, text="{}", error="HTTP 400")
        if mode == "fail":
            return FetchResult(url=url, status=404, text="", error="HTTP 404")
        if mode == "invalid":
            return FetchResult(url=url, status=200, text='{"ok": true}')
        if mode == "empty":
            return FetchResult(url=url, status=200, text='{"total": 0, "jobPostings": []}')
        if mode == "cap":
            return self._cap(url, body)
        payload = {
            "total": 1,
            "jobPostings": [
                _posting("JR100", "Software Engineer - New Grad", "2026-09-27T12:00:00Z"),
                _posting("JR200", "Software Engineer - New Grad", "2026-08-01T12:00:00Z"),
                _posting("JR300", "Account Manager", "2026-09-27T12:00:00Z"),
            ],
        }
        return FetchResult(url=url, status=200, text=json.dumps(payload))

    def _cap(self, url: str, body: dict) -> FetchResult:
        if body.get("searchText"):
            jobs = [_posting("JR-PART", "Software Engineer", "2026-09-27T12:00:00Z")]
            return FetchResult(url=url, status=200, text=json.dumps({"total": 1, "jobPostings": jobs}))
        offset = int(body.get("offset") or 0)
        if offset > 0:
            return FetchResult(url=url, status=200, text='{"total": 0, "jobPostings": []}')
        jobs = [
            _posting(f"R{index}", f"Software Engineer {index}", "2026-09-27T12:00:00Z")
            for index in range(20)
        ]
        return FetchResult(
            url=url,
            status=200,
            text=json.dumps({"total": 2000, "jobPostings": jobs}),
        )


def _config(tmp_config, *, extra: tuple = (), timeout: float | None = None, max_jobs: int | None = None):
    universe = tmp_config.universe.model_copy(update={"companies": tmp_config.universe.companies + extra})
    settings = tmp_config.settings
    if timeout is not None:
        settings = settings.model_copy(
            update={"run": settings.run.model_copy(update={"company_timeout_seconds": timeout})}
        )
    if max_jobs is not None:
        sources = settings.discovery.sources
        workday = sources.workday.model_copy(
            update={
                "max_jobs": max_jobs,
                "partitions_enabled": True,
                "max_partitions_per_company": 1,
                "max_jobs_per_partition": 20,
                "partition_search_texts": ("software engineer",),
            }
        )
        discovery = settings.discovery.model_copy(
            update={"sources": sources.model_copy(update={"workday": workday})}
        )
        settings = settings.model_copy(update={"discovery": discovery})
    return tmp_config.model_copy(
        update={
            "universe": universe,
            "settings": settings,
            "fixture_mode": False,
            "dry_run": True,
            "send_email": False,
        }
    )


def _state(config, http) -> PipelineState:
    state = PipelineState(config=config, resources={"http": http, "llm": NullLLMProvider()})
    state.summary.run_started_at = NOW
    return state


def _ctx(config, http) -> SourceContext:
    return SourceContext(config=config, http=http, logger=get_logger("phase19-workday"))


def _index(*urls: str) -> PublicBoardIndex:
    return PublicBoardIndex(
        status="ok",
        coverage="partial",
        workday=list(urls),
        workday_detail="GLOBAL DISCOVERY: PARTIAL COVERAGE",
    )


def test_workday_archive_query_is_a_bounded_public_seek() -> None:
    url = workday_archive_query_url(domain="myworkdayjobs.com", cluster="wd5", prefix="n", limit=40)
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert params["matchType"] == ["domain"]
    assert params["url"] == ["myworkdayjobs.com"]
    assert "appliedFacets" not in url
    text = zlib.decompress(base64.b64decode(params["resumeKey"][0])).decode()
    assert text.startswith("com,myworkdayjobs,wd5,n)")
    with pytest.raises(ValueError):
        workday_archive_query_url(domain="example.com", cluster="wd5", prefix="n", limit=5)
    assert CXS_PAGE_SIZE == 20
    body = cxs_request_body(limit=20, offset=0, search_text="")
    assert body == {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}


def test_workday_parser_rejects_non_career_urls() -> None:
    career = "https://nationwidechildrens.wd5.myworkdayjobs.com/en-US/NCHCareers"
    detail = career + "/details/Assistant_R-1"
    assert parse_public_workday_board(detail) == (
        "nationwidechildrens.wd5.myworkdayjobs.com",
        "nationwidechildrens",
        "NCHCareers",
    )
    assert workday_board_identity(detail) == workday_board_identity(career)
    nvidia = "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite"
    assert workday_board_identity(nvidia) == (
        "nvidia.wd5.myworkdayjobs.com|nvidia|nvidiaexternalcareersite"
    )
    assert parse_public_workday_board("https://evil.example/workday-jobs") is None
    assert parse_public_workday_board("https://2fboeing.wd1.myworkdayjobs.com/en-US/External") is None
    assert parse_public_workday_board("https://foo.impl-wd5.myworkdayjobs.com/en-US/External") is None
    assert parse_public_workday_board("https://foo.wd5.myworkdayjobs.com/assets/logo") is None
    body = json.dumps(
        [
            ["original", "statuscode"],
            [detail, "200"],
            [career + "/job/Austin/Engineer_R9", "200"],
            ["https://foo.impl-wd5.myworkdayjobs.com/en-US/External", "200"],
            ["https://evil.example/workday", "200"],
            ["https://foo.wd5.myworkdayjobs.com/favicon.ico", "200"],
        ]
    )
    urls, seen = parse_cdx_workday(body)
    assert urls == ["https://nationwidechildrens.wd5.myworkdayjobs.com/NCHCareers"]
    assert seen == 5
    assert select_workday_boards(urls, limit=0, now=NOW) == []
    assert workday_clusters_for_run(("wd12", "wd5", "wd1"), NOW) == workday_clusters_for_run(
        ("wd12", "wd5", "wd1"), NOW
    )


@pytest.mark.asyncio
async def test_global_workday_board_is_collected_when_absent_from_companies_yaml(tmp_config) -> None:
    yaml_text = (ROOT / "config" / "companies.yaml").read_text(encoding="utf-8")
    assert "phase19labs" not in yaml_text
    assert "Phase Nineteen Labs" not in yaml_text
    http = WorkdayHttp()
    state = _state(_config(tmp_config), http)
    found = await collect_global_boards(state, _ctx(state.config, http), index=_index(INDEX_URL, INDEX_URL, CAREER))
    ids = [posting.job_id for posting in found]
    assert ids == ["JR100", "JR200", "JR300"]
    assert found[0].company_name == "Phase Nineteen Labs"
    assert found[0].provenance["board_origin"] == "global_index"
    assert "global_index" in found[0].provenance["discovered_from"]
    assert all(body["limit"] == 20 and body["appliedFacets"] == {} for body in http.bodies)
    assert len(http.bodies) == 1
    profile = state.summary.discovery_profile["global_boards"]["workday"]
    assert profile["complete_boards"] == 1
    assert profile["boards_selected_for_collection"] == 1
    assert profile["unique_boards_collected"] == 1
    assert profile["duplicate_boards_skipped"] == 2
    assert profile["duplicate_boards"] == 2
    assert profile["coverage"] == "partial"
    report = render_pipeline_health(state)
    assert "GLOBAL DISCOVERY: PARTIAL COVERAGE" in report
    assert "all Workday jobs" not in report
    assert "GLOBAL WORKDAY COVERAGE" not in report
    assert len(ALL_COLUMNS) == 17


@pytest.mark.asyncio
async def test_configured_workday_board_is_not_collected_twice(tmp_config) -> None:
    configured = CompanyConfig(name="Configured Workday", ats_type="workday", ats_identifier=CONFIGURED)
    career_only_url = "https://careeronly.wd5.myworkdayjobs.com/en-US/External"
    career_only = CompanyConfig(name="Career Only", careers_url=career_only_url)
    http = WorkdayHttp()
    state = _state(_config(tmp_config, extra=(configured, career_only)), http)
    found = await collect_global_boards(
        state,
        _ctx(state.config, http),
        index=_index(
            CONFIGURED,
            CONFIGURED + "/job/Austin/Engineer_JR1",
            career_only_url,
            INDEX_URL,
        ),
    )
    assert [posting.job_id for posting in found] == ["JR100", "JR200", "JR300"]
    assert all(CONFIGURED_CXS not in url for _method, url in http.calls)
    assert all("careeronly.wd5" not in url for _method, url in http.calls)
    profile = state.summary.discovery_profile["global_boards"]["workday"]
    assert profile["configured_overlap"] == 2
    assert profile["duplicate_boards"] == 1
    assert profile["configured_boards"] == 2
    assert profile["complete_boards"] == 1
    assert profile["boards_selected_for_collection"] == 1
    assert profile["unique_boards_collected"] == 1
    assert profile["duplicate_boards_skipped"] == 3
    assert sum(1 for method, url in http.calls if method == "POST" and url == CXS) == 1


@pytest.mark.asyncio
async def test_failed_global_workday_board_does_not_stop_the_run(tmp_config) -> None:
    http = WorkdayHttp({"missing.wd5.myworkdayjobs.com": "fail"})
    missing = "https://missing.wd5.myworkdayjobs.com/en-US/External"
    state = _state(_config(tmp_config), http)
    found = await collect_global_boards(state, _ctx(state.config, http), index=_index(missing, INDEX_URL))
    assert [posting.job_id for posting in found] == ["JR100", "JR200", "JR300"]
    profile = state.summary.discovery_profile["global_boards"]["workday"]
    assert profile["failed_boards"] == 1
    assert profile["empty_boards"] == 0
    failed = [item for item in state.company_outcomes if item.http_status == 404]
    assert len(failed) == 1
    assert failed[0].status == "ERROR"
    assert failed[0].succeeded is False
    assert sum(1 for method, url in http.calls if method == "POST" and "missing.wd5" in url) == 1


@pytest.mark.asyncio
async def test_empty_global_workday_board_stays_empty(tmp_config) -> None:
    http = WorkdayHttp({"emptyco.wd5.myworkdayjobs.com": "empty"})
    empty = "https://emptyco.wd5.myworkdayjobs.com/en-US/External"
    state = _state(_config(tmp_config), http)
    found = await collect_global_boards(state, _ctx(state.config, http), index=_index(empty, INDEX_URL))
    assert any(posting.job_id == "JR100" for posting in found)
    profile = state.summary.discovery_profile["global_boards"]["workday"]
    assert profile["empty_boards"] == 1
    assert profile["failed_boards"] == 0
    empty_outcome = next(item for item in state.company_outcomes if item.status == "EMPTY")
    assert empty_outcome.succeeded is True
    assert empty_outcome.jobs_found == 0


@pytest.mark.asyncio
async def test_timeout_is_not_an_empty_workday_board(tmp_config) -> None:
    http = WorkdayHttp({"slowco.wd5.myworkdayjobs.com": "timeout"})
    slow = "https://slowco.wd5.myworkdayjobs.com/en-US/External"
    state = _state(_config(tmp_config, timeout=0.2), http)
    found = await collect_global_boards(state, _ctx(state.config, http), index=_index(slow, INDEX_URL))
    assert any(posting.job_id == "JR100" for posting in found)
    profile = state.summary.discovery_profile["global_boards"]["workday"]
    assert profile["failed_boards"] == 1
    assert profile["empty_boards"] == 0
    timed = next(item for item in state.company_outcomes if "timed out" in (item.error or ""))
    assert timed.status == "ERROR"
    assert timed.succeeded is False


@pytest.mark.asyncio
async def test_capped_global_board_stays_partial_and_reuses_partitions(tmp_config) -> None:
    http = WorkdayHttp({"phase19labs.wd5.myworkdayjobs.com": "cap"})
    state = _state(_config(tmp_config, max_jobs=20), http)
    found = await collect_global_boards(state, _ctx(state.config, http), index=_index(INDEX_URL))
    ids = {posting.job_id for posting in found}
    assert "R0" in ids
    assert "JR-PART" in ids
    profile = state.summary.discovery_profile["global_boards"]["workday"]
    assert profile["partial_boards"] == 1
    assert profile["complete_boards"] == 0
    assert all(body["limit"] == 20 and body["appliedFacets"] == {} for body in http.bodies)
    assert len([body for body in http.bodies if not body.get("searchText")]) == 1
    assert any(body.get("searchText") == "software engineer" for body in http.bodies)
    outcome = state.company_outcomes[0]
    assert outcome.diagnostics.get("source_list_cap_reached") is True
    assert outcome.diagnostics.get("partitions_attempted", 0) >= 1
    assert outcome.status != "EMPTY"


@pytest.mark.asyncio
async def test_invalid_cxs_payload_is_not_a_zero_job_board(tmp_config) -> None:
    http = WorkdayHttp({"phase19labs.wd5.myworkdayjobs.com": "invalid"})
    state = _state(_config(tmp_config), http)
    found = await collect_global_boards(state, _ctx(state.config, http), index=_index(INDEX_URL))
    assert found == []
    profile = state.summary.discovery_profile["global_boards"]["workday"]
    assert profile["invalid_boards"] == 1
    assert profile["empty_boards"] == 0
    assert profile["failed_boards"] == 0
    assert state.company_outcomes[0].status == "ERROR"


@pytest.mark.asyncio
async def test_global_workday_job_reaches_output_and_stale_is_not_a_role_mismatch(tmp_config) -> None:
    yaml_text = (ROOT / "config" / "companies.yaml").read_text(encoding="utf-8")
    assert "phase19labs" not in yaml_text
    http = WorkdayHttp()
    config = _config(tmp_config)
    candidate = config.settings.candidate.model_copy(
        update={"profile_path": "data/candidate/does-not-exist-phase191.json"}
    )
    config = config.model_copy(update={"settings": config.settings.model_copy(update={"candidate": candidate})})
    assert config.settings.discovery.persist_ats_registry is False
    assert config.settings.discovery.sources.jobright.enabled is False
    assert config.settings.discovery.sources.workday.browser_enabled is False
    assert config.settings.discovery.sources.workday.page_size == 20
    state = _state(config, http)
    found = await collect_global_boards(state, _ctx(config, http), index=_index(INDEX_URL))
    assert {posting.job_id for posting in found} == {"JR100", "JR200", "JR300"}
    assert all(posting.source == "workday" for posting in found)
    state.raw_postings = found
    state.config.fixture_mode = True
    state.resources["freshness_now"] = NOW
    await run_extraction(state)
    await run_role_classification(state)
    await run_seniority(state)
    await run_location_employment(state)
    pending = {job.job_id: job for job in state.jobs}
    for job in pending.values():
        job.first_seen_at = NOW
    await run_freshness(state)
    assert [job.job_id for job in state.jobs] == ["JR100"]
    fresh = state.jobs[0]
    assert fresh.freshness_tier == "VERY_FRESH"
    assert fresh.employment_type.value == "Full-time"
    assert any(
        decision.agent == "role" and decision.passed and decision.decided_by is DecisionSource.DETERMINISTIC
        for decision in fresh.decisions
    )
    stale_job = pending["JR200"]
    assert stale_job.freshness_tier == "OLD"
    assert stale_job.posted_at == datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    assert stale_job.first_seen_at == NOW
    assert stale_job.first_seen_at != stale_job.posted_at
    stale = next(item for item in state.rejected if item.url and "JR200" in item.url)
    account = next(item for item in state.rejected if item.title == "Account Manager")
    assert stale.reason is RejectionReason.FRESHNESS
    assert "OLD" in (stale.detail or "")
    assert account.reason is RejectionReason.ROLE
    await run_url_verification(state)
    await run_h1b_enrichment(state)
    await run_dedup(state)
    await run_quality_control(state)
    await run_job_intelligence(state)
    await run_output(state)
    assert [job.job_id for job in state.jobs] == ["JR100"]
    assert state.jobs[0].posted_at == datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    assert state.jobs[0].first_seen_at == NOW
    assert state.jobs[0].first_seen_at != state.jobs[0].posted_at
    assert state.summary.xlsx_status == "skipped"
    assert "dry-run" in (state.summary.workbook_path or "")


def test_discovery_profile_keeps_the_workday_global_line(tmp_config) -> None:
    http = WorkdayHttp()
    state = _state(tmp_config, http)
    state.summary.discovery_profile["global_boards"] = {
        "status": "ok",
        "coverage": "partial",
        "workday": {
            "configured_boards": 5,
            "discovered_boards": 8,
            "duplicate_boards": 1,
            "configured_overlap": 2,
            "complete_boards": 3,
            "partial_boards": 1,
            "failed_boards": 1,
            "empty_boards": 1,
            "jobs": 40,
        },
    }
    _record_discovery_profile(state, _ctx(tmp_config, http))
    report = render_pipeline_health(state)
    assert "global_boards: not_run" not in report
    assert "workday_global: GLOBAL DISCOVERY: PARTIAL COVERAGE" in report
    assert "configured_boards=5" in report
    assert "globally_discovered_boards=8" in report
    assert "duplicate_boards_skipped=3" in report
    assert "partial_boards=1" in report
    assert "jobs_discovered=40" in report
    assert "all Workday jobs" not in report


@pytest.mark.asyncio
async def test_full_pipeline_keeps_global_workday_on_the_final_report(tmp_config) -> None:
    """The profile rewrite after discovery must not drop the global Workday line."""
    configured = CompanyConfig(
        name="Configured Workday",
        ats_type="workday",
        ats_identifier=CONFIGURED,
    )
    http = WorkdayHttp()
    state = await run_pipeline(
        _config(tmp_config, extra=(configured,)),
        resources={
            "http": http,
            "llm": NullLLMProvider(),
            "public_board_index": _index(CONFIGURED, INDEX_URL),
        },
    )
    profile = state.summary.discovery_profile
    assert "fresh_preview_audit" in profile
    board = profile["global_boards"]["workday"]
    assert board["configured_overlap"] == 1
    assert board["boards_selected_for_collection"] == 1
    assert board["unique_boards_collected"] == 1
    assert board["duplicate_boards_skipped"] == 1
    report = render_pipeline_health(state)
    assert "global_boards: not_run" not in report
    assert "workday_global: disabled" not in report
    assert "workday_global: GLOBAL DISCOVERY: PARTIAL COVERAGE" in report
    assert f"configured_boards={board['configured_boards']}" in report
    assert f"globally_discovered_boards={board['discovered_boards']}" in report
    assert "duplicate_boards_skipped=1" in report
    assert "failed_boards=0" in report
    assert "empty_boards=0" in report
    assert "partial_boards=0" in report
    assert f"jobs_discovered={board['jobs']}" in report
    assert sum(1 for method, url in http.calls if method == "POST" and url == CONFIGURED_CXS) == 1
    assert sum(1 for method, url in http.calls if method == "POST" and url == CXS) == 1

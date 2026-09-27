"""Phase 9: Workday search partitions, fallback reasons, and freshness recall."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.discovery_agent import _career_fallback_reason, _union_seconds
from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.models.state import PipelineState
from src.services.ats_discovery import detect_ats_from_html, detect_ats_from_url, is_valid_ats_identifier
from src.services.freshness import is_fresh
from src.services.production_report import render_pipeline_health
from src.sources.base import FetchResult, HttpClient, SourceContext
from src.sources.company_career import CompanyCareerSource
from src.sources.workday import WorkdaySource
from src.utils.dates import parse_datetime
from src.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
ADOBE = "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"


def _company() -> CompanyConfig:
    return CompanyConfig(name="Adobe", ats_type="workday", ats_identifier=ADOBE, careers_url=ADOBE)


def _ctx(config, http) -> SourceContext:
    return SourceContext(config=config, http=http, logger=get_logger("phase9"))


class PartitionHttp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.bodies: list[dict] = []
        self.workday_post_count = 0

    async def request(self, method: str, url: str, **kwargs):
        body = dict(kwargs.get("json_body") or {})
        self.bodies.append(body)
        offset = int(body.get("offset") or 0)
        text = str(body.get("searchText") or "")
        if text == "fail":
            return FetchResult(url=url, status=404, text="", error="HTTP 404")
        if text:
            jobs = self.payload["partitions"].get(text, [])
            batch = jobs if offset == 0 else []
            return FetchResult(url=url, status=200, text=json.dumps({"total": len(jobs), "jobPostings": batch}))
        if offset >= 40:
            return FetchResult(url=url, status=200, text='{"total":0,"jobPostings":[]}')
        jobs = [
            {
                "title": "Software Engineer",
                "externalPath": f"/job/swe_{offset + i}",
                "id": f"R{offset + i}",
                "postedOn": "Posted 30+ Days Ago",
            }
            for i in range(20)
        ]
        total = 2000 if offset == 0 else 0
        return FetchResult(url=url, status=200, text=json.dumps({"total": total, "jobPostings": jobs}))


def _settings(config, **updates):
    workday = config.settings.discovery.sources.workday.model_copy(update=updates)
    sources = config.settings.discovery.sources.model_copy(update={"workday": workday})
    discovery = config.settings.discovery.model_copy(update={"sources": sources})
    config.settings = config.settings.model_copy(update={"discovery": discovery})
    config.fixture_mode = False
    return config


@pytest.mark.asyncio
async def test_capped_board_partitions_with_search_text_and_dedupes(tmp_config) -> None:
    payload = json.loads((ROOT / "tests" / "fixtures" / "workday" / "phase9_partitions.json").read_text())
    http = PartitionHttp(payload)
    config = _settings(
        tmp_config,
        partitions_enabled=True,
        max_partitions_per_company=2,
        max_jobs_per_partition=20,
        max_jobs=40,
        partition_search_texts=("software engineer", "backend engineer", "platform engineer"),
    )
    result = await WorkdaySource(_ctx(config, http)).discover_result(_company())
    assert result.success is True
    assert result.diagnostics["source_list_cap_reached"] is True
    assert result.diagnostics["partitions_attempted"] == 2
    assert result.diagnostics["partitions_successful"] == 2
    searched = [body["searchText"] for body in http.bodies if body["searchText"]]
    assert searched == ["software engineer", "backend engineer"]
    ids = [job.job_id for job in result.jobs]
    assert ids.count("R9001") == 1
    assert "R9002" in ids
    shared = next(job for job in result.jobs if job.job_id == "R9001")
    assert shared.provenance["workday_partitions"] == ["software engineer", "backend engineer"]
    assert shared.date_source is DateSource.POSTED_DATE
    fresh = [job for job in result.jobs if job.job_id == "R9001"]
    assert len(fresh) == 1
    stale_same_title = [job for job in result.jobs if job.title == "Software Engineer" and job.job_id == "R9002"]
    assert len(stale_same_title) == 1
    assert stale_same_title[0].job_id != shared.job_id


@pytest.mark.asyncio
async def test_partition_failure_keeps_the_unfiltered_board(tmp_config) -> None:
    payload = {"partitions": {}}
    http = PartitionHttp(payload)
    config = _settings(
        tmp_config,
        partitions_enabled=True,
        max_partitions_per_company=1,
        max_jobs=40,
        partition_search_texts=("fail",),
    )
    result = await WorkdaySource(_ctx(config, http)).discover_result(_company())
    assert result.success is True
    assert result.discovered_count == 40
    assert result.diagnostics["partitions_failed"] == 1
    assert result.diagnostics["partitions_successful"] == 0


@pytest.mark.asyncio
async def test_partition_request_budget_stops_the_company_slice(tmp_config) -> None:
    payload = json.loads((ROOT / "tests" / "fixtures" / "workday" / "phase9_partitions.json").read_text())
    http = PartitionHttp(payload)
    config = _settings(
        tmp_config,
        partitions_enabled=True,
        max_partitions_per_company=3,
        max_jobs=40,
        max_requests_per_company=2,
        partition_search_texts=("software engineer", "backend engineer"),
    )
    result = await WorkdaySource(_ctx(config, http)).discover_result(_company())
    assert result.success is True
    assert len(http.bodies) == 2
    assert result.diagnostics["partitions_attempted"] == 0


@pytest.mark.asyncio
async def test_company_timeout_cancels_workday_discovery(tmp_config) -> None:
    class SlowHttp:
        workday_post_count = 0

        async def request(self, method: str, url: str, **kwargs):
            await asyncio.sleep(1)
            return FetchResult(url=url, status=200, text='{"total":0,"jobPostings":[]}')

    config = _settings(tmp_config, partitions_enabled=False, max_jobs=40)
    source = WorkdaySource(_ctx(config, SlowHttp()))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(source.discover_result(_company()), timeout=0.05)


def test_false_icims_is_rejected_and_a_jobs_url_is_kept() -> None:
    valid = "https://example.icims.com/jobs/1234/software-engineer/job"
    false = "https://internal-amd.icims.com/jobs/search?back=none&amp;redirect=search"
    assert is_valid_ats_identifier("icims", valid) is True
    assert is_valid_ats_identifier("icims", false) is False
    kept = detect_ats_from_url(valid)
    assert kept is not None and kept.ok
    rejected = detect_ats_from_url(false)
    assert rejected is not None and rejected.method == "rejected" and not rejected.ok
    html = (ROOT / "tests" / "fixtures" / "ats" / "false_icims.html").read_text(encoding="utf-8")
    from_html = detect_ats_from_html(html)
    assert from_html is not None and not from_html.ok
    assert detect_ats_from_html("<html><body>Join our team</body></html>") is None


def test_career_fallback_reasons() -> None:
    structured = CompanyConfig(name="A", ats_type="workday", ats_identifier=ADOBE)
    plain = CompanyConfig(name="B", careers_url="https://example.com/careers")
    assert _career_fallback_reason(structured, "manual", True, []) == "ATS_FAILED"
    assert _career_fallback_reason(structured, "manual", False, []) == "ATS_RETURNED_ZERO"
    assert _career_fallback_reason(plain, "uncertain", False, []) == "ATS_DETECTION_UNCERTAIN"
    assert _career_fallback_reason(plain, "none", False, []) == "NO_STRUCTURED_ATS"
    assert _career_fallback_reason(structured, "manual", False, [RawJobPosting(source="workday")]) is None


@pytest.mark.asyncio
async def test_career_403_stops_without_detail_pages(tmp_config) -> None:
    class Denied:
        def __init__(self) -> None:
            self.calls = 0

        async def get_text(self, url: str, **kwargs):
            self.calls += 1
            return FetchResult(
                url=url,
                status=403,
                error="access denied (HTTP 403); source requires authentication or blocks automated access",
            )

    tmp_config.fixture_mode = False
    http = Denied()
    company = CompanyConfig(name="Blocked", careers_url="https://example.com/careers")
    source = CompanyCareerSource(_ctx(tmp_config, http))
    result = await source.discover_result(company)
    assert result.success is False
    assert http.calls == 1
    assert result.http_status in (None, 403)


@pytest.mark.asyncio
async def test_playwright_render_cap_skips_another_browser(tmp_config) -> None:
    tmp_config.fixture_mode = False
    http = HttpClient(tmp_config)
    http.renderer.render_count = tmp_config.settings.scraping.playwright.max_renders_per_run
    source = CompanyCareerSource(_ctx(tmp_config, http))
    rendered = await source._maybe_render("https://example.com/careers", "<html></html>")
    assert rendered is None
    await http.aclose()


def test_today_is_unknown_and_elapsed_phrases_stay_exact() -> None:
    assert parse_datetime("Posted Today", now=NOW) is None
    assert parse_datetime("today", now=NOW) is None
    posted = parse_datetime("Posted 2 Hours Ago", now=NOW)
    assert posted is not None
    job = RawJobPosting(
        source="workday",
        company_name="Adobe",
        title="Software Engineer",
        posted_at=posted,
        date_source=DateSource.POSTED_DATE,
    )
    assert is_fresh(job, 24, now=NOW)[0] is True
    exact = RawJobPosting(
        source="greenhouse",
        company_name="A",
        posted_at=NOW - timedelta(hours=24),
        date_source=DateSource.POSTED_DATE,
    )
    over = exact.model_copy(update={"posted_at": NOW - timedelta(hours=24, seconds=1)})
    assert is_fresh(exact, 24, now=NOW)[0] is True
    assert is_fresh(over, 24, now=NOW)[0] is False
    crawled = RawJobPosting(source="company_career", company_name="A", title="Software Engineer")
    assert is_fresh(crawled, 24, now=NOW) == (False, None)
    zoned = parse_datetime("2026-09-26T07:30:00-04:00")
    assert zoned is not None and zoned.utcoffset() == timedelta(0)


def test_overlapping_durations_are_not_reported_as_additive_wall_clock(tmp_config) -> None:
    assert _union_seconds([(0.0, 10.0), (5.0, 12.0)]) == 12.0
    state = PipelineState(config=tmp_config)
    state.summary.discovery_profile = {
            "by_source": {
                "workday": {"seconds_sum": 30.0, "wall_seconds": 12.0, "jobs": 2},
            },
            "workday_partitions": {"companies_capped": 1, "attempted": 2, "successful": 2, "failed": 0, "jobs": 1, "duplicates": 1, "requests": 4},
            "freshness_preview": {"fresh": 1, "stale": 1, "unknown": 0},
            "playwright_renders": 0,
            "playwright_seconds": 0,
        }
    text = render_pipeline_health(state)
    assert "source_seconds_sum:" in text
    assert "source_wall_seconds:" in text
    assert "workday=30.0s/2jobs" in text
    assert "workday=12.0s" in text
    assert "description" not in text
    assert "GEMINI_API_KEY" not in text
    assert "resume" not in text.lower()

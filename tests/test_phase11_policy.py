"""Phase 11: remote policy, junior titles, and Workday completeness diagnostics."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.discovery_agent import _dedupe_raw
from src.agents.location_agent import run_location_employment
from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting, RejectionReason, RemoteType
from src.models.state import PipelineState, SourceHealth
from src.services.ats_discovery import detect_ats_from_html
from src.services.freshness import is_fresh
from src.services.production_report import source_completeness
from src.services.roles import classify_role
from src.services.seniority import classify_seniority
from src.sources.base import FetchResult, SourceContext
from src.sources.company_career import CompanyCareerSource
from src.sources.workday import WorkdaySource
from src.utils.logging import get_logger
from src.utils.normalization import normalize_location
from tests.conftest import make_job

ROOT = Path(__file__).resolve().parents[1]
AMD_HTML = (ROOT / "tests" / "fixtures" / "company_career" / "phase11_amd.html").read_text(encoding="utf-8")
NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)
ADOBE = "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"


def _ctx(config, http) -> SourceContext:
    return SourceContext(config=config, http=http, logger=get_logger("phase11"))


def test_remote_location_matrix() -> None:
    united_states = (
        "Remote - US",
        "Remote - USA",
        "Remote - U.S.",
        "Remote - United States",
        "Remote, US",
        "Remote, USA",
        "Remote, United States",
        "United States - Remote",
        "US - Remote",
        "USA - Remote",
    )
    for raw in united_states:
        info = normalize_location(raw)
        assert info.is_us is True, raw
        assert info.is_international_only is False, raw
    international = (
        "Remote - Canada",
        "Remote, Canada",
        "Remote - Europe",
        "Remote - UK",
        "Remote - India",
        "Remote - Germany",
        "Remote + London",
        "London, United Kingdom",
        "London, United Kingdom; Remote",
    )
    for raw in international:
        info = normalize_location(raw)
        assert info.is_us is False, raw
        assert info.is_international_only is True, raw
    unknown = ("Remote", "Remote - North America", "Remote + Boston")
    for raw in unknown:
        info = normalize_location(raw)
        assert info.is_us is False, raw
        assert info.is_international_only is False, raw
    both_orders = (
        "Boston, MA; London, United Kingdom",
        "London, United Kingdom; Boston, MA",
        "Boston, MA + London, United Kingdom",
        "London, United Kingdom + Boston, MA",
    )
    for raw in both_orders:
        info = normalize_location(raw)
        assert info.is_us is True, raw
        assert info.is_international_only is False, raw


@pytest.mark.asyncio
async def test_bare_remote_is_rejected_and_explicit_us_remote_is_kept(tmp_config) -> None:
    bare = make_job(location="Remote", remote_type=RemoteType.REMOTE, job_id="bare", direct_application_url="https://example.com/jobs/bare")
    explicit = make_job(
        location="Remote - United States",
        remote_type=RemoteType.REMOTE,
        job_id="us",
        direct_application_url="https://example.com/jobs/us",
    )
    north_america = make_job(
        location="Remote - North America",
        remote_type=RemoteType.REMOTE,
        job_id="na",
        direct_application_url="https://example.com/jobs/na",
    )
    state = PipelineState(config=tmp_config)
    state.jobs = [bare, explicit, north_america]
    await run_location_employment(state)
    assert [job.job_id for job in state.jobs] == ["us"]
    reasons = {item.job_id if hasattr(item, "job_id") else item.title: item.detail for item in state.rejected}
    assert any("could not confirm" in (item.detail or "") for item in state.rejected)
    assert len(state.rejected) == 2
    assert reasons


def test_junior_titles_follow_the_role_gate(tmp_config) -> None:
    roles = tmp_config.roles
    for title in (
        "Junior Software Engineer",
        "Junior Backend Engineer",
        "Junior Platform Engineer",
        "Junior Software Developer",
    ):
        verdict = classify_role(title, "Write production code.", roles)
        assert verdict.is_software_engineering is True, title
        seniority = classify_seniority(title, "", roles, max_required_years=2)
        assert seniority.fits_entry_level is True, title
    for title in (
        "Junior Account Manager",
        "Junior Marketing Manager",
        "Junior Counsel",
        "Junior Sales Engineer",
    ):
        verdict = classify_role(title, "Own accounts and customer relationships.", roles)
        assert verdict.is_software_engineering is False, title
    contradictory = classify_seniority(
        "Senior Junior Software Engineer",
        "",
        roles,
        max_required_years=2,
    )
    assert contradictory.fits_entry_level is False
    kept = classify_seniority(
        "Senior Software Engineer",
        "Required Qualifications\n0-2 years of experience.",
        roles,
        max_required_years=2,
    )
    assert kept.fits_entry_level is True
    one_year = classify_seniority(
        "Senior Software Engineer",
        "Required Qualifications\n1+ years of experience.",
        roles,
        max_required_years=2,
    )
    assert one_year.fits_entry_level is True
    rejected = classify_seniority(
        "Senior Software Engineer",
        "Required Qualifications\n5+ years of experience.",
        roles,
        max_required_years=2,
    )
    assert rejected.fits_entry_level is False


def test_exact_freshness_boundary_and_timestamp_priority() -> None:
    fresh = RawJobPosting(
        source="greenhouse",
        company_name="A",
        posted_at=NOW - timedelta(hours=1),
        date_source=DateSource.POSTED_DATE,
        discovered_at=NOW,
    )
    stale = fresh.model_copy(update={"posted_at": NOW - timedelta(days=3)})
    exact = fresh.model_copy(update={"posted_at": NOW - timedelta(hours=24)})
    over = fresh.model_copy(update={"posted_at": NOW - timedelta(hours=24, seconds=1)})
    unknown = RawJobPosting(source="greenhouse", company_name="A", discovered_at=NOW)
    updated_only = RawJobPosting(
        source="greenhouse",
        company_name="A",
        updated_at=NOW - timedelta(hours=2),
        date_source=DateSource.UPDATED_DATE,
        discovered_at=NOW,
    )
    assert is_fresh(fresh, 24, now=NOW)[0] is True
    assert is_fresh(stale, 24, now=NOW)[0] is False
    assert is_fresh(exact, 24, now=NOW)[0] is True
    assert is_fresh(over, 24, now=NOW)[0] is False
    assert is_fresh(unknown, 24, now=NOW) == (False, None)
    assert is_fresh(updated_only, 24, now=NOW)[0] is True
    posted_wins = updated_only.model_copy(update={"posted_at": NOW - timedelta(days=10), "date_source": DateSource.POSTED_DATE})
    assert is_fresh(posted_wins, 24, now=NOW)[0] is False


def test_dedup_timestamp_rules_and_distinct_ids() -> None:
    fresh = RawJobPosting(
        source="greenhouse",
        company_name="Example",
        title="Software Engineer",
        job_id="1",
        apply_url="https://example.com/jobs/1",
        posted_at=NOW - timedelta(hours=1),
        date_source=DateSource.POSTED_DATE,
        discovered_at=NOW,
    )
    stale = fresh.model_copy(update={"source": "lever", "posted_at": NOW - timedelta(days=20)})
    kept = _dedupe_raw([fresh, stale])
    assert len(kept) == 1 and kept[0].posted_at == fresh.posted_at
    unknown = fresh.model_copy(update={"job_id": "2", "apply_url": "https://example.com/jobs/2", "posted_at": None, "date_source": DateSource.UNKNOWN})
    later = unknown.model_copy(update={"source": "ashby", "posted_at": NOW - timedelta(hours=2), "date_source": DateSource.POSTED_DATE})
    merged = _dedupe_raw([unknown, later])
    assert merged[0].posted_at == later.posted_at
    assert merged[0].discovered_at == unknown.discovered_at
    other = fresh.model_copy(update={"job_id": "9", "apply_url": "https://example.com/jobs/9", "title": "Software Engineer"})
    assert len(_dedupe_raw([fresh, other])) == 2


def test_source_completeness_treats_a_cap_as_partial() -> None:
    health = SourceHealth(source="workday", attempted=2, successful=2, jobs_discovered=20)
    profile = {"by_source": {"workday": {"caps": 1}}}
    assert source_completeness("workday", health, profile) == "PARTIAL"
    greenhouse = SourceHealth(source="greenhouse", attempted=1, successful=1, jobs_discovered=4)
    assert source_completeness("greenhouse", greenhouse, {"by_source": {"greenhouse": {"caps": 0}}}) == "COMPLETE"
    failed = SourceHealth(source="icims", attempted=1, failed=1, error=1)
    assert source_completeness("icims", failed, {}) == "FAILED"


@pytest.mark.asyncio
async def test_amd_like_page_returns_no_jobs_and_does_not_select_icims(tmp_config) -> None:
    detection = detect_ats_from_html(AMD_HTML, page_url="https://careers.example.com/careers-home/jobs")
    assert detection is None or detection.ok is False

    class Http:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.renderer = type("R", (), {"enabled": False, "render_count": 0})()

        async def get_text(self, url: str, **kwargs):
            self.calls.append(url)
            return FetchResult(url=url, status=200, text=AMD_HTML)

    tmp_config.fixture_mode = False
    http = Http()
    result = await CompanyCareerSource(_ctx(tmp_config, http)).discover_result(
        CompanyConfig(name="Advanced Micro Devices", careers_url="https://careers.example.com/careers-home")
    )
    assert http.calls == ["https://careers.example.com/careers-home"]
    assert result.jobs == []
    assert result.diagnostics.get("page_outcome") == "UNSUPPORTED_STRUCTURE"


@pytest.mark.asyncio
async def test_capped_workday_partition_can_recover_a_fresh_job(tmp_config) -> None:
    class Board:
        def __init__(self) -> None:
            self.bodies: list[dict] = []
            self.workday_post_count = 0

        async def request(self, method: str, url: str, **kwargs):
            body = dict(kwargs.get("json_body") or {})
            self.bodies.append(body)
            text = str(body.get("searchText") or "")
            offset = int(body.get("offset") or 0)
            if text == "fail":
                return FetchResult(url=url, status=404, text="", error="HTTP 404")
            if text:
                jobs = [
                    {
                        "title": "Software Engineer",
                        "externalPath": "/job/hidden",
                        "id": "FRESH1",
                        "postedOn": "Posted 2 Hours Ago",
                    }
                ]
                return FetchResult(url=url, status=200, text=json.dumps({"total": 1, "jobPostings": jobs if offset == 0 else []}))
            if offset >= 40:
                return FetchResult(url=url, status=200, text='{"total":0,"jobPostings":[]}')
            jobs = [
                {
                    "title": "Software Engineer",
                    "externalPath": f"/job/old_{offset}",
                    "id": f"OLD{offset}",
                    "postedOn": "Posted 30+ Days Ago",
                }
                for _ in range(20)
            ]
            total = 2000 if offset == 0 else 0
            return FetchResult(url=url, status=200, text=json.dumps({"total": total, "jobPostings": jobs}))

    workday = tmp_config.settings.discovery.sources.workday.model_copy(
        update={
            "partitions_enabled": True,
            "max_partitions_per_company": 1,
            "max_jobs": 40,
            "max_jobs_per_partition": 20,
            "partition_search_texts": ["software engineer"],
        }
    )
    sources = tmp_config.settings.discovery.sources.model_copy(update={"workday": workday})
    discovery = tmp_config.settings.discovery.model_copy(update={"sources": sources})
    tmp_config.settings = tmp_config.settings.model_copy(update={"discovery": discovery})
    tmp_config.fixture_mode = False
    http = Board()
    result = await WorkdaySource(_ctx(tmp_config, http)).discover_result(
        CompanyConfig(name="NVIDIA", ats_type="workday", ats_identifier=ADOBE, careers_url=ADOBE)
    )
    assert result.diagnostics.get("source_list_cap_reached") is True
    assert result.diagnostics.get("estimated_incomplete") is True
    assert int(result.diagnostics.get("recovered_fresh_jobs") or 0) == 1
    assert any(job.job_id == "FRESH1" for job in result.jobs)
    assert sum(1 for body in http.bodies if body.get("searchText")) == 1

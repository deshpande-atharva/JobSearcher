"""Registry priority, company discovery reports, and source isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.models.config import CompanyConfig
from src.models.job import DateSource, RejectionReason
from src.models.state import CompanyOutcome, PipelineState, RejectedJob, RunSummary
from src.services.ats_discovery import detect_ats, detect_ats_from_url, is_valid_ats_identifier
from src.services.ats_registry import AtsRegistry
from src.services.discovery_report import (
    CompanyDiscoveryRow,
    attach_discovery_reports,
    render_company_discovery_report,
    render_coverage,
)
from src.services.freshness import is_fresh
from src.sources.base import SourceResult
from tests.conftest import make_job


def test_invalid_identifier_is_rejected() -> None:
    assert is_valid_ats_identifier("greenhouse", None) is False
    assert is_valid_ats_identifier("greenhouse", "") is False
    assert is_valid_ats_identifier("greenhouse", "embed") is False
    assert is_valid_ats_identifier("greenhouse", "auto") is False
    assert is_valid_ats_identifier("workday", "adobe") is False
    assert is_valid_ats_identifier("greenhouse", "airbnb") is True
    assert (
        is_valid_ats_identifier(
            "workday", "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"
        )
        is True
    )


def test_registry_refuses_invalid_identifier(tmp_path) -> None:
    registry = AtsRegistry(tmp_path / "ats_registry.yaml")
    assert (
        registry.remember(
            "Example",
            ats_type="greenhouse",
            ats_identifier="embed",
            method="automatic",
        )
        is False
    )
    assert registry.get_verified("Example") is None


def test_registry_failed_discovery_has_no_identifier(tmp_path) -> None:
    registry = AtsRegistry(tmp_path / "ats_registry.yaml")
    registry.remember_failure("Quiet Corp", careers_url="https://quiet.example/careers")
    entry = registry.get("Quiet Corp")
    assert entry is not None
    assert entry.discovery_method == "failed"
    assert entry.ats_identifier is None
    assert entry.verified is False
    assert registry.get_verified("Quiet Corp") is None


def test_registry_automatic_then_failed_does_not_clobber_verified(tmp_path) -> None:
    registry = AtsRegistry(tmp_path / "ats_registry.yaml")
    registry.remember(
        "Airbnb",
        ats_type="greenhouse",
        ats_identifier="airbnb",
        method="automatic",
    )
    registry.remember_failure("Airbnb")
    cached = registry.get_verified("Airbnb")
    assert cached is not None
    assert cached.ats_identifier == "airbnb"


def test_manual_company_config_beats_registry_and_url() -> None:
    company = CompanyConfig(
        name="Example",
        careers_url="https://jobs.lever.co/otherco",
        ats={"type": "greenhouse", "identifier": "example", "discovery": "manual"},
    )
    assert company.ats_discovery_mode == "manual"
    assert company.ats_type == "greenhouse"
    assert company.ats_identifier == "example"
    auto = detect_ats_from_url(company.careers_url)
    assert auto is not None and auto.ats_type == "lever"


@pytest.mark.asyncio
async def test_cache_beats_auto_url_detection(tmp_config, tmp_path) -> None:
    from src.agents.discovery_agent import resolve_company_ats
    from src.models.state import PipelineState

    company = CompanyConfig(
        name="Cached Co",
        careers_url="https://jobs.lever.co/otherco",
        ats={"type": "auto", "identifier": None, "discovery": "auto"},
    )
    registry = AtsRegistry(tmp_path / "ats_registry.yaml")
    registry.remember(
        "Cached Co",
        ats_type="greenhouse",
        ats_identifier="cachedco",
        method="automatic",
    )
    state = PipelineState(config=tmp_config)

    class _Ctx:
        http = None

    resolved, method = await resolve_company_ats(state, _Ctx(), registry, company)
    assert method == "registry"
    assert resolved.ats_type == "greenhouse"
    assert resolved.ats_identifier == "cachedco"


@pytest.mark.asyncio
async def test_manual_beats_registry_cache(tmp_config, tmp_path) -> None:
    from src.agents.discovery_agent import resolve_company_ats

    company = CompanyConfig(
        name="Manual Co",
        careers_url="https://jobs.lever.co/otherco",
        ats={"type": "ashby", "identifier": "manualco", "discovery": "manual"},
    )
    registry = AtsRegistry(tmp_path / "ats_registry.yaml")
    registry.remember(
        "Manual Co",
        ats_type="greenhouse",
        ats_identifier="cached",
        method="automatic",
    )
    state = PipelineState(config=tmp_config)

    class _Ctx:
        http = None

    resolved, method = await resolve_company_ats(state, _Ctx(), registry, company)
    assert method == "manual"
    assert resolved.ats_type == "ashby"
    assert resolved.ats_identifier == "manualco"


def test_ats_discovery_result_confidence_from_url() -> None:
    hit = detect_ats("https://boards.greenhouse.io/airbnb")
    assert hit.detected is True
    assert hit.confidence == 1.0
    assert hit.ats_type == "greenhouse"
    assert hit.identifier == "airbnb"


def test_company_discovery_report_uses_pipeline_numbers() -> None:
    summary = RunSummary()
    rows = [
        CompanyDiscoveryRow(
            company="Company A",
            ats="Greenhouse",
            source="ATS",
            raw=83,
            swe=17,
            exp=5,
            fresh=2,
            final=2,
        )
    ]
    summary.company_discovery_rows = rows
    text = render_company_discovery_report(rows)
    assert "COMPANY DISCOVERY REPORT" in text
    assert "Company A" in text
    assert "83" in text
    assert "17" in text
    assert "2" in text


def test_coverage_and_funnel_distinguish_bottlenecks(tmp_config) -> None:
    state = PipelineState(config=tmp_config)
    state.company_outcomes = [
        CompanyOutcome(
            company="Acme Robotics",
            source="greenhouse",
            succeeded=True,
            jobs_found=10,
            ats_type="greenhouse",
            ats_identifier="acmerobotics",
            detection_method="manual",
        ),
        CompanyOutcome(
            company="Northwind Labs",
            source="lever",
            succeeded=False,
            jobs_found=0,
            error="HTTP 403",
            ats_type="lever",
            ats_identifier="northwindlabs",
            detection_method="manual",
        ),
        CompanyOutcome(
            company="Northwind Labs",
            source="company_career",
            succeeded=True,
            jobs_found=3,
            fallback_used=True,
            detection_method="manual",
        ),
    ]
    state.jobs = [make_job(company="Acme Robotics", job_id="ok")]
    state.rejected = [
        RejectedJob(
            company="Acme Robotics",
            title="Data Analyst",
            reason=RejectionReason.ROLE,
            source="greenhouse",
        ),
        RejectedJob(
            company="Acme Robotics",
            title="Staff Engineer",
            reason=RejectionReason.SENIORITY,
            source="greenhouse",
        ),
        RejectedJob(
            company="Acme Robotics",
            title="SWE",
            reason=RejectionReason.FRESHNESS,
            source="greenhouse",
            date_source=DateSource.UNKNOWN,
        ),
    ]
    state.summary.jobs_discovered = 13
    state.summary.jobs_extracted = 4
    state.summary.jobs_accepted = 1
    state.summary.rejected_by_role = 1
    state.summary.rejected_by_seniority = 1
    state.summary.rejected_by_freshness = 1
    attach_discovery_reports(state)

    acme = next(r for r in state.summary.company_discovery_rows if r.company == "Acme Robotics")
    north = next(r for r in state.summary.company_discovery_rows if r.company == "Northwind Labs")
    assert acme.raw == 10
    assert acme.swe == 3
    assert acme.exp == 2
    assert acme.final == 1
    assert north.fallback_used is True
    assert north.raw == 3
    assert "HTTP 403" in (north.failure or "")
    assert state.summary.coverage is not None
    assert state.summary.coverage.companies_with_career_fallback >= 1
    coverage_text = render_coverage(state.summary.coverage)
    assert "Greenhouse" in coverage_text
    funnel = state.summary.filter_funnel()
    assert "UNKNOWN TIMESTAMP JOBS" in funnel
    assert "FINAL ACCEPTED" in funnel


def test_zero_jobs_success_vs_failure_remain_distinct() -> None:
    ok_empty = SourceResult.ok("workday", "Elevance Health", [])
    failed = SourceResult.fail("workday", "Elevance Health", "WORKDAY_UNSUPPORTED_CONFIGURATION")
    assert ok_empty.success is True
    assert ok_empty.discovered_count == 0
    assert failed.success is False
    assert failed.discovered_count == 0


def test_freshness_window_and_unknown(tmp_config) -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    under = make_job(posted_at=now - timedelta(hours=23, minutes=59), date_source=DateSource.POSTED_DATE)
    exact = make_job(posted_at=now - timedelta(hours=24), date_source=DateSource.POSTED_DATE)
    over = make_job(posted_at=now - timedelta(hours=24, minutes=1), date_source=DateSource.POSTED_DATE)
    unknown = make_job(posted_at=None, updated_at=None)
    unknown.date_source = DateSource.UNKNOWN
    assert is_fresh(under, 24, now=now)[0] is True
    assert is_fresh(exact, 24, now=now)[0] is True
    assert is_fresh(over, 24, now=now)[0] is False
    assert is_fresh(unknown, 24, now=now)[0] is False
    assert is_fresh(over, 72, now=now)[0] is True


def test_safety_cap_is_round_robin_and_reported() -> None:
    from src.agents.discovery_agent import _apply_safety_cap
    from src.models.job import RawJobPosting

    posts = []
    for company, count in (("Airbnb", 10), ("Notion", 10), ("Palantir", 10)):
        for i in range(count):
            posts.append(
                RawJobPosting(
                    source="greenhouse",
                    company_name=company,
                    title=f"Software Engineer {i}",
                    job_id=f"{company}-{i}",
                    apply_url=f"https://boards.greenhouse.io/{company.lower()}/jobs/{i}",
                )
            )
    kept, truncated = _apply_safety_cap(posts, 12)
    assert truncated == 18
    counts = {}
    for item in kept:
        counts[item.company_name] = counts.get(item.company_name, 0) + 1
    assert counts == {"Airbnb": 4, "Notion": 4, "Palantir": 4}


def test_freshness_breakdown_and_nearest_misses() -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    summary = RunSummary()
    summary.freshness_hours_used = 24
    summary.freshness_breakdown = {
        "candidates": 37,
        "posted_within": 0,
        "posted_older": 31,
        "updated_within": 0,
        "updated_older": 4,
        "unknown": 2,
    }
    summary.nearest_freshness_misses = [
        {
            "company": "Airbnb",
            "title": "Software Engineer",
            "age_hours": 27.0,
            "date_source": "POSTED_DATE",
        }
    ]
    text = summary.freshness_candidate_report()
    assert "Candidates before freshness: 37" in text
    assert "within 24h: 0" in text
    assert "older than 24h: 31" in text
    misses = summary.nearest_misses_report()
    assert "Airbnb" in misses
    assert "27.0h" in misses
    assert is_fresh(make_job(posted_at=now - timedelta(hours=27)), 24, now=now)[0] is False


def test_generic_careers_homepage_rejected_when_specific_url_exists() -> None:
    from src.models.config import load_config
    from src.models.job import RawJobPosting
    from src.services.url_verification import pick_direct_url
    from tests.conftest import ROOT

    policy = load_config(ROOT / "config", env={}, fixture_mode=True, send_email=False).url_policy
    raw = RawJobPosting(
        source="company_career",
        company_name="Airbnb",
        title="Software Engineer",
        apply_url="https://careers.airbnb.com/",
        alternate_urls=["https://boards.greenhouse.io/airbnb/jobs/4012345"],
    )
    check = pick_direct_url(raw, policy)
    assert check.accepted is True
    assert "greenhouse.io" in (check.url or "")

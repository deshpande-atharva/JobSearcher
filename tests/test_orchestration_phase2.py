"""Phase 2 fixtures: accepted jobs, dedup, browser-only discovery, and profile reporting."""

from __future__ import annotations

from datetime import timedelta

from src.agents.resume_intelligence_agent import CanonicalProfile, ResumeMeta, SkillItem
from src.main import build_parser
from src.models.config import CompanyConfig, CompanyUniverse
from src.models.job import DateSource, RawJobPosting
from src.services.discovery_orchestrator import (
    CANONICAL_PROFILE_SHA256,
    SourceDiscoveryError,
    collect_workday,
    grounding_status,
    orchestrate,
)
from src.services.freshness import is_fresh
from src.utils.dates import utcnow
from tests.test_discovery_orchestrator import _posting, _profile


def test_phase2_commands_do_not_enable_production_browser(tmp_config) -> None:
    browser = build_parser().parse_args(
        ["--multi-source-browser-smoke-test", "--sources", "greenhouse,workday", "--dry-run", "--no-email"]
    )
    downstream = build_parser().parse_args(
        ["--live-downstream-smoke-test", "--source", "workday", "--dry-run", "--no-email"]
    )
    assert browser.multi_source_browser_smoke_test is True
    assert browser.dry_run is True
    assert browser.no_email is True
    assert downstream.live_downstream_smoke_test is True
    assert downstream.source == "workday"
    assert tmp_config.settings.discovery.sources.workday.browser_enabled is False


def test_profile_validity_is_separate_from_evidence_coverage() -> None:
    resume = ResumeMeta(filename="resume.pdf", sha256=CANONICAL_PROFILE_SHA256, processed_at="2026-01-01T00:00:00Z")
    grounded = CanonicalProfile(
        profile_version=1,
        resume=resume,
        skills=[SkillItem(name="Java", confidence="high", status="EXPLICIT", evidence=[{"quote": "Java services"}])],
    )
    assert grounding_status(grounded) == ("PASS", "PASS")
    bare = grounded.model_copy(
        update={"skills": [SkillItem(name="Java", confidence="high", status="EXPLICIT", evidence=[])]}
    )
    assert grounding_status(bare) == ("PASS", "FAIL")
    wrong = grounded.model_copy(
        update={"resume": ResumeMeta(filename="resume.pdf", sha256="abc", processed_at="2026-01-01T00:00:00Z")}
    )
    assert grounding_status(wrong) == ("FAIL", "PASS")
    assert grounding_status(None) == ("FAIL", "FAIL")


def test_discovered_at_does_not_manufacture_freshness() -> None:
    now = utcnow()
    fresh = _posting("workday", "f", posted_at=now - timedelta(hours=1))
    stale = _posting("workday", "s", posted_at=now - timedelta(hours=48))
    updated = _posting("workday", "u", posted_at=None, date_source=DateSource.UPDATED_DATE)
    updated = updated.model_copy(update={"updated_at": now - timedelta(hours=1), "posted_at": None})
    unknown = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        apply_url="https://nvidia.wd5.myworkdayjobs.com/job/X_JR1",
        discovered_at=now,
        date_source=DateSource.UNKNOWN,
    )
    assert is_fresh(fresh, 24)[0] is True
    assert is_fresh(stale, 24)[0] is False
    assert is_fresh(updated, 24, use_updated_when_posted_missing=True)[0] is True
    assert is_fresh(unknown, 24)[0] is False


async def test_both_sources_can_reach_intelligence(tmp_config) -> None:
    async def greenhouse():
        return [_posting("greenhouse", f"g{index}") for index in range(3)]

    async def workday():
        return [_posting("workday", f"w{index}", method="cxs") for index in range(3)]

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse, "workday": workday},
        profile=_profile(),
    )
    assert result.sources["greenhouse"].discovered == 3
    assert result.sources["workday"].discovered == 3
    assert len(result.postings) == 6
    assert result.intelligence_evaluated >= 2
    assert result.critic_reviewed >= 1
    assert result.funnel.final_count == result.intelligence_evaluated


async def test_same_title_and_different_ids_stay_separate(tmp_config) -> None:
    async def greenhouse():
        return [_posting("greenhouse", "same", title="Software Engineer", location="Austin, TX")]

    async def workday():
        return [
            _posting("workday", "same", method="cxs", title="Software Engineer", location="Austin, TX"),
            _posting("workday", "other", method="cxs", title="Software Engineer", location="Austin, TX"),
        ]

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse, "workday": workday},
        profile=_profile(),
    )
    ids = [job.job_id for job in result.funnel.state.jobs]
    assert ids.count("same") == 1
    assert "other" in ids
    merged = next(job for job in result.funnel.state.jobs if job.job_id == "same")
    assert "workday" in merged.additional_sources


def _browser_config(config):
    nvidia = CompanyConfig(
        name="NVIDIA",
        ats_type="workday",
        ats_identifier="https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite",
        careers_url="https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite",
    )
    config = config.model_copy(
        update={"fixture_mode": False, "universe": CompanyUniverse(companies=(nvidia,))}
    )
    workday = config.settings.discovery.sources.workday.model_copy(update={"enabled": True, "browser_enabled": True})
    sources = config.settings.discovery.sources.model_copy(update={"workday": workday})
    discovery = config.settings.discovery.model_copy(update={"sources": sources})
    settings = config.settings.model_copy(update={"discovery": discovery})
    return config.model_copy(update={"settings": settings})


async def test_browser_only_skips_cxs(tmp_config) -> None:
    seen: list[str] = []

    async def runner(company: str, board: str) -> list[RawJobPosting]:
        seen.append(board)
        return [_posting("workday", "JR1", method="browser", url="https://nvidia.wd5.myworkdayjobs.com/job/X_JR1")]

    jobs = await collect_workday(
        _browser_config(tmp_config),
        None,
        browser_only=True,
        browser_company="NVIDIA",
        browser_runner=runner,
    )
    assert seen
    assert "myworkdayjobs.com" in seen[0]
    assert jobs[0].provenance["discovery_method"] == "browser"
    assert jobs[0].source == "workday"


async def test_cxs_still_requires_an_http_client(tmp_config) -> None:
    try:
        await collect_workday(_browser_config(tmp_config), None, browser_only=False)
    except SourceDiscoveryError as exc:
        assert "HTTP" in str(exc)
    else:
        raise AssertionError("CXS ran without an HTTP client")

"""Phase 10: fresh-preview diagnostics, gate fixtures, and career-page bounds."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.discovery_agent import _dedupe_raw
from src.graph.pipeline import build_graph
from src.models.job import DateSource, RawJobPosting, RejectionReason
from src.models.state import PipelineState, RejectedJob
from src.services.fresh_preview import build_fresh_preview
from src.services.production_report import render_pipeline_health
from src.services.roles import classify_role
from src.services.seniority import classify_seniority
from src.sources.base import FetchResult, SourceContext
from src.sources.company_career import CompanyCareerSource
from src.sources.workday import WorkdaySource
from src.utils.logging import get_logger
from src.utils.normalization import detect_employment_type, extract_experience_requirement, normalize_location
from src.models.config import CompanyConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "company_career"
NOW = datetime(2026, 9, 27, 0, 17, tzinfo=timezone.utc)
ADOBE = "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"


def _roles(tmp_config):
    return tmp_config.roles


def _ctx(config, http) -> SourceContext:
    return SourceContext(config=config, http=http, logger=get_logger("phase10"))


def _career(tmp_config, http) -> CompanyCareerSource:
    tmp_config.fixture_mode = False
    return CompanyCareerSource(_ctx(tmp_config, http))


class _Pages:
    def __init__(self, responses: list[FetchResult]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []
        self.renderer = type("R", (), {"enabled": False, "render_count": 0})()

    async def get_text(self, url: str, **kwargs):
        self.calls.append(url)
        return self.responses.pop(0)


def test_graph_order_keeps_freshness_after_location() -> None:
    edges = {(edge.source, edge.target) for edge in build_graph().get_graph().edges}
    assert ("discovery", "extraction") in edges
    assert ("extraction", "role") in edges
    assert ("role", "seniority") in edges
    assert ("seniority", "location") in edges
    assert ("location", "freshness") in edges
    assert ("freshness", "url") in edges
    assert ("qc", "intelligence") in edges


def test_equivalent_titles_pass_and_unrelated_titles_do_not(tmp_config) -> None:
    roles = _roles(tmp_config)
    positives = (
        "Software Engineer",
        "Software Engineer, New Grad",
        "Software Engineer II",
        "Backend Engineer",
        "Platform Software Engineer",
        "Infrastructure Software Engineer",
        "Product Engineer",
        "Systems Software Engineer",
        "AI Software Engineer",
        "ML Platform Engineer",
    )
    for title in positives:
        verdict = classify_role(title, "Design and implement production services.", roles)
        assert verdict.is_software_engineering and not verdict.needs_llm, title
    negatives = (
        "Data Scientist",
        "Research Scientist",
        "Product Manager",
        "Program Manager",
        "Account Manager",
        "Business Development Manager",
        "Sales Engineer",
        "Solutions Architect",
        "Engineering Manager",
        "Engineer",
    )
    for title in negatives:
        verdict = classify_role(title, "Own the roadmap and talk to customers.", roles)
        assert verdict.is_software_engineering is False, title


def test_phase9_fresh_preview_titles_are_role_rejections(tmp_config) -> None:
    """The seven Greenhouse jobs that were fresh at the Phase 9 run."""
    roles = _roles(tmp_config)
    titles = (
        "Senior Enterprise Account Manager",
        "Legislative Counsel",
        "Legislative Counsel",
        "Privacy Counsel",
        "Privacy Counsel",
        "Senior Marketing Manager II, Canada",
        "Engineering Manager",
    )
    for title in titles:
        verdict = classify_role(title, "Required qualifications are unrelated to software.", roles)
        assert verdict.is_software_engineering is False
        assert verdict.needs_llm is False


def test_experience_band_and_preferred_years(tmp_config) -> None:
    roles = _roles(tmp_config)

    def years(title: str, description: str):
        return classify_seniority(
            title,
            description,
            roles,
            max_required_years=2,
            ignore_preferred=True,
        )

    assert years("Software Engineer", "Required Qualifications\n0 years of experience.").fits_entry_level
    assert years("Software Engineer", "Required Qualifications\n1 year of experience.").fits_entry_level
    assert years("Software Engineer", "Required Qualifications\n2 years of experience.").fits_entry_level
    assert years("Software Engineer", "Required Qualifications\n0-2 years of experience.").fits_entry_level
    assert years("Software Engineer II", "Required Qualifications\n1+ years of experience.").fits_entry_level
    assert years("Software Engineer", "Required Qualifications\n2+ years of experience.").fits_entry_level
    assert years("Software Engineer", "Required Qualifications\n3+ years of experience.").fits_entry_level is False
    assert years("Software Engineer", "Required Qualifications\n4+ years of experience.").fits_entry_level is False
    assert years("Software Engineer", "Required Qualifications\n5+ years of experience.").fits_entry_level is False
    preferred = years(
        "Software Engineer",
        "Required Qualifications\n1+ years of experience.\n3+ years of experience preferred.",
    )
    assert preferred.fits_entry_level is True
    parsed = extract_experience_requirement(
        "3+ years of experience preferred.",
        title="Software Engineer",
    )
    assert parsed.min_years is None
    assert parsed.preferred_min_years == 3
    internal = years("Internal Software Engineer", "Build internal tools.")
    assert internal.fits_entry_level is False
    assert any("intern" == signal for signal in internal.signals) is False


def test_location_us_international_remote_and_multi(tmp_config) -> None:
    us_cases = (
        "United States",
        "US",
        "USA",
        "U.S.",
        "California, United States",
        "New York, NY",
        "Boston, MA",
        "Remote - United States",
        "Remote, US",
        "Boston, MA; New York, NY",
        "London, United Kingdom; Boston, MA",
        "United States; Berlin, Germany",
    )
    for raw in us_cases:
        info = normalize_location(raw)
        assert info.is_us is True, raw
        assert info.is_international_only is False, raw
    international = (
        "Canada",
        "United Kingdom",
        "India",
        "Germany",
        "Remote - Europe",
        "Remote - Canada",
    )
    for raw in international:
        info = normalize_location(raw)
        assert info.is_us is False, raw
        assert info.is_international_only is True, raw
    bare = normalize_location("Remote")
    assert bare.is_us is False
    assert bare.is_international_only is False
    compact = normalize_location("5 Locations")
    assert compact.is_us is False
    assert compact.is_international_only is False


def test_employment_types_keep_target_and_reject_others() -> None:
    assert detect_employment_type("Full-time").value == "Full-time"
    assert detect_employment_type("Full time").value == "Full-time"
    assert detect_employment_type("Contract").value == "Contract"
    assert detect_employment_type("Internship").value == "Internship"
    assert detect_employment_type("Co-op").value == "Co-op"
    assert detect_employment_type(None, title="Software Engineer Intern").value == "Internship"
    assert detect_employment_type("Part-time").value == "Part-time"
    assert detect_employment_type("Temporary").value == "Temporary"
    assert detect_employment_type("Volunteer").value == "Volunteer"


def test_fresh_preview_is_diagnostic_and_stage_aware(tmp_config) -> None:
    fresh_url = "https://boards.greenhouse.io/example/jobs/1"
    senior_url = "https://boards.greenhouse.io/example/jobs/2"
    # is_fresh measures age from the real clock. Keep the two fresh rows inside
    # 24 hours and the stale row outside it, without changing the funnel counts.
    moment = datetime.now(timezone.utc)
    stale = RawJobPosting(
        source="greenhouse",
        company_name="Example",
        title="Software Engineer",
        apply_url="https://boards.greenhouse.io/example/jobs/3",
        posted_at=moment - timedelta(days=10),
        date_source=DateSource.POSTED_DATE,
        description="private description that must not be copied",
    )
    fresh_role_mismatch = RawJobPosting(
        source="greenhouse",
        company_name="Airbnb",
        title="Senior Enterprise Account Manager",
        location_raw="Singapore",
        job_id="8232474",
        apply_url=fresh_url,
        posted_at=moment - timedelta(hours=2),
        date_source=DateSource.POSTED_DATE,
        description="private description that must not be copied",
    )
    fresh_senior = RawJobPosting(
        source="greenhouse",
        company_name="Example",
        title="Software Engineer",
        location_raw="Boston, MA",
        job_id="2",
        apply_url=senior_url,
        posted_at=moment - timedelta(hours=1),
        date_source=DateSource.POSTED_DATE,
        description="Required Qualifications\n5+ years of experience.",
    )
    state = PipelineState(config=tmp_config)
    state.raw_postings = [stale, fresh_role_mismatch, fresh_senior]
    state.rejected = [
        RejectedJob(
            company="Airbnb",
            title="Senior Enterprise Account Manager",
            reason=RejectionReason.ROLE,
            detail="management title is outside the software-engineering target",
            url=fresh_url,
            source="greenhouse",
        ),
        RejectedJob(
            company="Example",
            title="Software Engineer",
            reason=RejectionReason.SENIORITY,
            detail="required experience starts at 5.0 years (cap 2.0)",
            url=senior_url,
            source="greenhouse",
        ),
    ]
    before_jobs = list(state.jobs)
    audit = build_fresh_preview(state)
    assert state.jobs == before_jobs
    assert audit["count"] == 2
    assert audit["funnel"] == {
        "fresh_preview": 2,
        "fresh_after_role": 1,
        "fresh_after_seniority": 0,
        "fresh_after_location": 0,
        "fresh_after_employment": 0,
        "freshness_gate": 0,
    }
    assert audit["by_pipeline_reason"]["role"] == 1
    assert audit["by_pipeline_reason"]["seniority"] == 1
    blob = json.dumps(audit)
    assert "private description" not in blob
    assert "GEMINI_API_KEY" not in blob
    state.summary.discovery_profile = {
        "freshness_preview": {
            "fresh": 2,
            "stale": 1,
            "unknown": 0,
            "by_source": {"greenhouse": {"fresh": 2, "stale": 1, "unknown": 0}},
        },
        "fresh_preview_audit": audit,
        "by_source": {"greenhouse": {"seconds_sum": 4.0, "wall_seconds": 2.0, "jobs": 3}},
    }
    text = render_pipeline_health(state)
    assert "freshness_by_source: greenhouse fresh=2 stale=1 unknown=0" in text
    assert "freshness_stage_funnel: preview=2 after_role=1" in text
    rendered = state.summary.fresh_preview_report()
    assert "Senior Enterprise Account Manager" in rendered
    assert "private description" not in rendered


def test_dedupe_keeps_stronger_posted_timestamp_and_separate_ids() -> None:
    fresh = RawJobPosting(
        source="greenhouse",
        company_name="Example",
        title="Software Engineer",
        job_id="same",
        apply_url="https://example.com/jobs/1",
        posted_at=NOW - timedelta(hours=1),
        date_source=DateSource.POSTED_DATE,
        discovered_at=NOW,
    )
    stale = fresh.model_copy(
        update={
            "source": "company_career",
            "posted_at": NOW - timedelta(days=20),
            "posted_at_raw": "stale",
            "discovered_at": NOW,
        }
    )
    kept = _dedupe_raw([fresh, stale])
    assert len(kept) == 1
    assert kept[0].posted_at == fresh.posted_at
    assert "company_career" in kept[0].provenance["discovered_from"]

    unknown = RawJobPosting(
        source="workday",
        company_name="Example",
        title="Software Engineer",
        job_id="other",
        apply_url="https://example.com/jobs/2",
        discovered_at=NOW,
    )
    later = unknown.model_copy(
        update={
            "source": "greenhouse",
            "posted_at": NOW - timedelta(hours=3),
            "date_source": DateSource.POSTED_DATE,
            "discovered_at": NOW,
        }
    )
    merged = _dedupe_raw([unknown, later])
    assert len(merged) == 1
    assert merged[0].posted_at == later.posted_at
    assert merged[0].date_source is DateSource.POSTED_DATE
    assert merged[0].discovered_at == unknown.discovered_at

    first_stale = later.model_copy(update={"posted_at": NOW - timedelta(days=30), "job_id": "third", "apply_url": "https://example.com/jobs/3"})
    second_fresh = first_stale.model_copy(update={"posted_at": NOW - timedelta(hours=1), "source": "lever"})
    stayed = _dedupe_raw([first_stale, second_fresh])
    assert stayed[0].posted_at == first_stale.posted_at

    same_title_a = RawJobPosting(
        source="workday",
        company_name="NVIDIA",
        title="Software Engineer",
        job_id="A",
        apply_url="https://nvidia.wd5.myworkdayjobs.com/job/A",
    )
    same_title_b = same_title_a.model_copy(
        update={"job_id": "B", "apply_url": "https://nvidia.wd5.myworkdayjobs.com/job/B"}
    )
    assert len(_dedupe_raw([same_title_a, same_title_b])) == 2


@pytest.mark.asyncio
async def test_career_permanent_failures_and_nav_page_stop_early(tmp_config) -> None:
    denied = _Pages([
        FetchResult(url="https://example.com/careers", status=403, error="access denied"),
    ])
    result = await _career(tmp_config, denied).discover_result(
        CompanyConfig(name="Blocked", careers_url="https://example.com/careers")
    )
    assert result.success is False
    assert denied.calls == ["https://example.com/careers"]
    assert result.diagnostics.get("page_outcome") == "HTTP_FORBIDDEN"
    assert result.http_status == 403

    missing = _Pages([
        FetchResult(url="https://example.com/careers", status=404, text="", error="HTTP 404"),
    ])
    result = await _career(tmp_config, missing).discover_result(
        CompanyConfig(name="Missing", careers_url="https://example.com/careers")
    )
    assert result.success is False
    assert missing.calls == ["https://example.com/careers"]
    assert result.diagnostics.get("page_outcome") == "HTTP_NOT_FOUND"

    empty = _Pages([
        FetchResult(url="https://example.com/careers", status=200, text="   "),
    ])
    result = await _career(tmp_config, empty).discover_result(
        CompanyConfig(name="Empty", careers_url="https://example.com/careers")
    )
    assert result.success is True
    assert result.jobs == []
    assert empty.calls == ["https://example.com/careers"]
    assert result.diagnostics.get("page_outcome") == "EMPTY_PAGE"

    shell = _Pages([
        FetchResult(
            url="https://example.com/careers",
            status=200,
            text=(FIXTURES / "phase10_shell.html").read_text(encoding="utf-8"),
        ),
    ])
    result = await _career(tmp_config, shell).discover_result(
        CompanyConfig(name="Shell", careers_url="https://example.com/careers")
    )
    assert shell.calls == ["https://example.com/careers"]
    assert result.jobs == []
    assert result.diagnostics.get("page_outcome") == "JS_SHELL"

    nav = _Pages([
        FetchResult(
            url="https://careers.example.com/careers-home",
            status=200,
            text=(FIXTURES / "phase10_nav.html").read_text(encoding="utf-8"),
        ),
    ])
    result = await _career(tmp_config, nav).discover_result(
        CompanyConfig(name="Advanced Micro Devices", careers_url="https://careers.example.com/careers-home")
    )
    assert nav.calls == ["https://careers.example.com/careers-home"]
    assert result.jobs == []
    assert result.diagnostics.get("page_outcome") == "UNSUPPORTED_STRUCTURE"


@pytest.mark.asyncio
async def test_career_static_and_rendered_extraction(tmp_config) -> None:
    static = _Pages([
        FetchResult(
            url="https://example.com/careers",
            status=200,
            text=(FIXTURES / "phase10_static.html").read_text(encoding="utf-8"),
        ),
    ])
    result = await _career(tmp_config, static).discover_result(
        CompanyConfig(name="Static", careers_url="https://example.com/careers")
    )
    assert len(static.calls) == 1
    assert len(result.jobs) == 1
    assert result.jobs[0].title == "Software Engineer"
    assert result.jobs[0].posted_at is not None
    assert result.diagnostics.get("page_outcome") == "OK"

    class Renderer:
        enabled = True
        render_count = 0

        async def render(self, url: str) -> str:
            self.render_count += 1
            return (FIXTURES / "phase10_rendered.html").read_text(encoding="utf-8")

    shell = _Pages([
        FetchResult(
            url="https://example.com/app",
            status=200,
            text=(FIXTURES / "phase10_shell.html").read_text(encoding="utf-8"),
        ),
    ])
    shell.renderer = Renderer()
    result = await _career(tmp_config, shell).discover_result(
        CompanyConfig(name="Rendered", careers_url="https://example.com/app")
    )
    assert shell.renderer.render_count == 1
    assert len(shell.calls) == 1
    assert result.jobs[0].title == "Backend Engineer"
    assert result.diagnostics.get("page_outcome") == "OK"


@pytest.mark.asyncio
async def test_career_stage_budget_skips_detail_fetches(tmp_config) -> None:
    discovery = tmp_config.settings.discovery.model_copy(update={"career_stage_budget_seconds": 0.01})
    tmp_config.settings = tmp_config.settings.model_copy(update={"discovery": discovery})

    class Slow:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.renderer = type("R", (), {"enabled": False, "render_count": 0})()

        async def get_text(self, url: str, **kwargs):
            self.calls.append(url)
            await asyncio.sleep(0.05)
            return FetchResult(
                url=url,
                status=200,
                text='<html><body><a href="https://example.com/jobs/123/software-engineer">Software Engineer</a></body></html>',
            )

    http = Slow()
    result = await _career(tmp_config, http).discover_result(
        CompanyConfig(name="Slow", careers_url="https://example.com/careers")
    )
    assert http.calls == ["https://example.com/careers"]
    assert result.diagnostics.get("page_outcome") == "TIMEOUT"
    assert result.jobs == []


@pytest.mark.asyncio
async def test_uncapped_workday_board_does_not_partition(tmp_config) -> None:
    class Board:
        def __init__(self) -> None:
            self.bodies: list[dict] = []
            self.workday_post_count = 0

        async def request(self, method: str, url: str, **kwargs):
            body = dict(kwargs.get("json_body") or {})
            self.bodies.append(body)
            jobs = [
                {
                    "title": "Software Engineer",
                    "externalPath": "/job/swe",
                    "id": "R1",
                    "postedOn": "Posted 10 Days Ago",
                }
            ]
            return FetchResult(
                url=url,
                status=200,
                text=json.dumps({"total": 1, "jobPostings": jobs}),
            )

    tmp_config.fixture_mode = False
    http = Board()
    source = WorkdaySource(_ctx(tmp_config, http))
    company = CompanyConfig(name="Adobe", ats_type="workday", ats_identifier=ADOBE, careers_url=ADOBE)
    result = await source.discover_result(company)
    assert result.success is True
    assert len(result.jobs) == 1
    assert len(http.bodies) == 1
    assert http.bodies[0]["searchText"] == ""
    assert result.diagnostics.get("partitions_attempted", 0) == 0

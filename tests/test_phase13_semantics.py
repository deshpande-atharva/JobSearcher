"""Phase 13: role semantics, golden fresh jobs, and separate gate timings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.dedup_agent import run_dedup
from src.agents.discovery_agent import _dedupe_raw
from src.agents.freshness_agent import run_freshness
from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
from src.agents.location_agent import run_location_employment
from src.agents.qc_agent import run_quality_control
from src.agents.role_agent import run_role_classification
from src.agents.seniority_agent import run_seniority
from src.agents.url_agent import run_url_verification
from src.graph.pipeline import build_graph
from src.llm.base import NullLLMProvider
from src.models.job import DateSource, EmploymentType, RawJobPosting, RejectionReason
from src.models.state import PipelineState
from src.services.freshness import is_fresh
from src.services.rejection_codes import rejection_code
from src.services.roles import classify_role
from src.sources.workday import cxs_request_body
from src.utils.dates import utcnow
from tests.conftest import make_job
from tests.test_llm_reliability import ScriptLLM, _settings

NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)
WORK = "Write production code and run code review for backend services."


def _disable_visa(config) -> None:
    visa = config.settings.visa.model_copy(update={"enabled": False})
    config.settings = config.settings.model_copy(update={"visa": visa})


async def _qualify(config, job, llm=None):
    state = PipelineState(config=config, jobs=[job], resources={"llm": llm or NullLLMProvider()})
    await run_role_classification(state)
    await run_seniority(state)
    await run_location_employment(state)
    await run_freshness(state)
    await run_url_verification(state)
    await run_h1b_enrichment(state)
    await run_dedup(state)
    await run_quality_control(state)
    return state


def _fresh_job(**overrides):
    data = dict(
        job_title="Software Engineer",
        location="Boston, MA",
        description="Required Qualifications\n0-2 years of experience.\n" + WORK,
        posted_at=NOW - timedelta(hours=2),
        date_source=DateSource.POSTED_DATE,
        employment_type=EmploymentType.FULL_TIME,
    )
    data.update(overrides)
    return make_job(**data)


def test_graph_order_is_unchanged() -> None:
    edges = {(edge.source, edge.target) for edge in build_graph().get_graph().edges}
    assert ("seniority", "location") in edges
    assert ("location", "freshness") in edges
    assert ("freshness", "url") in edges
    assert ("qc", "intelligence") in edges


def test_clear_titles_and_ambiguous_families(tmp_config) -> None:
    roles = tmp_config.roles
    eligible = (
        "Software Engineer",
        "Software Engineer II",
        "Software Developer",
        "Backend Engineer",
        "Backend Software Engineer",
        "Frontend Engineer",
        "Full Stack Engineer",
        "Full-Stack Software Engineer",
        "Platform Engineer",
        "Infrastructure Engineer",
        "Systems Software Engineer",
        "Product Engineer",
        "AI Software Engineer",
        "ML Platform Engineer",
        "Junior Software Engineer",
        "Junior Backend Engineer",
        "Junior Platform Engineer",
        "DevOps Engineer",
        "Site Reliability Engineer",
        "SRE",
        "Cloud Engineer",
        "Release Engineer",
        "Build Engineer",
        "SDET",
        "Automation Engineer",
    )
    for title in eligible:
        verdict = classify_role(title, "", roles)
        assert verdict.is_software_engineering is True and verdict.needs_llm is False, title
    rejected = (
        "Account Manager",
        "Enterprise Account Manager",
        "Marketing Manager",
        "Product Manager",
        "Program Manager",
        "Project Manager",
        "Engineering Manager",
        "Research Scientist",
        "Data Scientist",
        "Counsel",
        "Privacy Counsel",
        "Legislative Counsel",
        "Business Development Manager",
        "Sales Engineer",
        "Solutions Architect",
        "DevOps Manager",
        "SRE Manager",
        "Platform Engineering Manager",
    )
    for title in rejected:
        verdict = classify_role(title, WORK + " Python SQL APIs machine learning.", roles)
        assert verdict.is_software_engineering is False and verdict.needs_llm is False, title

    # AI/ML/Data stay ambiguous until the description shows software work.
    for title in ("AI Engineer", "ML Engineer", "Machine Learning Engineer", "Data Engineer", "Analytics Engineer"):
        open_title = classify_role(title, "Python, SQL, and APIs.", roles)
        assert open_title.needs_llm is True and open_title.is_software_engineering is False, title
        promoted = classify_role(title, WORK, roles)
        assert promoted.is_software_engineering is True and promoted.needs_llm is False, title
    scientist = classify_role("Data Scientist", "Python, SQL, APIs, and machine learning. " + WORK, roles)
    assert scientist.is_software_engineering is False and scientist.needs_llm is False

    # QA titles are ambiguous. Two software-work signals decide them without an LLM.
    for title in ("QA Engineer", "Test Engineer"):
        unclear = classify_role(title, "Manual testing of business workflows.", roles)
        assert unclear.needs_llm is True and unclear.is_software_engineering is False, title
        clear = classify_role(title, WORK, roles)
        assert clear.is_software_engineering is True and clear.needs_llm is False, title

    # Solutions Engineer is a configured core product-engineer keyword.
    # The description does not override that title match.
    for description in (
        "Customer implementation and training.",
        "Technical integrations with customer systems.",
        WORK,
        "Pre-sales demos and proposals.",
        "Build APIs and production systems.",
    ):
        verdict = classify_role("Solutions Engineer", description, roles)
        assert verdict.is_software_engineering is True
        assert verdict.family == "PRODUCT_ENGINEER"
        assert verdict.needs_llm is False
    for title in ("Developer Advocate", "Technical Engineer"):
        verdict = classify_role(title, "", roles)
        assert verdict.needs_llm is True and verdict.is_software_engineering is False, title


def test_posted_date_beats_updated_and_duplicate_order() -> None:
    posted = RawJobPosting(
        source="greenhouse",
        company_name="Example",
        title="Software Engineer",
        job_id="1",
        apply_url="https://boards.greenhouse.io/example/jobs/1",
        posted_at=NOW - timedelta(hours=24),
        updated_at=NOW - timedelta(days=10),
        date_source=DateSource.POSTED_DATE,
        discovered_at=NOW,
    )
    assert is_fresh(posted, 24, now=NOW)[0] is True
    assert is_fresh(posted.model_copy(update={"posted_at": NOW - timedelta(hours=24, seconds=1)}), 24, now=NOW)[0] is False
    stale_posted_fresh_updated = posted.model_copy(
        update={"posted_at": NOW - timedelta(days=3), "updated_at": NOW - timedelta(hours=1)}
    )
    assert is_fresh(stale_posted_fresh_updated, 24, now=NOW)[0] is False
    assert is_fresh(RawJobPosting(source="greenhouse", company_name="Example", discovered_at=NOW), 24, now=NOW) == (False, None)
    fresh = posted.model_copy(update={"posted_at": NOW - timedelta(hours=1)})
    stale = fresh.model_copy(update={"posted_at": NOW - timedelta(days=4)})
    assert _dedupe_raw([stale, fresh])[0].posted_at == fresh.posted_at
    assert _dedupe_raw([fresh, stale])[0].posted_at == fresh.posted_at
    assert _dedupe_raw([fresh, stale])[0].discovered_at == fresh.discovered_at


def test_workday_public_body_is_unchanged() -> None:
    body = cxs_request_body(limit=20, offset=40, search_text="")
    assert list(body) == ["appliedFacets", "limit", "offset", "searchText"]
    assert body["appliedFacets"] == {}
    assert body["limit"] == 20


@pytest.mark.asyncio
async def test_golden_fresh_jobs_reach_final(tmp_config) -> None:
    _disable_visa(tmp_config)
    cases = (
        _fresh_job(job_id="g1", job_title="Software Engineer", location="Boston, MA"),
        _fresh_job(
            job_id="g2",
            job_title="Junior Backend Engineer",
            location="Remote - United States",
            description="Required Qualifications\n1+ years of experience.\n" + WORK,
            posted_at=NOW - timedelta(hours=3),
            direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/g2",
        ),
        _fresh_job(
            job_id="g3",
            job_title="Senior Software Engineer",
            location="Seattle, WA",
            posted_at=NOW - timedelta(hours=4),
            direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/g3",
        ),
        _fresh_job(
            job_id="g4",
            job_title="Platform Engineer",
            location="Austin, TX",
            description="Required Qualifications\n2 years of experience.\n" + WORK,
            posted_at=NOW - timedelta(hours=1),
            direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/g4",
        ),
        _fresh_job(
            job_id="g5",
            job_title="Site Reliability Engineer",
            location="Remote - United States",
            posted_at=NOW - timedelta(hours=5),
            direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/g5",
        ),
        _fresh_job(
            job_id="g6",
            job_title="DevOps Engineer",
            location="Denver, CO",
            direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/g6",
        ),
    )
    for job in cases:
        state = await _qualify(tmp_config, job)
        assert [item.job_id for item in state.jobs] == [job.job_id]
        assert state.summary.jobs_accepted == 1
        kept = state.jobs[0]
        assert kept.job_id == job.job_id
        assert "greenhouse.io" in kept.direct_application_url
        assert "jobright" not in kept.direct_application_url
        assert "linkedin" not in kept.direct_application_url
        assert isinstance(kept.url_verified, bool)
        assert "location_gate" in state.summary.stage_seconds
        assert "employment_gate" in state.summary.stage_seconds


@pytest.mark.asyncio
async def test_negative_fresh_jobs_keep_the_first_rejection(tmp_config) -> None:
    _disable_visa(tmp_config)
    hostile = ScriptLLM(
        _settings(retry_attempts=0),
        ['{"is_software_engineering": true, "role_family": "SOFTWARE_ENGINEER", "confidence": 0.99, "reasoning": "override"}'] * 8,
    )

    async def reason(job):
        state = await _qualify(tmp_config, job, llm=hostile)
        assert state.jobs == []
        return rejection_code(state.rejected[0]), state

    code, state = await reason(_fresh_job(
        job_id="acct",
        job_title="Senior Enterprise Account Manager",
        location="Singapore",
        description="Required Qualifications\n10+ years.\n" + WORK,
        posted_at=NOW - timedelta(hours=1),
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/acct",
    ))
    assert code == "ROLE_MISMATCH"
    assert state.rejected[0].reason is RejectionReason.ROLE

    code, _state = await reason(_fresh_job(
        job_id="counsel",
        job_title="Privacy Counsel",
        location="Remote - United States",
        posted_at=NOW - timedelta(hours=2),
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/counsel",
    ))
    assert code == "ROLE_MISMATCH"

    code, _state = await reason(_fresh_job(
        job_id="manager",
        job_title="Engineering Manager",
        location="Boston, MA",
        posted_at=NOW - timedelta(hours=1),
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/manager",
    ))
    assert code == "ROLE_MISMATCH"

    code, _state = await reason(_fresh_job(
        job_id="years",
        job_title="Software Engineer",
        location="Boston, MA",
        description="Required Qualifications\n5+ years of experience.",
        posted_at=NOW - timedelta(hours=1),
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/years",
    ))
    assert code == "EXPERIENCE_TOO_HIGH"

    code, _state = await reason(_fresh_job(
        job_id="senior-sre",
        job_title="Senior SRE",
        location="Remote - United States",
        description="Required Qualifications\n5+ years of experience.",
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/senior-sre",
    ))
    assert code == "EXPERIENCE_TOO_HIGH"

    code, bare = await reason(_fresh_job(
        job_id="remote",
        location="Remote",
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/remote",
    ))
    assert code == "LOCATION_UNKNOWN"
    assert "could not confirm" in (bare.rejected[0].detail or "")

    code, _state = await reason(_fresh_job(
        job_id="canada",
        location="Remote - Canada",
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/canada",
    ))
    assert code == "NON_US_LOCATION"

    code, _state = await reason(_fresh_job(
        job_id="stale",
        posted_at=utcnow() - timedelta(hours=25),
        direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/stale",
    ))
    assert code == "STALE_JOB"
    assert hostile.generated == 0


@pytest.mark.asyncio
async def test_clear_role_ignores_a_hostile_or_unavailable_model(tmp_config) -> None:
    _disable_visa(tmp_config)
    reject_all = ScriptLLM(
        _settings(retry_attempts=0, circuit_breaker_failures=1, circuit_reset_seconds=999),
        ['{"is_software_engineering": false, "role_family": "NOT_SOFTWARE", "confidence": 0.99, "reasoning": "no"}'],
    )
    state = await _qualify(tmp_config, _fresh_job(job_id="kept"), llm=reject_all)
    assert [job.job_id for job in state.jobs] == ["kept"]
    assert reject_all.generated == 0
    unavailable = await _qualify(tmp_config, _fresh_job(job_id="kept-2", direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/kept-2"))
    assert unavailable.summary.jobs_accepted == 1

"""Greenhouse and Workday share one discovery and qualification path."""

from __future__ import annotations

from datetime import timedelta

from src.main import build_parser
from src.models.job import DateSource, RawJobPosting, RejectionReason, VisaSponsorshipStatus
from src.pilot.workday_target import browser_discovery_enabled
from src.services.candidate_profile import CandidateProfile, SkillEvidence
from src.services.discovery_orchestrator import ensure_provenance, orchestrate
from src.utils.dates import utcnow

COMPANY = "Example Robotics"
GH = "https://boards.greenhouse.io/example/jobs/{job_id}"


def _profile() -> CandidateProfile:
    return CandidateProfile(
        profile_version=1,
        professional_months=18,
        max_required_years=2,
        require_us_location=True,
        employment_types=["Full-time"],
        education=["Bachelors in Electronics and Telecommunication"],
        experience_evidence=[
            SkillEvidence(skill="Java", evidence="Developed backend services using Java.", status="EXPLICIT"),
            SkillEvidence(skill="PostgreSQL", evidence="Built REST APIs with PostgreSQL.", status="EXPLICIT"),
        ],
    )


def _posting(
    source: str,
    job_id: str,
    *,
    title: str = "Software Engineer",
    location: str = "Boston, MA",
    url: str | None = None,
    description: str | None = None,
    posted_at=None,
    date_source: DateSource = DateSource.POSTED_DATE,
    method: str = "api",
) -> RawJobPosting:
    now = utcnow()
    text = description or (
        "Minimum Qualifications\n"
        "0-2 years of experience.\n"
        "BS in Computer Science or Engineering.\n"
        "Java and PostgreSQL required.\n"
    )
    return RawJobPosting(
        source=source,
        company_name=COMPANY,
        title=title,
        location_raw=location,
        job_id=job_id,
        apply_url=url if url is not None else GH.format(job_id=job_id),
        employment_type_raw="Full-time",
        description=text,
        posted_at=now - timedelta(hours=2) if posted_at is None and date_source is not DateSource.UNKNOWN else posted_at,
        date_source=date_source,
        provenance={
            "discovery_method": method,
            "source_job_id": job_id,
            "discovered_from": [source],
        },
    )


def _disable_workday(config):
    workday = config.settings.discovery.sources.workday.model_copy(update={"enabled": False, "browser_enabled": False})
    sources = config.settings.discovery.sources.model_copy(update={"workday": workday})
    discovery = config.settings.discovery.model_copy(update={"sources": sources})
    settings = config.settings.model_copy(update={"discovery": discovery})
    return config.model_copy(update={"settings": settings})


def test_orchestration_flags_stay_opt_in(tmp_config) -> None:
    args = build_parser().parse_args(["--multi-source-smoke-test", "--sources", "greenhouse,workday", "--dry-run", "--no-email"])
    assert args.multi_source_smoke_test is True
    assert args.no_email is True
    assert tmp_config.settings.discovery.sources.greenhouse.enabled is True
    assert tmp_config.settings.discovery.sources.workday.browser_enabled is False
    assert tmp_config.settings.discovery.sources.jobright.enabled is False
    assert browser_discovery_enabled(tmp_config) is False


def test_provenance_keeps_source_identity() -> None:
    posting = ensure_provenance(
        _posting("workday", "JR2024713", method="browser", url="https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/X_JR2024713")
    )
    assert posting.provenance["discovery_method"] == "browser"
    assert posting.provenance["source_job_id"] == "JR2024713"
    assert posting.provenance["source_url"]
    assert "workday" in posting.provenance["discovered_from"]
    assert posting.discovered_at is not None


async def test_cross_source_fixture_dedupes_without_merging_distinct_jobs(tmp_config) -> None:
    now = utcnow()
    duplicate_url = GH.format(job_id="123")
    greenhouse = [
        _posting("greenhouse", "123", url=duplicate_url, method="structured"),
        _posting("greenhouse", "1001"),
        _posting("greenhouse", "1002"),
        _posting("greenhouse", "1003", location="New York, NY"),
        _posting("greenhouse", "1004", posted_at=now - timedelta(days=40)),
        _posting("greenhouse", "1005", location="London, United Kingdom"),
        _posting(
            "greenhouse",
            "1006",
            title="Senior Software Engineer",
            description="Minimum Qualifications\n5+ years of experience.\nJava required.\n",
        ),
        _posting("greenhouse", "1007", posted_at=None, date_source=DateSource.UNKNOWN),
        RawJobPosting(source="greenhouse", company_name=COMPANY, title=None, apply_url=None),
        "not-a-job",
    ]

    async def greenhouse_jobs():
        return greenhouse

    async def workday_jobs():
        return [
            _posting("workday", "123", url=duplicate_url, method="cxs"),
        ]

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse_jobs, "workday": workday_jobs},
        profile=_profile(),
    )
    assert result.profile_loads == 1
    assert result.sources["greenhouse"].status == "PARTIAL"
    assert result.sources["workday"].status == "SUCCESS"
    assert result.sources["workday"].browser_enabled is False
    ids = [job.job_id for job in result.funnel.state.jobs]
    assert ids.count("123") == 1
    assert "1001" in ids and "1002" in ids
    assert "1003" in ids
    assert "1004" not in ids
    assert "1005" not in ids
    assert "1006" not in ids
    assert "1007" not in ids
    kept = next(job for job in result.funnel.state.jobs if job.job_id == "123")
    assert kept.source == "greenhouse"
    assert "workday" in kept.additional_sources
    assert kept.provenance["discovery_method"] == "structured"
    assert kept.provenance["source_job_id"] == "123"
    assert kept.visa_sponsorship_status is VisaSponsorshipStatus.UNKNOWN
    reasons = {item.reason for item in result.funnel.state.rejected}
    assert RejectionReason.FRESHNESS in reasons
    assert RejectionReason.LOCATION in reasons
    assert RejectionReason.SENIORITY in reasons
    assert RejectionReason.EXTRACTION_FAILED in reasons
    assert any(failure.stage == "normalize" for failure in result.sources["greenhouse"].failures)
    assert result.intelligence_evaluated == len(result.funnel.state.jobs)
    assert result.critic_reviewed == result.intelligence_evaluated
    assert result.llm_calls == 0
    assert all(critique.unsupported_claims == [] for critique in result.fits)


async def test_one_source_failure_does_not_drop_the_other(tmp_config) -> None:
    async def broken():
        raise RuntimeError("api_key=supersecret greenhouse timeout")

    async def workday_jobs():
        return [_posting("workday", "JR1", method="cxs", url="https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Boston/Software-Engineer_JR1")]

    failed_greenhouse = await orchestrate(
        tmp_config,
        collectors={"greenhouse": broken, "workday": workday_jobs},
        profile=_profile(),
    )
    assert failed_greenhouse.sources["greenhouse"].status == "FAILED"
    assert failed_greenhouse.sources["greenhouse"].discovery == "FAIL"
    assert failed_greenhouse.sources["workday"].status == "SUCCESS"
    assert [job.job_id for job in failed_greenhouse.funnel.state.jobs] == ["JR1"]
    message = failed_greenhouse.sources["greenhouse"].failures[0].message
    assert "supersecret" not in message
    assert failed_greenhouse.sources["greenhouse"].failures[0].error_type == "RuntimeError"

    async def greenhouse_jobs():
        return [_posting("greenhouse", "2001")]

    async def broken_workday():
        raise TimeoutError("workday cxs timed out")

    failed_workday = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse_jobs, "workday": broken_workday},
        profile=_profile(),
    )
    assert failed_workday.sources["workday"].status == "FAILED"
    assert failed_workday.sources["workday"].timeouts == 1
    assert [job.job_id for job in failed_workday.funnel.state.jobs] == ["2001"]

    async def boom():
        raise ConnectionError("down")

    both = await orchestrate(tmp_config, collectors={"greenhouse": boom, "workday": boom})
    assert both.sources["greenhouse"].status == "FAILED"
    assert both.sources["workday"].status == "FAILED"
    assert both.funnel.final_count == 0
    assert both.sources["greenhouse"].failures[0].stage == "discovery"


async def test_disabled_workday_is_not_called_and_browser_stays_off(tmp_config) -> None:
    called = []

    async def workday_jobs():
        called.append("workday")
        return []

    async def browser(company, board):
        called.append("browser")
        return []

    result = await orchestrate(
        _disable_workday(tmp_config),
        collectors={"greenhouse": _empty, "workday": workday_jobs},
        browser_runner=browser,
    )
    assert called == []
    assert result.sources["workday"].status == "DISABLED"
    assert result.sources["workday"].browser_enabled is False
    assert result.sources["greenhouse"].status == "EMPTY"
    assert result.sources["greenhouse"].discovered == 0


async def _empty():
    return []


async def test_h1b_lookup_failure_keeps_the_job(tmp_config) -> None:
    class _Down:
        async def lookup(self, company, aliases=()):
            raise RuntimeError("H1BGrader unavailable")

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": _empty, "workday": _one_workday},
        profile=_profile(),
        h1b=_Down(),
    )
    assert [job.job_id for job in result.funnel.state.jobs] == ["JR9"]
    job = result.funnel.state.jobs[0]
    assert job.visa_sponsorship_status is VisaSponsorshipStatus.UNKNOWN
    assert result.funnel.state.summary.h1b_lookup_failures == 1


async def _one_workday():
    return [
        _posting(
            "workday",
            "JR9",
            method="cxs",
            url="https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Boston/Software-Engineer_JR9",
        )
    ]


async def test_jobright_is_not_added(tmp_config) -> None:
    called = []

    async def jobright():
        called.append("jobright")
        return []

    result = await orchestrate(tmp_config, sources=("jobright",), collectors={"jobright": jobright})
    assert called == []
    assert result.sources["jobright"].status == "DISABLED"
    assert result.sources["jobright"].discovery == "DISABLED"

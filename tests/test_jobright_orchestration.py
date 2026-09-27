"""Jobright is a discovery source. Official URLs are the only final links."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from src.main import build_parser
from src.models.job import DateSource, RawJobPosting, RejectionReason
from src.services.discovery_orchestrator import orchestrate
from src.sources.base import SourceContext
from src.sources.jobright import JobrightSource, _company_selected, ambiguous_jobright_date, parse_jobright_cards
from src.utils.dates import utcnow
from src.utils.logging import get_logger

GREENHOUSE = "https://boards.greenhouse.io/example/jobs/123"
BODY = (
    "Minimum Qualifications\n"
    "0-2 years of experience.\n"
    "BS in Computer Science or Engineering.\n"
    "Java and PostgreSQL required.\n"
    "Full-time software engineer building backend services.\n"
)


def _enable(config):
    jobright = config.settings.discovery.sources.jobright.model_copy(update={"enabled": True})
    sources = config.settings.discovery.sources.model_copy(update={"jobright": jobright})
    discovery = config.settings.discovery.model_copy(update={"sources": sources})
    settings = config.settings.model_copy(update={"discovery": discovery})
    return config.model_copy(update={"settings": settings})


def _source(config) -> JobrightSource:
    return JobrightSource(SourceContext(config=config, http=None, logger=get_logger("jobright-test")))


def _entry(source: JobrightSource, **fields) -> RawJobPosting:
    payload = {
        "company": "Example Corp",
        "title": "Software Engineer",
        "location": "Boston, MA",
        "employmentType": "Full-time",
        "postedAt": "1 hour ago",
        "description": BODY,
        "jobId": "abc",
        "applyUrl": GREENHOUSE,
    }
    payload.update(fields)
    posting = source._entry_to_posting(payload, page_url="https://jobright.ai/entry-level-jobs")
    assert posting is not None
    return posting


def _greenhouse(job_id: str = "123", **kwargs) -> RawJobPosting:
    return RawJobPosting(
        source="greenhouse",
        company_name=kwargs.get("company", "Example Corp"),
        title=kwargs.get("title", "Software Engineer"),
        location_raw=kwargs.get("location", "Boston, MA"),
        job_id=job_id,
        apply_url=kwargs.get("url", f"https://boards.greenhouse.io/example/jobs/{job_id}"),
        employment_type_raw="Full-time",
        description=kwargs.get("description", BODY),
        posted_at=kwargs.get("posted_at", utcnow() - timedelta(hours=2)),
        date_source=kwargs.get("date_source", DateSource.POSTED_DATE),
        provenance={"discovery_method": "structured", "discovered_from": ["greenhouse"], "source_job_id": job_id},
    )


def test_public_card_keeps_jobright_id_and_unlabeled_time_unknown(tmp_config) -> None:
    html = """
    <a href="https://jobright.ai/jobs/info/abc123?utm_source=1146">
      <div>NVIDIA</div><div>2 hours ago</div>
      <div>Software Engineer</div><div>United States</div><div>Remote</div>
      <div>Build backend services in Java and PostgreSQL for a full-time role with 0-2 years of experience.</div>
    </a>
    """
    card = parse_jobright_cards(html)[0]
    posting = _source(tmp_config)._entry_to_posting(card, page_url="https://jobright.ai/remote-jobs/software-engineering")
    assert posting is not None
    assert posting.company_name == "NVIDIA"
    assert posting.title == "Software Engineer"
    assert posting.job_id is None
    assert posting.provenance["source_job_id"] == "abc123"
    assert posting.date_source is DateSource.UNKNOWN
    assert posting.posted_at is None
    assert posting.apply_url == "https://jobright.ai/jobs/info/abc123"
    assert posting.provenance["official_source"] == ""


def test_jobright_smoke_flag_parses() -> None:
    args = build_parser().parse_args(["--jobright-smoke-test", "--dry-run", "--no-email"])
    assert args.jobright_smoke_test is True
    assert args.dry_run is True
    assert args.no_email is True


def test_ambiguous_jobright_dates_are_not_posted(tmp_config) -> None:
    source = _source(tmp_config)
    fresh = _entry(source)
    assert fresh.job_id == "123"
    assert fresh.provenance["source_job_id"] == "abc"
    assert fresh.provenance["discovery_method"] == "jobright"
    assert fresh.date_source is DateSource.POSTED_DATE
    assert "jobright.ai" not in (fresh.apply_url or "")

    ambiguous = _entry(source, postedAt="found today", jobId="jr-ambiguous", applyUrl=GREENHOUSE)
    assert ambiguous.date_source is DateSource.UNKNOWN
    assert ambiguous.posted_at is None
    assert ambiguous_jobright_date("listed today")
    assert ambiguous_jobright_date("discovered today")

    aggregator = _entry(source, jobId="jr-only", applyUrl="https://jobright.ai/jobs/jr-only")
    assert aggregator.job_id is None
    assert aggregator.provenance["source_job_id"] == "jr-only"
    assert "jobright.ai" in (aggregator.apply_url or "")


def test_live_company_filter_respects_enabled_companies(tmp_config) -> None:
    companies = tuple(
        company.model_copy(update={"enabled": False}) if company.name == "Northwind Labs" else company
        for company in tmp_config.universe.companies
    )
    universe = tmp_config.universe.model_copy(update={"companies": companies})
    live = tmp_config.model_copy(update={"fixture_mode": False, "universe": universe})
    selected = RawJobPosting(
        source="jobright",
        company_name="Acme Robotics",
        title="Software Engineer",
        apply_url="https://example.com",
    )
    assert _company_selected(selected, live) is True
    assert _company_selected(selected.model_copy(update={"company_name": "Example Corp"}), live) is False
    assert _company_selected(selected.model_copy(update={"company_name": "Northwind Labs"}), live) is False


def test_result_cap_is_recorded(tmp_config) -> None:
    source = _source(tmp_config)
    first = _entry(source, jobId="a", applyUrl="https://boards.greenhouse.io/example/jobs/111")
    second = _entry(source, jobId="b", applyUrl="https://boards.greenhouse.io/example/jobs/222")
    kept = source._limit([first, second], 1, "listing")
    assert len(kept) == 1
    assert source.cap_reached is True
    assert "listing_cap=1" in source.cap_note


def test_fixture_mode_still_runs_jobright_when_production_flag_is_off(tmp_config) -> None:
    assert tmp_config.fixture_mode is True
    assert tmp_config.settings.discovery.sources.jobright.enabled is False
    assert _source(tmp_config).enabled is True


async def test_jobright_and_greenhouse_collapse_to_one_official_job(tmp_config) -> None:
    source = _source(tmp_config)
    discovered = _entry(source, jobId="abc", applyUrl="https://jobright.ai/jobs/abc?url=https://boards.greenhouse.io/example/jobs/123")

    async def greenhouse():
        return [_greenhouse()]

    async def jobright():
        return [discovered]

    result = await orchestrate(
        _enable(tmp_config),
        sources=("greenhouse", "jobright"),
        collectors={"greenhouse": greenhouse, "jobright": jobright},
    )
    assert len(result.postings) == 1
    posting = result.postings[0]
    assert posting.job_id == "123"
    assert set(posting.provenance["discovered_from"]) == {"greenhouse", "jobright"}
    assert "boards.greenhouse.io" in (posting.apply_url or "")
    jobs = result.funnel.state.jobs
    assert len(jobs) == 1
    assert jobs[0].direct_application_url == GREENHOUSE
    assert "jobright.ai" not in jobs[0].direct_application_url
    assert "linkedin.com" not in jobs[0].direct_application_url


async def test_jobright_fixtures_share_the_qualification_pipeline(tmp_config) -> None:
    source = _source(tmp_config)
    fresh = _entry(source, jobId="jr-fresh", applyUrl="https://boards.greenhouse.io/example/jobs/2001")
    stale = _entry(source, jobId="jr-stale", applyUrl="https://boards.greenhouse.io/example/jobs/2002", postedAt="40 days ago")
    senior = _entry(
        source,
        jobId="jr-senior",
        title="Senior Software Engineer",
        applyUrl="https://boards.greenhouse.io/example/jobs/2003",
        description="Minimum Qualifications\n5+ years of experience required.\nJava and PostgreSQL.\n",
    )
    international = _entry(
        source,
        jobId="jr-intl",
        applyUrl="https://boards.greenhouse.io/example/jobs/2004",
        location="London, United Kingdom",
    )
    other_id = _entry(source, jobId="jr-other", applyUrl="https://boards.greenhouse.io/example/jobs/456")
    other_location = _entry(
        source,
        jobId="jr-nyc",
        applyUrl="https://boards.greenhouse.io/example/jobs/789",
        location="New York, NY",
    )
    no_official = _entry(source, jobId="jr-none", applyUrl="https://www.linkedin.com/jobs/view/999")
    ambiguous = _entry(source, jobId="jr-date", applyUrl="https://boards.greenhouse.io/example/jobs/321", postedAt="discovered today")

    async def jobright():
        return [fresh, stale, senior, international, other_id, other_location, no_official, ambiguous, "not-a-job"]

    result = await orchestrate(_enable(tmp_config), sources=("jobright",), collectors={"jobright": jobright})
    report = result.sources["jobright"]
    assert report.status == "PARTIAL"
    ids = {posting.job_id for posting in result.postings}
    assert ids == {"2001", "2002", "2003", "2004", "456", "789", "321", None}
    reasons = {item.reason for item in result.funnel.state.rejected}
    assert RejectionReason.FRESHNESS in reasons
    assert RejectionReason.SENIORITY in reasons
    assert RejectionReason.LOCATION in reasons
    assert RejectionReason.INVALID_URL in reasons
    finals = result.funnel.state.jobs
    assert finals
    assert all("jobright.ai" not in (job.direct_application_url or "") for job in finals)
    assert all("linkedin.com" not in (job.direct_application_url or "") for job in finals)
    assert any(job.job_id == "2001" and "boards.greenhouse.io" in job.direct_application_url for job in finals)


async def test_jobright_failure_does_not_stop_greenhouse(tmp_config) -> None:
    async def greenhouse():
        return [_greenhouse()]

    async def jobright():
        raise RuntimeError("jobright listing failed")

    result = await orchestrate(
        _enable(tmp_config),
        sources=("greenhouse", "jobright"),
        collectors={"greenhouse": greenhouse, "jobright": jobright},
    )
    assert result.sources["jobright"].status == "FAILED"
    assert result.sources["greenhouse"].status == "SUCCESS"
    assert result.postings[0].source == "greenhouse"


async def test_greenhouse_failure_does_not_stop_jobright(tmp_config) -> None:
    source = _source(tmp_config)

    async def greenhouse():
        raise RuntimeError("greenhouse down")

    async def jobright():
        return [_entry(source)]

    result = await orchestrate(
        _enable(tmp_config),
        sources=("greenhouse", "jobright"),
        collectors={"greenhouse": greenhouse, "jobright": jobright},
    )
    assert result.sources["greenhouse"].status == "FAILED"
    assert result.postings
    assert result.postings[0].source == "jobright"
    assert result.postings[0].apply_url == GREENHOUSE


def test_jobright_adapter_has_no_evasion_or_apply_path() -> None:
    text = Path("src/sources/jobright.py").read_text(encoding="utf-8").lower()
    for banned in ("stealth", "fingerprint", "proxy", "page.evaluate", "subprocess", "submit application"):
        assert banned not in text

"""Lever and Ashby are authoritative ATS sources on the shared pipeline."""

from __future__ import annotations

from pathlib import Path

from src.main import build_parser
from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting, RejectionReason
from src.services.discovery_orchestrator import orchestrate
from src.services.seniority import classify_seniority
from src.sources.ashby import AshbySource
from src.sources.base import SourceContext
from src.sources.lever import LeverSource
from src.utils.logging import get_logger

LEVER_URL = "https://jobs.lever.co/example/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
ASHBY_URL = "https://jobs.ashbyhq.com/example/bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
BODY = (
    "Minimum Qualifications\n"
    "0-2 years of experience.\n"
    "BS in Computer Science or Engineering.\n"
    "Java and PostgreSQL required.\n"
    "Full-time software engineer building backend services.\n"
    "Nice to have\n"
    "5+ years of experience.\n"
)


def _source(cls, config):
    return cls(SourceContext(config=config, http=None, logger=get_logger("board-test")))


def _company(name: str, ats: str) -> CompanyConfig:
    return CompanyConfig(name=name, ats_type=ats, ats_identifier="example")


def _lever(config, **fields) -> RawJobPosting:
    payload = {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "text": "Software Engineer",
        "hostedUrl": LEVER_URL,
        "applyUrl": LEVER_URL + "/apply",
        "categories": {"location": "Boston, MA", "commitment": "Full-time"},
        "workplaceType": "hybrid",
        "createdAt": "1 hour ago",
        "descriptionPlain": BODY,
        "lists": [
            {"text": "Minimum Qualifications", "content": "0-2 years of experience. Java and PostgreSQL."},
            {"text": "Nice to have", "content": "5+ years of experience is a plus."},
        ],
    }
    payload.update(fields)
    posting = _source(LeverSource, config)._to_posting(payload, _company("Example Corp", "lever"), "example")
    assert posting is not None
    return posting


def _ashby(config, **fields) -> RawJobPosting:
    payload = {
        "id": "bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
        "title": "Software Engineer",
        "jobUrl": ASHBY_URL,
        "applyUrl": ASHBY_URL + "/application",
        "location": "Boston, MA",
        "employmentType": "FullTime",
        "isListed": True,
        "publishedAt": "1 hour ago",
        "descriptionPlain": BODY,
    }
    payload.update(fields)
    posting = _source(AshbySource, config)._to_posting(payload, _company("Example Corp", "ashby"), "example")
    assert posting is not None
    return posting


def test_board_smoke_flags_parse() -> None:
    lever = build_parser().parse_args(["--lever-smoke-test", "--dry-run", "--no-email"])
    ashby = build_parser().parse_args(["--ashby-smoke-test", "--dry-run", "--no-email"])
    assert lever.lever_smoke_test is True
    assert ashby.ashby_smoke_test is True
    assert lever.dry_run is True and lever.no_email is True


def test_lever_official_url_id_and_ambiguous_date(tmp_config) -> None:
    fresh = _lever(tmp_config)
    assert fresh.job_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert fresh.provenance["official_job_id"] == fresh.job_id
    assert fresh.provenance["source_job_id"] == fresh.job_id
    assert fresh.provenance["discovery_method"] == "api"
    assert fresh.apply_url == LEVER_URL
    assert "Minimum Qualifications" in (fresh.description or "")
    assert fresh.date_source is DateSource.POSTED_DATE
    ambiguous = _lever(tmp_config, createdAt="found today")
    assert ambiguous.date_source is DateSource.UNKNOWN
    assert ambiguous.posted_at is None
    aggregator = _lever(tmp_config, hostedUrl="https://www.linkedin.com/jobs/view/999", applyUrl=None, id="lever-only")
    assert aggregator.job_id is None
    assert aggregator.provenance["official_url"] == ""
    verdict = classify_seniority(
        "Software Engineer",
        fresh.description,
        tmp_config.roles,
        max_required_years=2,
        ignore_preferred=True,
    )
    assert verdict.fits_entry_level is True


def test_ashby_official_url_and_unlisted_skip(tmp_config) -> None:
    fresh = _ashby(tmp_config)
    assert fresh.apply_url == ASHBY_URL
    assert fresh.provenance["official_source"] == "ashby"
    assert fresh.job_id == "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    hidden = _source(AshbySource, tmp_config)._to_posting(
        {"title": "Software Engineer", "jobUrl": ASHBY_URL, "isListed": False, "id": "hidden"},
        _company("Example Corp", "ashby"),
        "example",
    )
    assert hidden is None
    ambiguous = _ashby(tmp_config, publishedAt="listed today")
    assert ambiguous.posted_at is None


def test_board_job_cap_is_recorded(tmp_config) -> None:
    lever_settings = tmp_config.settings.discovery.sources.lever.model_copy(update={"max_jobs": 1})
    sources = tmp_config.settings.discovery.sources.model_copy(update={"lever": lever_settings})
    discovery = tmp_config.settings.discovery.model_copy(update={"sources": sources})
    settings = tmp_config.settings.model_copy(update={"discovery": discovery})
    config = tmp_config.model_copy(update={"settings": settings})
    source = _source(LeverSource, config)
    kept = source._limit([_lever(config), _lever(config, id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeef", text="Backend Engineer")])
    assert len(kept) == 1
    assert source.cap_reached is True


async def test_lever_fixtures_share_qualification_and_collapse_duplicates(tmp_config) -> None:
    fresh = _lever(tmp_config)
    duplicate = _lever(tmp_config)
    stale = _lever(tmp_config, id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeee01", createdAt="5 days ago", hostedUrl=LEVER_URL.replace("eeeeeeeeeeee", "eeeeeeeeee01"))
    senior = _lever(
        tmp_config,
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeee02",
        text="Senior Software Engineer",
        hostedUrl=LEVER_URL.replace("eeeeeeeeeeee", "eeeeeeeeee02"),
        descriptionPlain="Minimum Qualifications\n5+ years of experience required.\n",
        lists=[],
    )
    international = _lever(
        tmp_config,
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeee03",
        hostedUrl=LEVER_URL.replace("eeeeeeeeeeee", "eeeeeeeeee03"),
        categories={"location": "London, United Kingdom", "commitment": "Full-time"},
    )
    other_location = _lever(
        tmp_config,
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeee04",
        text="Software Engineer",
        hostedUrl=LEVER_URL.replace("eeeeeeeeeeee", "eeeeeeeeee04"),
        categories={"location": "New York, NY", "commitment": "Full-time"},
    )
    bad_url = _lever(
        tmp_config,
        id="not-official",
        hostedUrl="https://www.linkedin.com/jobs/view/42",
        applyUrl=None,
    )
    ambiguous = _lever(
        tmp_config,
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeee05",
        createdAt="discovered today",
        hostedUrl=LEVER_URL.replace("eeeeeeeeeeee", "eeeeeeeeee05"),
    )

    async def lever():
        return [fresh, duplicate, stale, senior, international, other_location, bad_url, ambiguous]

    result = await orchestrate(tmp_config, sources=("lever",), collectors={"lever": lever})
    same_id = [posting for posting in result.postings if posting.job_id == fresh.job_id]
    assert len(same_id) == 1
    titles = {posting.job_id for posting in result.postings if posting.title == "Software Engineer" and posting.job_id}
    assert len(titles) >= 2
    reasons = {item.reason for item in result.funnel.state.rejected}
    assert RejectionReason.FRESHNESS in reasons
    assert RejectionReason.SENIORITY in reasons
    assert RejectionReason.LOCATION in reasons
    assert RejectionReason.INVALID_URL in reasons
    finals = result.funnel.state.jobs
    assert any(job.job_id == fresh.job_id and job.direct_application_url == LEVER_URL for job in finals)
    assert all("jobright.ai" not in (job.direct_application_url or "") for job in finals)
    assert all("linkedin.com" not in (job.direct_application_url or "") for job in finals)


async def test_ashby_fixtures_and_cross_source_identity(tmp_config) -> None:
    fresh = _ashby(tmp_config)
    duplicate = _ashby(tmp_config)
    other = _ashby(
        tmp_config,
        id="bbbbbbbb-cccc-dddd-eeee-ffffffffff01",
        title="Software Engineer",
        jobUrl=ASHBY_URL.replace("ffffffffffff", "ffffffffff01"),
        location="Austin, TX",
    )

    async def ashby():
        return [fresh, duplicate, other]

    async def greenhouse():
        return [
            RawJobPosting(
                source="greenhouse",
                company_name="Example Corp",
                title="Software Engineer",
                location_raw="Boston, MA",
                job_id=fresh.job_id,
                apply_url=ASHBY_URL,
                employment_type_raw="Full-time",
                description=BODY,
                posted_at=fresh.posted_at,
                date_source=DateSource.POSTED_DATE,
                provenance={"discovery_method": "structured", "discovered_from": ["greenhouse"], "source_job_id": fresh.job_id},
            )
        ]

    result = await orchestrate(
        tmp_config,
        sources=("greenhouse", "ashby"),
        collectors={"greenhouse": greenhouse, "ashby": ashby},
    )
    matched = [posting for posting in result.postings if posting.job_id == fresh.job_id]
    assert len(matched) == 1
    assert set(matched[0].provenance["discovered_from"]) >= {"greenhouse", "ashby"}
    assert "ashbyhq.com" in (matched[0].apply_url or "")
    distinct = [posting for posting in result.postings if posting.title == "Software Engineer"]
    assert len(distinct) == 2
    assert any(job.direct_application_url == ASHBY_URL for job in result.funnel.state.jobs)


async def test_lever_failure_does_not_stop_ashby(tmp_config) -> None:
    async def lever():
        raise RuntimeError("lever board failed")

    async def ashby():
        return [_ashby(tmp_config)]

    result = await orchestrate(
        tmp_config,
        sources=("lever", "ashby"),
        collectors={"lever": lever, "ashby": ashby},
    )
    assert result.sources["lever"].status == "FAILED"
    assert result.sources["ashby"].status == "SUCCESS"
    assert result.postings[0].apply_url == ASHBY_URL


def test_board_adapters_have_no_evasion_or_apply_path() -> None:
    for name in ("src/sources/lever.py", "src/sources/ashby.py", "src/services/board_priority.py"):
        text = Path(name).read_text(encoding="utf-8").lower()
        for banned in ("stealth", "fingerprint", "proxy", "page.evaluate", "subprocess", "submit application", "captcha"):
            assert banned not in text


def test_production_board_settings_stay_bounded(tmp_config) -> None:
    sources = tmp_config.settings.discovery.sources
    assert sources.jobright.enabled is False
    assert sources.workday.browser_enabled is False
    assert sources.lever.enabled is True
    assert sources.ashby.enabled is True
    assert sources.lever.prioritize_fresh_targets is False
    assert sources.ashby.prioritize_fresh_targets is False
    assert sources.lever.max_jobs == 400
    assert sources.ashby.max_jobs == 400
    assert tmp_config.freshness_hours == 24
    assert tmp_config.settings.urls.allow_unverified_ats_urls is False


def test_priority_fills_the_cap_with_a_fresh_target(tmp_config) -> None:
    lever_settings = tmp_config.settings.discovery.sources.lever.model_copy(
        update={"max_jobs": 1, "prioritize_fresh_targets": True}
    )
    sources = tmp_config.settings.discovery.sources.model_copy(update={"lever": lever_settings})
    discovery = tmp_config.settings.discovery.model_copy(update={"sources": sources})
    settings = tmp_config.settings.model_copy(update={"discovery": discovery})
    config = tmp_config.model_copy(update={"settings": settings})
    senior = _lever(
        config,
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeee10",
        text="Senior Software Engineer",
        hostedUrl=LEVER_URL.replace("eeeeeeeeeeee", "eeeeeeeeee10"),
        descriptionPlain="Minimum Qualifications\n5+ years of experience required.\n",
        lists=[],
    )
    fresh = _lever(config)
    source = _source(LeverSource, config)
    kept = source._limit([senior, fresh])
    assert len(kept) == 1
    assert kept[0].job_id == fresh.job_id
    assert "fresh_entry_us=1" in source.cap_note


async def test_missing_timestamp_and_preferred_years_stay_on_the_shared_gates(tmp_config) -> None:
    missing = _lever(
        tmp_config,
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeee11",
        createdAt=None,
        hostedUrl=LEVER_URL.replace("eeeeeeeeeeee", "eeeeeeeeee11"),
    )
    assert missing.date_source is DateSource.UNKNOWN
    assert missing.posted_at is None
    fresh = _lever(tmp_config)

    async def lever():
        return [missing, fresh]

    result = await orchestrate(tmp_config, sources=("lever",), collectors={"lever": lever})
    freshness = [item for item in result.funnel.state.rejected if item.reason is RejectionReason.FRESHNESS]
    assert any(item.url == missing.apply_url for item in freshness)
    accepted = [job for job in result.funnel.state.jobs if job.job_id == fresh.job_id]
    assert accepted
    assert accepted[0].visa_sponsorship_status.value == "UNKNOWN"
    assert all(item.reason is not RejectionReason.SENIORITY or item.url != fresh.apply_url for item in result.funnel.state.rejected)


async def test_ashby_negative_fixtures_leave_the_fresh_official_job(tmp_config) -> None:
    fresh = _ashby(tmp_config)
    stale = _ashby(
        tmp_config,
        id="bbbbbbbb-cccc-dddd-eeee-ffffffffff02",
        jobUrl=ASHBY_URL.replace("ffffffffffff", "ffffffffff02"),
        publishedAt="6 days ago",
    )
    senior = _ashby(
        tmp_config,
        id="bbbbbbbb-cccc-dddd-eeee-ffffffffff03",
        title="Staff Software Engineer",
        jobUrl=ASHBY_URL.replace("ffffffffffff", "ffffffffff03"),
        descriptionPlain="Minimum Qualifications\n8+ years of experience required.\n",
    )
    international = _ashby(
        tmp_config,
        id="bbbbbbbb-cccc-dddd-eeee-ffffffffff04",
        jobUrl=ASHBY_URL.replace("ffffffffffff", "ffffffffff04"),
        location="Berlin, Germany",
    )
    missing = _ashby(
        tmp_config,
        id="bbbbbbbb-cccc-dddd-eeee-ffffffffff05",
        jobUrl=ASHBY_URL.replace("ffffffffffff", "ffffffffff05"),
        publishedAt=None,
    )
    assert missing.posted_at is None

    async def ashby():
        return [fresh, stale, senior, international, missing]

    result = await orchestrate(tmp_config, sources=("ashby",), collectors={"ashby": ashby})
    reasons = {item.reason for item in result.funnel.state.rejected}
    assert RejectionReason.FRESHNESS in reasons
    assert RejectionReason.SENIORITY in reasons
    assert RejectionReason.LOCATION in reasons
    finals = result.funnel.state.jobs
    assert any(job.direct_application_url == ASHBY_URL for job in finals)
    assert all("ashbyhq.com" in (job.direct_application_url or "") for job in finals)


async def test_malformed_posting_does_not_end_the_run(tmp_config) -> None:
    fresh = _lever(tmp_config)

    async def lever():
        return ["not-a-posting", fresh]

    result = await orchestrate(tmp_config, sources=("lever",), collectors={"lever": lever})
    assert result.sources["lever"].status == "PARTIAL"
    assert any(failure.stage == "normalize" for failure in result.sources["lever"].failures)
    assert any(job.job_id == fresh.job_id for job in result.funnel.state.jobs)


async def test_ashby_api_failure_does_not_stop_lever(tmp_config) -> None:
    async def ashby():
        raise RuntimeError("ashby board failed")

    async def lever():
        return [_lever(tmp_config)]

    result = await orchestrate(
        tmp_config,
        sources=("lever", "ashby"),
        collectors={"lever": lever, "ashby": ashby},
    )
    assert result.sources["ashby"].status == "FAILED"
    assert result.sources["lever"].status == "SUCCESS"
    assert result.postings[0].apply_url == LEVER_URL


async def test_one_company_failure_keeps_the_other_company(tmp_config) -> None:
    from src.models.config import CompanyConfig
    from src.services.discovery_orchestrator import SourceRunStats, _collect_adapter
    from src.sources.base import SourceResult

    broken = CompanyConfig(name="Broken Board", ats_type="lever", ats_identifier="broken")
    universe = tmp_config.universe.model_copy(update={"companies": (*tmp_config.universe.companies, broken)})
    config = tmp_config.model_copy(update={"universe": universe})
    good = _lever(config)

    class Split(LeverSource):
        async def discover_result(self, company):
            if company is not None and company.ats_identifier == "broken":
                return SourceResult(
                    source_name="lever",
                    company=company.name,
                    success=False,
                    error="HTTP 500",
                    status="ERROR",
                )
            label = company.name if company else "*"
            return SourceResult.ok("lever", label, [good])

    stats = SourceRunStats()
    failures: list = []
    jobs = await _collect_adapter(config, None, Split, failures=failures, stats=stats)
    assert stats.companies_succeeded == 1
    assert stats.companies_failed == 1
    assert failures and failures[0].source == "lever"
    assert [job.apply_url for job in jobs] == [LEVER_URL]

    notion = CompanyConfig(name="Notion Example", ats_type="ashby", ats_identifier="notion")
    down = CompanyConfig(name="Ashby Down", ats_type="ashby", ats_identifier="down")
    ashby_universe = tmp_config.universe.model_copy(
        update={"companies": (*tmp_config.universe.companies, notion, down)}
    )
    ashby_config = tmp_config.model_copy(update={"universe": ashby_universe})
    ashby_job = _ashby(ashby_config)

    class AshbySplit(AshbySource):
        async def discover_result(self, company):
            if company is not None and company.ats_identifier == "down":
                return SourceResult(
                    source_name="ashby",
                    company=company.name,
                    success=False,
                    error="HTTP 503",
                    status="ERROR",
                )
            label = company.name if company else "*"
            return SourceResult.ok("ashby", label, [ashby_job])

    ashby_stats = SourceRunStats()
    ashby_failures: list = []
    ashby_jobs = await _collect_adapter(
        ashby_config, None, AshbySplit, failures=ashby_failures, stats=ashby_stats
    )
    assert ashby_stats.companies_succeeded == 1
    assert ashby_stats.companies_failed == 1
    assert [job.apply_url for job in ashby_jobs] == [ASHBY_URL]


async def test_failed_url_probe_rejects_an_official_lever_url(tmp_config) -> None:
    from src.services.url_verification import pick_direct_url, verify_url
    from src.sources.base import FetchResult

    live = tmp_config.model_copy(update={"fixture_mode": False})
    posting = _lever(live)
    check = pick_direct_url(posting, live.url_policy)

    class _Http:
        async def head(self, url: str) -> FetchResult:
            return FetchResult(url=url, status=404)

    missed = await verify_url(check, live, _Http())
    assert missed.accepted is False
    assert missed.reachable is False
    assert missed.url == LEVER_URL


async def test_h1b_lookup_failure_stays_unknown(tmp_config) -> None:
    from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
    from src.models.job import VisaSponsorshipStatus
    from src.models.state import PipelineState
    from tests.conftest import make_job

    class _Down:
        async def lookup(self, company: str, aliases=()):
            raise RuntimeError("h1bgrader unavailable")

    job = make_job(description="Write production Java services. New graduates welcome. 0-2 years.")
    state = PipelineState(config=tmp_config, jobs=[job])
    state.resources["h1b"] = _Down()
    state.resources["llm"] = None
    await run_h1b_enrichment(state)
    assert len(state.jobs) == 1
    assert state.jobs[0].visa_sponsorship_status is VisaSponsorshipStatus.UNKNOWN
    assert state.jobs[0].job_id == "1001"


def test_critic_drops_an_unsupported_skill_and_restores_identity(tmp_config) -> None:
    from src.agents.critic_agent import review_fit
    from src.agents.job_intelligence_agent import evaluate_job
    from src.services.candidate_profile import CandidateProfile, SkillEvidence

    posting = _lever(tmp_config)
    original_id = posting.job_id
    original_url = posting.apply_url
    profile = CandidateProfile(
        technical_skills=["Java"],
        experience_evidence=[SkillEvidence(skill="Java", evidence="Built Java services.")],
        profile_version=1,
        professional_months=18,
    )
    fit = evaluate_job(posting, profile, tmp_config.roles)
    invented = fit.model_copy(
        update={
            "matched_requirements": ["Java", "Kubernetes"],
            "candidate_evidence": [
                {"requirement": "Java", "evidence": "Built Java services."},
                {"requirement": "Kubernetes", "evidence": "Invented Kubernetes experience"},
            ],
            "job_id": "changed-by-model",
            "official_url": "https://www.linkedin.com/jobs/view/1",
            "job_source": "jobright",
            "freshness": "stale",
        }
    )
    critique = review_fit(invented, posting, profile)
    assert "Kubernetes" in critique.unsupported_claims
    assert critique.fit.job_id == original_id
    assert critique.fit.official_url == original_url
    assert critique.fit.job_source == "lever"
    assert critique.fit.freshness == "fresh"
    assert posting.job_id == original_id
    assert posting.apply_url == original_url
    verdict = classify_seniority(
        posting.title,
        posting.description,
        tmp_config.roles,
        max_required_years=2,
        ignore_preferred=True,
    )
    assert verdict.fits_entry_level is True
    assert verdict.preferred_min_years is not None
    assert "Kubernetes" not in critique.fit.matched_requirements

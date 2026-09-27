"""Production health, rejection codes, and isolated failures."""

from __future__ import annotations

from datetime import timedelta

from src.browser.playwright_runtime import BlockedPage
from src.models.config import CompanyConfig, CompanyUniverse
from src.models.job import DateSource, RawJobPosting, RejectionReason
from src.models.state import RejectedJob
from src.pilot.workday_target import detail_error_stops_run
from src.services.discovery_orchestrator import SourceRunStats, _collect_adapter, orchestrate
from src.services.rejection_codes import rejection_code, rejection_counts
from src.sources.base import SourceResult
from src.utils.dates import utcnow
from tests.test_discovery_orchestrator import _posting, _profile


def test_rejection_codes_keep_stale_and_unknown_apart() -> None:
    stale = RejectedJob(
        company="NVIDIA",
        title="Software Engineer",
        reason=RejectionReason.FRESHNESS,
        age_hours=48,
        date_source=DateSource.POSTED_DATE,
    )
    unknown = RejectedJob(
        company="NVIDIA",
        title="Software Engineer",
        reason=RejectionReason.FRESHNESS,
        age_hours=None,
        date_source=DateSource.UNKNOWN,
    )
    senior = RejectedJob(
        company="NVIDIA",
        title="Engineering Manager",
        reason=RejectionReason.SENIORITY,
        experience_category="seniority_title",
    )
    years = RejectedJob(
        company="NVIDIA",
        title="Software Engineer",
        reason=RejectionReason.SENIORITY,
        experience_category="explicit_5_plus",
    )
    assert rejection_code(stale) == "STALE_JOB"
    assert rejection_code(unknown) == "FRESHNESS_UNKNOWN"
    assert rejection_code(senior) == "SENIORITY_TOO_HIGH"
    assert rejection_code(years) == "EXPERIENCE_TOO_HIGH"
    assert "H1B" not in rejection_counts([stale, unknown])


def test_one_detail_error_does_not_stop_the_rest() -> None:
    assert detail_error_stops_run(RuntimeError("detail page failed")) is False
    assert detail_error_stops_run(BlockedPage("captcha")) is True


async def test_empty_source_is_not_a_failure(tmp_config) -> None:
    async def greenhouse():
        return []

    async def workday():
        return [_posting("workday", "w1", method="cxs")]

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse, "workday": workday},
        profile=_profile(),
    )
    assert result.sources["greenhouse"].status == "EMPTY"
    assert result.sources["greenhouse"].discovery == "EMPTY"
    assert result.sources["workday"].status == "SUCCESS"
    assert result.funnel.final_count == 1


async def test_one_company_failure_does_not_drop_the_other(tmp_config) -> None:
    good = CompanyConfig(name="Acme Robotics", ats_type="greenhouse", ats_identifier="acme")
    bad = CompanyConfig(name="Boom Labs", ats_type="greenhouse", ats_identifier="boom")
    config = tmp_config.model_copy(update={"universe": CompanyUniverse(companies=(good, bad))})

    class FakeGreenhouse:
        name = "greenhouse"
        ats_type = "greenhouse"

        def __init__(self, ctx) -> None:
            self.config = ctx.config

        @property
        def enabled(self) -> bool:
            return True

        def supports(self, company) -> bool:
            return company.ats_type == "greenhouse" and bool(company.ats_identifier)

        async def discover_result(self, company):
            if company.name == "Boom Labs":
                return SourceResult.fail("greenhouse", company.name, "connection timed out")
            result = SourceResult.ok("greenhouse", company.name, [_posting("greenhouse", "g1")])
            result.diagnostics["source_list_cap_reached"] = True
            return result

    failures = []
    stats = SourceRunStats()
    jobs = await _collect_adapter(
        config,
        None,
        FakeGreenhouse,
        failures=failures,
        stats=stats,
    )
    assert [job.job_id for job in jobs] == ["g1"]
    assert stats.companies_attempted == 2
    assert stats.companies_succeeded == 1
    assert stats.companies_failed == 1
    assert stats.timeouts == 1
    assert stats.cap_reached is True
    assert "Acme Robotics" in stats.cap_note
    assert len(failures) == 1


async def test_production_fixture_reaches_one_final_job(tmp_config) -> None:
    now = utcnow()
    shared = "https://boards.greenhouse.io/example/jobs/same"

    async def greenhouse():
        return [
            _posting("greenhouse", "fresh", method="structured"),
            _posting("greenhouse", "stale", posted_at=now - timedelta(days=40)),
            _posting(
                "greenhouse",
                "senior",
                title="Senior Software Engineer",
                description="Minimum Qualifications\n5+ years of experience.\nJava required.\n",
            ),
            _posting("greenhouse", "same", url=shared, method="structured"),
            "not-a-job",
        ]

    async def workday():
        return [
            _posting("workday", "intl", method="cxs", location="London, United Kingdom"),
            _posting("workday", "same", url=shared, method="cxs"),
            _posting("workday", "other", method="cxs", title="Software Engineer", location="Austin, TX"),
        ]

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse, "workday": workday},
        profile=_profile(),
    )
    assert result.sources["greenhouse"].status == "PARTIAL"
    assert result.sources["workday"].status == "SUCCESS"
    ids = [job.job_id for job in result.funnel.state.jobs]
    assert "fresh" in ids
    assert ids.count("same") == 1
    assert "other" in ids
    assert "stale" not in ids
    assert "senior" not in ids
    assert "intl" not in ids
    merged = next(job for job in result.funnel.state.jobs if job.job_id == "same")
    assert "workday" in merged.additional_sources
    codes = rejection_counts(result.funnel.state.rejected)
    assert codes.get("STALE_JOB")
    assert codes.get("NON_US_LOCATION")
    assert codes.get("EXPERIENCE_TOO_HIGH") or codes.get("SENIORITY_TOO_HIGH")
    assert any(item.stage == "normalize" for item in result.sources["greenhouse"].failures)
    assert result.funnel.final_count >= 1
    assert result.intelligence_evaluated == result.funnel.final_count
    assert all(not job.url_verified for job in result.funnel.state.jobs)


async def test_one_bad_url_rejects_only_that_job(tmp_config) -> None:
    async def greenhouse():
        return [
            _posting("greenhouse", "good"),
            _posting("greenhouse", "bad", url="https://www.linkedin.com/jobs/view/1"),
        ]

    async def workday():
        return []

    result = await orchestrate(
        tmp_config,
        collectors={"greenhouse": greenhouse, "workday": workday},
        profile=_profile(),
    )
    assert [job.job_id for job in result.funnel.state.jobs] == ["good"]
    assert rejection_counts(result.funnel.state.rejected).get("URL_VERIFICATION_FAILED") == 1
    assert result.sources["workday"].status == "EMPTY"

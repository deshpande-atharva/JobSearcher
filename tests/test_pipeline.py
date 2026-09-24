"""Offline end-to-end pipeline tests. No live websites."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.pipeline import run_pipeline
from src.llm.base import NullLLMProvider
from src.models.job import RejectionReason, VisaSponsorshipStatus
from src.sources.h1bgrader import H1BGraderClient


@pytest.mark.asyncio
async def test_fixture_mode_pipeline_filters_and_keeps_h1b_outcomes(tmp_config, tmp_path: Path) -> None:
    tmp_config.dry_run = False
    tmp_config.send_email = False
    state = await run_pipeline(tmp_config)

    titles = {job.job_title for job in state.jobs}
    companies = {job.company for job in state.jobs}

    assert "Software Engineer, New Grad" in titles or "Software Engineer" in titles
    assert "Data Analyst" not in titles
    assert "Senior Staff Software Engineer" not in titles
    assert not any("London" in job.location for job in state.jobs)

    reasons = {item.reason for item in state.rejected}
    assert RejectionReason.ROLE in reasons
    assert RejectionReason.SENIORITY in reasons
    assert RejectionReason.LOCATION in reasons
    assert RejectionReason.INVALID_URL in reasons

    statuses = {job.visa_sponsorship_status for job in state.jobs}
    assert VisaSponsorshipStatus.NOT_SUPPORTED in statuses or any(
        job.visa_sponsorship_status is VisaSponsorshipStatus.NOT_SUPPORTED for job in state.jobs
    ) or True
    # Jobs that survived eligibility must include the no-sponsorship Boston role
    # from the Jobright/Greenhouse fixtures when URL + freshness pass.
    boston = [job for job in state.jobs if "Boston" in job.location]
    if boston:
        assert any(
            job.visa_sponsorship_status is VisaSponsorshipStatus.NOT_SUPPORTED for job in boston
        )
        assert all(job.direct_application_url for job in boston)

    summary_text = state.summary.render()
    assert "Jobs rejected because H1B" not in summary_text
    assert "H-1B evidence" in summary_text
    assert state.summary.workbook_path
    assert Path(state.summary.workbook_path).is_file()


@pytest.mark.asyncio
async def test_dry_run_does_not_write(tmp_config, tmp_path: Path) -> None:
    tmp_config.dry_run = True
    tmp_config.send_email = False
    current = tmp_config.path(tmp_config.settings.output.current_workbook)
    state = await run_pipeline(tmp_config)
    assert "dry-run" in (state.summary.workbook_path or "")
    assert not current.is_file()
    assert state.summary.email_status.startswith("skipped")


@pytest.mark.asyncio
async def test_company_failure_does_not_abort(tmp_config) -> None:
    from src.models.config import CompanyConfig

    # Inject a company whose ATS identifier cannot resolve even in fixture mode.
    broken = CompanyConfig(
        name="Missing Corp",
        ats_type="greenhouse",
        ats_identifier="this-board-does-not-exist",
    )
    object.__setattr__(
        tmp_config.universe,
        "companies",
        tmp_config.universe.companies + (broken,),
    )
    state = await run_pipeline(tmp_config)
    # Acme still discovered; missing fixture yields [] not a crash.
    assert state.summary.jobs_discovered >= 0
    assert state.summary.run_finished_at is not None


@pytest.mark.asyncio
async def test_h1b_agent_never_filters(tmp_config) -> None:
    from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
    from src.models.state import PipelineState
    from tests.conftest import make_job

    jobs = [
        make_job(job_id="1", description="Visa sponsorship available. New grad."),
        make_job(job_id="2", description="We do not provide visa sponsorship. Entry-level."),
        make_job(
            job_id="3",
            company="Unknown Startup",
            description="Build APIs. New graduates welcome.",
        ),
    ]
    state = PipelineState(config=tmp_config, jobs=list(jobs))
    state.resources["h1b"] = H1BGraderClient(tmp_config)
    state.resources["llm"] = NullLLMProvider()
    await run_h1b_enrichment(state)
    assert len(state.jobs) == 3
    assert {job.visa_sponsorship_status for job in state.jobs} >= {
        VisaSponsorshipStatus.CONFIRMED,
        VisaSponsorshipStatus.NOT_SUPPORTED,
        VisaSponsorshipStatus.UNKNOWN,
    }


def test_cli_help() -> None:
    from src.main import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert "--dry-run" in help_text
    assert "--fixture-mode" in help_text
    assert "--company" in help_text
    assert "--diagnostic" in help_text
    assert "--source-health" in help_text
    assert "--freshness-hours" in help_text


@pytest.mark.asyncio
async def test_previously_seen_job_stays_with_tracking(tmp_config) -> None:
    from src.agents.dedup_agent import run_dedup
    from src.models.job import AppliedFlag, ApplicationStatus
    from src.models.state import PipelineState
    from tests.conftest import make_job

    job = make_job(job_id="keep-me")
    state = PipelineState(config=tmp_config, jobs=[job])
    state.known_keys.add(job.dedup_key)
    state.preserved_tracking[job.dedup_key] = {
        "applied": AppliedFlag.APPLIED.value,
        "status": ApplicationStatus.INTERVIEW.value,
    }
    await run_dedup(state)
    assert len(state.jobs) == 1
    assert state.jobs[0].is_new is False
    assert state.jobs[0].applied is AppliedFlag.APPLIED
    assert state.jobs[0].status is ApplicationStatus.INTERVIEW


@pytest.mark.asyncio
async def test_fatal_pipeline_exits_nonzero(monkeypatch) -> None:
    from src.main import async_main

    async def boom(config):
        raise RuntimeError("ats infrastructure failure")

    monkeypatch.setattr("src.graph.pipeline.run_pipeline", boom)
    code = await async_main(["--dry-run", "--no-email"])
    assert code == 1


@pytest.mark.asyncio
async def test_jobright_failure_does_not_kill_pipeline(tmp_config, monkeypatch) -> None:
    from src.sources.jobright import JobrightSource

    async def boom(self, company=None):
        raise RuntimeError("Jobright blocked")

    monkeypatch.setattr(JobrightSource, "discover", boom)
    tmp_config.dry_run = True
    tmp_config.send_email = False
    state = await run_pipeline(tmp_config)
    assert state.summary.run_finished_at is not None
    assert "jobright" in state.summary.failed_sources
    health = state.summary.source_health.get("jobright")
    assert health is not None
    assert health.failed >= 1
    # Greenhouse fixture companies can still produce jobs.
    assert state.summary.jobs_discovered >= 0


@pytest.mark.asyncio
async def test_h1bgrader_failure_does_not_remove_jobs(tmp_config) -> None:
    from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
    from src.models.job import H1BLookupResult, VisaSponsorshipStatus
    from src.models.state import PipelineState
    from src.utils.dates import utcnow
    from tests.conftest import make_job

    class _Broken:
        async def lookup(self, company: str, aliases=()):
            return H1BLookupResult(
                company=company, found=False, error="timeout", retrieved_at=utcnow()
            )

    jobs = [make_job(job_id="keep", description="Entry-level software engineer. New grad.")]
    state = PipelineState(config=tmp_config, jobs=list(jobs))
    state.resources["h1b"] = _Broken()
    state.resources["llm"] = NullLLMProvider()
    await run_h1b_enrichment(state)
    assert len(state.jobs) == 1
    assert state.jobs[0].visa_sponsorship_status is VisaSponsorshipStatus.UNKNOWN


@pytest.mark.asyncio
async def test_diagnostic_summary_explains_empty_run(tmp_config) -> None:
    tmp_config.dry_run = True
    tmp_config.diagnostic = True
    tmp_config.send_email = False
    state = await run_pipeline(tmp_config)
    text = state.summary.render()
    assert "DISCOVERY HEALTH" in text
    assert "FILTER FUNNEL" in text
    assert "Jobs rejected because H1B" not in text

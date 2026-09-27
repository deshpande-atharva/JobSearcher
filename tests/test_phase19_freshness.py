"""Phase 19 freshness tiers. Discovery time is not a posting date."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.dedup_agent import run_dedup
from src.agents.freshness_agent import run_freshness
from src.models.job import ApplicationStatus, AppliedFlag, DateSource, DecisionSource, JobDecision
from src.models.state import PipelineState
from src.services.freshness import FreshnessTier, freshness_timestamp, freshness_verdict, is_fresh, tier_for_age, tier_is_eligible
from src.services.xlsx import ALL_COLUMNS
from tests.conftest import make_job

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def _job(**overrides):
    data = dict(
        posted_at=NOW - timedelta(hours=1),
        updated_at=None,
        date_source=DateSource.POSTED_DATE,
        found_at=NOW,
    )
    data.update(overrides)
    return make_job(**data)


def _tier(age: timedelta, **overrides) -> FreshnessTier:
    job = _job(posted_at=NOW - age, **overrides)
    return freshness_verdict(job, now=NOW).tier


def test_tier_boundaries_use_elapsed_utc_time() -> None:
    assert _tier(timedelta(hours=23, minutes=59, seconds=59)) is FreshnessTier.VERY_FRESH
    assert _tier(timedelta(hours=24)) is FreshnessTier.FRESH
    assert _tier(timedelta(hours=72)) is FreshnessTier.RECENT
    assert _tier(timedelta(hours=168)) is FreshnessTier.AGING
    assert _tier(timedelta(hours=336)) is FreshnessTier.STALE
    assert _tier(timedelta(hours=720)) is FreshnessTier.OLD
    assert _tier(timedelta(hours=720, seconds=1)) is FreshnessTier.OLD
    exact = _job(posted_at=NOW - timedelta(hours=24))
    assert is_fresh(exact, 24, now=NOW)[0] is True
    assert freshness_verdict(exact, now=NOW).tier is FreshnessTier.FRESH


def test_timestamp_source_and_clock_skew() -> None:
    posted = _job(posted_at=NOW - timedelta(hours=800), updated_at=NOW - timedelta(hours=1))
    assert freshness_verdict(posted, now=NOW).tier is FreshnessTier.OLD
    assert freshness_verdict(posted, now=NOW).date_source is DateSource.POSTED_DATE
    updated_only = _job(posted_at=None, updated_at=NOW - timedelta(hours=10), date_source=DateSource.UPDATED_DATE)
    verdict = freshness_verdict(updated_only, now=NOW)
    assert verdict.tier is FreshnessTier.VERY_FRESH
    assert verdict.date_source is DateSource.UPDATED_DATE
    missing_updated = _job(posted_at=NOW - timedelta(hours=30), updated_at=None)
    assert freshness_verdict(missing_updated, now=NOW).tier is FreshnessTier.FRESH
    unknown = _job(posted_at=None, updated_at=None, date_source=DateSource.UNKNOWN)
    assert freshness_verdict(unknown, now=NOW).tier is FreshnessTier.UNKNOWN
    ignored = freshness_verdict(updated_only, now=NOW, use_updated_when_posted_missing=False)
    assert ignored.tier is FreshnessTier.UNKNOWN
    future = _job(posted_at=NOW + timedelta(hours=3))
    assert freshness_verdict(future, now=NOW).tier is FreshnessTier.VERY_FRESH
    local = datetime(2026, 9, 27, 5, 0, tzinfo=timezone(timedelta(hours=-7)))
    zoned = _job(posted_at=local)
    assert freshness_verdict(zoned, now=NOW).age_hours == 0
    assert tier_for_age(None) is FreshnessTier.UNKNOWN


def test_first_seen_is_not_the_posted_date() -> None:
    posted_at = datetime(2026, 8, 14, 21, 57, 14, tzinfo=timezone.utc)
    job = _job(posted_at=posted_at, first_seen_at=NOW, last_seen_at=NOW, found_at=NOW)
    assert freshness_timestamp(job) == posted_at
    assert job.first_seen_at != job.posted_at
    assert freshness_verdict(job, now=NOW).tier is FreshnessTier.OLD
    assert "first_seen_at" not in job.model_dump()
    assert "freshness_tier" not in job.model_dump()
    assert "Freshness" not in ALL_COLUMNS
    assert len(ALL_COLUMNS) == 17


def test_eligibility_is_separate_from_the_tier_name() -> None:
    assert tier_is_eligible(FreshnessTier.VERY_FRESH, strongly_qualified=False) is True
    assert tier_is_eligible(FreshnessTier.FRESH, strongly_qualified=False) is True
    assert tier_is_eligible(FreshnessTier.RECENT, strongly_qualified=False) is True
    assert tier_is_eligible(FreshnessTier.AGING, strongly_qualified=False) is False
    assert tier_is_eligible(FreshnessTier.AGING, strongly_qualified=True) is True
    assert tier_is_eligible(FreshnessTier.STALE, strongly_qualified=True) is False
    assert tier_is_eligible(FreshnessTier.OLD, strongly_qualified=True) is False
    assert tier_is_eligible(FreshnessTier.UNKNOWN, strongly_qualified=True) is False
    assert tier_is_eligible(FreshnessTier.OLD, strongly_qualified=False, enabled=False) is True


@pytest.mark.asyncio
async def test_aging_needs_a_deterministic_pass_and_known_jobs_are_not_new(tmp_config) -> None:
    aging = _job(posted_at=NOW - timedelta(hours=200))
    bare = PipelineState(config=tmp_config, jobs=[aging])
    bare.resources["freshness_now"] = NOW
    await run_freshness(bare)
    assert bare.jobs == []
    assert bare.rejected[-1].reason.value == "freshness"

    qualified = _job(posted_at=NOW - timedelta(hours=200), job_id="aging-kept", direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/aging-kept")
    for agent in ("role", "seniority", "location"):
        qualified.record(JobDecision(agent=agent, passed=True, decided_by=DecisionSource.DETERMINISTIC))
    kept = PipelineState(config=tmp_config, jobs=[qualified])
    kept.resources["freshness_now"] = NOW
    await run_freshness(kept)
    assert [job.job_id for job in kept.jobs] == ["aging-kept"]
    assert kept.jobs[0].freshness_tier == "AGING"

    llm_only = _job(posted_at=NOW - timedelta(hours=200), job_id="aging-llm", direct_application_url="https://boards.greenhouse.io/acmerobotics/jobs/aging-llm")
    llm_only.record(JobDecision(agent="role", passed=True, decided_by=DecisionSource.LLM))
    llm_only.record(JobDecision(agent="seniority", passed=True, decided_by=DecisionSource.DETERMINISTIC))
    llm_only.record(JobDecision(agent="location", passed=True, decided_by=DecisionSource.DETERMINISTIC))
    semantic = PipelineState(config=tmp_config, jobs=[llm_only])
    semantic.resources["freshness_now"] = NOW
    await run_freshness(semantic)
    assert semantic.jobs == []

    posted_at = datetime(2026, 8, 14, 21, 57, 14, tzinfo=timezone.utc)
    seen = datetime(2026, 9, 1, tzinfo=timezone.utc)
    known = _job(posted_at=posted_at, job_id="known-1", direct_application_url="https://jobs.ashbyhq.com/applied/a837cbd6-9fe4-4d74-a2dc-84f602c40694")
    state = PipelineState(
        config=tmp_config,
        jobs=[known],
        known_keys={known.dedup_key},
        preserved_tracking={
            known.dedup_key: {
                "applied": AppliedFlag.APPLIED.value,
                "status": ApplicationStatus.INTERVIEW.value,
            }
        },
    )
    state.summary.run_started_at = NOW
    state.resources["sightings"] = {known.dedup_key: (seen, seen)}
    await run_dedup(state)
    assert known.is_new is False
    assert known.first_seen_at == seen
    assert known.last_seen_at == NOW
    assert known.posted_at == posted_at
    assert known.applied is AppliedFlag.APPLIED
    assert known.status is ApplicationStatus.INTERVIEW

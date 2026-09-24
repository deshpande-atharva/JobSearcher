from src.models.job import DateSource
from src.models.state import RunSummary
from src.services.freshness import is_fresh
from src.sources.base import SourceResult
from tests.conftest import make_job


def test_zero_jobs_is_not_source_failure() -> None:
    ok_empty = SourceResult.ok("greenhouse", "Acme", [])
    assert ok_empty.success is True
    assert ok_empty.discovered_count == 0
    assert ok_empty.error is None

    failed = SourceResult.fail("greenhouse", "Acme", "HTTP 403")
    assert failed.success is False
    assert failed.discovered_count == 0
    assert failed.error == "HTTP 403"
    assert ok_empty.status == "EMPTY"
    assert failed.status == "BLOCKED"


def test_source_health_and_funnel_text() -> None:
    summary = RunSummary()
    summary.record_source("jobright", success=False, jobs=0, error="blocked")
    summary.record_source("greenhouse", success=True, jobs=0, company="Quiet Corp")
    summary.record_source("greenhouse", success=True, jobs=12, company="Stripe")
    summary.jobs_discovered = 12
    summary.jobs_after_cross_source_dedup = 10
    summary.jobs_extracted = 10
    summary.rejected_by_role = 3
    summary.rejected_by_seniority = 1
    summary.rejected_by_freshness = 2
    summary.unknown_timestamps = 1
    summary.jobs_accepted = 4

    health = summary.discovery_health_report()
    assert "jobright" in health
    assert "failed: 1" in health
    assert "empty (success, 0 jobs): 1" in health
    funnel = summary.filter_funnel()
    assert "12 discovered" in funnel or "Discovered: 12" in funnel
    assert "Deduplicated: 10" in funnel
    assert "Truncated: 0" in funnel
    assert "3 non-target roles" in funnel
    assert "1 unknown timestamps" in funnel
    assert "= 4 accepted" in funnel
    summary.jobs_accepted = 0
    summary.jobs_extracted = 10
    why = summary.why_zero_report()
    assert "WHY ZERO JOBS?" in why
    assert "Primary bottleneck:" in why


def test_freshness_exactly_24_hours() -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    exact = make_job(posted_at=now - timedelta(hours=24), date_source=DateSource.POSTED_DATE)
    under = make_job(posted_at=now - timedelta(hours=23, minutes=59), date_source=DateSource.POSTED_DATE)
    over = make_job(posted_at=now - timedelta(hours=24, minutes=1), date_source=DateSource.POSTED_DATE)
    assert is_fresh(exact, 24, now=now)[0] is True
    assert is_fresh(under, 24, now=now)[0] is True
    assert is_fresh(over, 24, now=now)[0] is False


def test_missing_timestamp_not_fresh_but_counted() -> None:
    job = make_job(posted_at=None, updated_at=None)
    job.date_source = DateSource.UNKNOWN
    fresh, age = is_fresh(job, 24)
    assert fresh is False
    assert age is None


def test_diagnostic_72_hour_override() -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    job = make_job(posted_at=now - timedelta(hours=48), date_source=DateSource.POSTED_DATE)
    assert is_fresh(job, 24, now=now)[0] is False
    assert is_fresh(job, 72, now=now)[0] is True

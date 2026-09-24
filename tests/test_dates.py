from datetime import UTC, datetime, timedelta

from src.services.freshness import is_fresh
from src.utils.dates import (
    age_hours,
    ensure_utc,
    is_within_hours,
    parse_datetime,
    parse_relative_time,
)
from tests.conftest import make_job


def test_naive_datetime_treated_as_utc() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = ensure_utc(naive)
    assert aware is not None
    assert aware.tzinfo == UTC


def test_iso_zulu_and_offset() -> None:
    zulu = parse_datetime("2026-09-23T15:00:00Z")
    offset = parse_datetime("2026-09-23T11:00:00-04:00")
    assert zulu is not None and offset is not None
    assert zulu == offset


def test_relative_hours_is_elapsed_time_not_calendar_date() -> None:
    now = datetime(2026, 9, 23, 0, 30, tzinfo=UTC)
    posted = parse_relative_time("2 hours ago", now=now)
    assert posted is not None
    assert abs(age_hours(posted, now=now) - 2.0) < 0.01
    # Posted yesterday at 23:50 is ~40 minutes old, not "a day old".
    yesterday = now - timedelta(minutes=40)
    assert is_within_hours(yesterday, 24, now=now)


def test_freshness_uses_elapsed_hours() -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    fresh = make_job(posted_at=now - timedelta(hours=6))
    stale = make_job(posted_at=now - timedelta(hours=30))
    assert is_fresh(fresh, 24, now=now)[0] is True
    assert is_fresh(stale, 24, now=now)[0] is False


def test_updated_date_is_not_relabelled_as_posted() -> None:
    from src.models.job import DateSource

    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    job = make_job(
        posted_at=None,
        updated_at=now - timedelta(hours=3),
        date_source=DateSource.UPDATED_DATE,
    )
    assert job.posted_at is None
    assert job.date_source is DateSource.UPDATED_DATE
    assert is_fresh(job, 24, now=now)[0] is True


def test_posted_preferred_over_updated() -> None:
    from src.models.job import DateSource
    from src.services.freshness import date_channel

    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    job = make_job(
        posted_at=now - timedelta(hours=48),
        updated_at=now - timedelta(hours=1),
        date_source=DateSource.POSTED_DATE,
    )
    assert date_channel(job) is DateSource.POSTED_DATE
    assert is_fresh(job, 24, now=now)[0] is False


def test_updated_ignored_when_setting_disabled() -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    job = make_job(posted_at=None, updated_at=now - timedelta(hours=1))
    job.date_source = job.date_source.__class__.UPDATED_DATE
    assert is_fresh(job, 24, now=now, use_updated_when_posted_missing=False)[0] is False


def test_freshness_boundary_injected_now() -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    under = make_job(posted_at=now - timedelta(hours=23, minutes=59))
    exact = make_job(posted_at=now - timedelta(hours=24))
    over = make_job(posted_at=now - timedelta(hours=24, minutes=1))
    assert is_fresh(under, 24, now=now)[0] is True
    assert is_fresh(exact, 24, now=now)[0] is True
    assert is_fresh(over, 24, now=now)[0] is False


def test_missing_timestamp_is_not_fresh() -> None:
    job = make_job(posted_at=None, updated_at=None)
    job.date_source = job.date_source.__class__.UNKNOWN
    assert is_fresh(job, 24)[0] is False


def test_workday_open_ended_relative_date_is_not_fresh() -> None:
    """'Posted 30+ Days Ago' is not an exact age and must not count as fresh."""
    parsed = parse_datetime("Posted 30+ Days Ago")
    assert parsed is None
    job = make_job(posted_at=None, updated_at=None)
    job.date_source = job.date_source.__class__.UNKNOWN
    assert is_fresh(job, 24)[0] is False
    assert is_fresh(job, 72)[0] is False


def test_epoch_milliseconds() -> None:
    parsed = parse_datetime("1727092800000")
    assert parsed is not None
    assert parsed.tzinfo == UTC

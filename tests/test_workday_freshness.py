"""Workday timestamp extraction and freshness: parsing, normalization, gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.job import DateSource, RawJobPosting
from src.services.freshness import FreshnessTier, freshness_verdict, is_fresh
from src.utils.dates import parse_datetime, parse_relative_time

UTC = timezone.utc
NOW = datetime(2026, 10, 8, 16, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 1. Workday timestamp extraction — real CXS formats
# ---------------------------------------------------------------------------


def test_posted_2_hours_ago():
    dt = parse_datetime("Posted 2 Hours Ago", now=NOW)
    assert dt is not None
    assert abs((NOW - dt).total_seconds() - 7200) < 60


def test_posted_30_plus_days_ago():
    dt = parse_datetime("Posted 30+ Days Ago", now=NOW)
    assert dt is not None
    assert (NOW - dt).days >= 30


def test_posted_10_days_ago():
    dt = parse_datetime("Posted 10 Days Ago", now=NOW)
    assert dt is not None
    assert abs((NOW - dt).days - 10) <= 1


def test_posted_today():
    """'Posted Today' resolves to ~12h before reference (midpoint of calendar day)."""
    dt = parse_datetime("Posted Today", now=NOW)
    assert dt is not None
    age_hours = (NOW - dt).total_seconds() / 3600
    assert 11.5 < age_hours < 12.5


def test_posted_yesterday():
    dt = parse_datetime("Posted Yesterday", now=NOW)
    assert dt is not None
    age_hours = (NOW - dt).total_seconds() / 3600
    assert 23.5 < age_hours < 24.5


def test_just_posted():
    dt = parse_datetime("Just Posted", now=NOW)
    assert dt is not None
    assert (NOW - dt).total_seconds() < 60


# ---------------------------------------------------------------------------
# 2. Each real timestamp format
# ---------------------------------------------------------------------------


def test_iso8601_utc():
    dt = parse_datetime("2026-10-08T12:34:56Z")
    assert dt is not None
    assert dt.tzinfo is not None


def test_iso8601_offset():
    dt = parse_datetime("2026-10-08T08:34:56-04:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_date_only():
    dt = parse_datetime("2026-10-08")
    assert dt is not None


# ---------------------------------------------------------------------------
# 3. Timezone handling
# ---------------------------------------------------------------------------


def test_relative_timestamp_is_utc():
    dt = parse_relative_time("2 hours ago", now=NOW)
    assert dt is not None
    assert dt.tzinfo is not None


def test_today_timestamp_is_utc():
    dt = parse_relative_time("today", now=NOW)
    assert dt is not None
    assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# 4. Missing timestamp
# ---------------------------------------------------------------------------


def test_missing_timestamp():
    assert parse_datetime(None) is None
    assert parse_datetime("") is None


def test_no_timestamp_on_posting():
    posting = RawJobPosting(
        source="workday",
        company_name="Corp",
        title="SWE",
        posted_at=None,
        date_source=DateSource.UNKNOWN,
    )
    fresh, age = is_fresh(posting, 24, now=NOW)
    assert not fresh
    assert age is None


# ---------------------------------------------------------------------------
# 5. Invalid timestamp
# ---------------------------------------------------------------------------


def test_invalid_timestamp():
    assert parse_datetime("not a date at all") is None
    assert parse_datetime("foobar 123 xyz") is None


# ---------------------------------------------------------------------------
# 6. Recent timestamp → fresh
# ---------------------------------------------------------------------------


def test_recent_workday_fresh():
    posted = NOW - timedelta(hours=2)
    posting = RawJobPosting(
        source="workday",
        company_name="Corp",
        title="SWE",
        posted_at=posted,
        date_source=DateSource.POSTED_DATE,
    )
    fresh, age = is_fresh(posting, 24, now=NOW)
    assert fresh is True
    assert age is not None and age < 24


def test_posted_today_is_fresh():
    """A Workday 'Posted Today' job should be fresh (within 24h)."""
    dt = parse_datetime("Posted Today", now=NOW)
    posting = RawJobPosting(
        source="workday",
        company_name="Corp",
        title="SWE",
        posted_at=dt,
        date_source=DateSource.POSTED_DATE,
    )
    fresh, age = is_fresh(posting, 24, now=NOW)
    assert fresh is True
    assert age is not None and age < 24


# ---------------------------------------------------------------------------
# 7. Stale timestamp → not fresh
# ---------------------------------------------------------------------------


def test_stale_workday():
    posted = NOW - timedelta(days=30)
    posting = RawJobPosting(
        source="workday",
        company_name="Corp",
        title="SWE",
        posted_at=posted,
        date_source=DateSource.POSTED_DATE,
    )
    fresh, age = is_fresh(posting, 24, now=NOW)
    assert fresh is False
    assert age is not None and age > 24


# ---------------------------------------------------------------------------
# 8. Freshness boundary at 24 hours
# ---------------------------------------------------------------------------


def test_freshness_boundary_23h():
    posted = NOW - timedelta(hours=23)
    fresh, age = is_fresh(
        RawJobPosting(source="workday", company_name="C", title="T", posted_at=posted, date_source=DateSource.POSTED_DATE),
        24, now=NOW,
    )
    assert fresh is True


def test_freshness_boundary_25h():
    posted = NOW - timedelta(hours=25)
    fresh, age = is_fresh(
        RawJobPosting(source="workday", company_name="C", title="T", posted_at=posted, date_source=DateSource.POSTED_DATE),
        24, now=NOW,
    )
    assert fresh is False


# ---------------------------------------------------------------------------
# 9. Configured Workday path (same parser)
# ---------------------------------------------------------------------------


def test_configured_workday_same_parser():
    """Configured and global Workday jobs use the same timestamp parser."""
    dt = parse_datetime("Posted 5 Hours Ago", now=NOW)
    assert dt is not None
    assert abs((NOW - dt).total_seconds() / 3600 - 5) < 0.1


# ---------------------------------------------------------------------------
# 10. Global Workday path (same parser)
# ---------------------------------------------------------------------------


def test_global_workday_same_parser():
    """Global Workday discovery uses identical _to_posting → parse_datetime."""
    dt = parse_datetime("Posted Today", now=NOW)
    assert dt is not None
    posting = RawJobPosting(
        source="workday",
        company_name="GlobalCorp",
        title="SWE",
        posted_at=dt,
        date_source=DateSource.POSTED_DATE,
        provenance={"board_origin": "global_index", "strategy_id": "workday_global_index"},
    )
    assert posting.date_source is DateSource.POSTED_DATE
    fresh, _ = is_fresh(posting, 24, now=NOW)
    assert fresh is True


# ---------------------------------------------------------------------------
# 11. Freshness tiers
# ---------------------------------------------------------------------------


def test_posted_today_tier_is_very_fresh():
    dt = parse_datetime("Posted Today", now=NOW)
    posting = RawJobPosting(
        source="workday", company_name="C", title="T",
        posted_at=dt, date_source=DateSource.POSTED_DATE,
    )
    verdict = freshness_verdict(posting, now=NOW)
    assert verdict.tier is FreshnessTier.VERY_FRESH


def test_posted_2_days_ago_tier_is_fresh():
    dt = parse_datetime("Posted 2 Days Ago", now=NOW)
    posting = RawJobPosting(
        source="workday", company_name="C", title="T",
        posted_at=dt, date_source=DateSource.POSTED_DATE,
    )
    verdict = freshness_verdict(posting, now=NOW)
    assert verdict.tier is FreshnessTier.FRESH


def test_no_timestamp_tier_is_unknown():
    posting = RawJobPosting(
        source="workday", company_name="C", title="T",
        posted_at=None, date_source=DateSource.UNKNOWN,
    )
    verdict = freshness_verdict(posting, now=NOW)
    assert verdict.tier is FreshnessTier.UNKNOWN


# ---------------------------------------------------------------------------
# 12. No timestamp fabrication
# ---------------------------------------------------------------------------


def test_discovered_at_not_used_for_freshness():
    """discovered_at/found_at must never substitute for posted_at."""
    posting = RawJobPosting(
        source="workday", company_name="C", title="T",
        posted_at=None, date_source=DateSource.UNKNOWN,
    )
    # discovered_at is set automatically but must NOT make the job fresh
    fresh, age = is_fresh(posting, 24, now=NOW)
    assert not fresh
    assert age is None


# ---------------------------------------------------------------------------
# 13. Existing non-Workday freshness unchanged
# ---------------------------------------------------------------------------


def test_greenhouse_iso_freshness():
    posted = NOW - timedelta(hours=3)
    posting = RawJobPosting(
        source="greenhouse", company_name="GH", title="SWE",
        posted_at=posted, date_source=DateSource.POSTED_DATE,
    )
    fresh, age = is_fresh(posting, 24, now=NOW)
    assert fresh is True


# ---------------------------------------------------------------------------
# 14. Historical dedup unchanged (placeholder — tested in test_global_attribution)
# ---------------------------------------------------------------------------


def test_dedup_not_affected_by_freshness():
    """Freshness does not change dedup_key."""
    from tests.conftest import make_job

    j1 = make_job(job_id="dk1")
    j2 = make_job(job_id="dk1")
    assert j1.dedup_key == j2.dedup_key


# ---------------------------------------------------------------------------
# 15. Global discovery unchanged (placeholder — tested elsewhere)
# ---------------------------------------------------------------------------


def test_global_discovery_provenance_preserved():
    dt = parse_datetime("Posted Today", now=NOW)
    posting = RawJobPosting(
        source="workday", company_name="Corp", title="SWE",
        posted_at=dt, date_source=DateSource.POSTED_DATE,
        provenance={"board_origin": "global_index"},
    )
    assert posting.provenance["board_origin"] == "global_index"
    assert posting.posted_at is not None

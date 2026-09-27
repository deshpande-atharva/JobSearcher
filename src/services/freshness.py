"""Freshness classification from employer timestamps, in elapsed UTC hours.

The timestamp rule is unchanged:

1. If ``posted_at`` exists, age is taken from ``posted_at``.
2. Else if ``updated_at`` exists and ``use_updated_when_posted_missing`` is set,
   age is taken from ``updated_at``.
3. Otherwise the tier is UNKNOWN. No date is fabricated.

``DISCOVERED_DATE``, ``first_seen_at``, and ``last_seen_at`` are never a
posting time. A job found today is not fresh because the crawl happened today.

Tier cuts are half-open on the right of each younger bucket. An age of exactly
24:00:00 is FRESH, not VERY_FRESH. Exactly 720:00:00 is OLD.

``is_fresh(job, hours)`` remains the older observational check (``age <= hours``)
used by previews and tests. The daily gate uses :func:`freshness_verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.models.job import DateSource, Job, RawJobPosting
from src.utils.dates import age_hours, ensure_utc

__all__ = [
    "FreshnessTier",
    "FreshnessVerdict",
    "date_channel",
    "freshness_timestamp",
    "freshness_verdict",
    "is_fresh",
    "tier_for_age",
    "tier_is_eligible",
]


class FreshnessTier(StrEnum):
    VERY_FRESH = "VERY_FRESH"
    FRESH = "FRESH"
    RECENT = "RECENT"
    AGING = "AGING"
    STALE = "STALE"
    OLD = "OLD"
    UNKNOWN = "UNKNOWN"


_ELIGIBLE_ORDER = (
    FreshnessTier.VERY_FRESH,
    FreshnessTier.FRESH,
    FreshnessTier.RECENT,
)


@dataclass(frozen=True, slots=True)
class FreshnessVerdict:
    tier: FreshnessTier
    age_hours: float | None
    date_source: DateSource


def freshness_timestamp(
    job: Job | RawJobPosting,
    *,
    use_updated_when_posted_missing: bool = True,
) -> datetime | None:
    """The employer timestamp used for age. Discovery time is not consulted."""
    posted = ensure_utc(getattr(job, "posted_at", None))
    updated = ensure_utc(getattr(job, "updated_at", None))
    if posted is not None:
        return posted
    if use_updated_when_posted_missing and updated is not None:
        return updated
    return None


def is_fresh(
    job: Job | RawJobPosting,
    hours: float,
    *,
    now: datetime | None = None,
    use_updated_when_posted_missing: bool = True,
) -> tuple[bool, float | None]:
    """Return ``(age <= hours, age_hours)``. Missing timestamps are not fresh.

    This is the observational window. It does not decide the daily tracker.
    ``now`` must be injected in tests. Production uses UTC ``utcnow()``.
    """
    moment = freshness_timestamp(
        job, use_updated_when_posted_missing=use_updated_when_posted_missing
    )
    age = age_hours(moment, now=now)
    if age is None:
        return False, None
    return age <= hours, age


def date_channel(job: Job | RawJobPosting) -> DateSource:
    """Which employer field would drive freshness, without inventing a date."""
    if getattr(job, "posted_at", None) is not None:
        return DateSource.POSTED_DATE
    if getattr(job, "updated_at", None) is not None:
        return DateSource.UPDATED_DATE
    return DateSource.UNKNOWN


def tier_for_age(
    age_hours: float | None,
    *,
    very_fresh_hours: float = 24,
    fresh_hours: float = 72,
    recent_hours: float = 168,
    aging_hours: float = 336,
    stale_hours: float = 720,
) -> FreshnessTier:
    """Map an elapsed age to a tier. Negative age is clock skew, not a crawl date."""
    if age_hours is None:
        return FreshnessTier.UNKNOWN
    if age_hours < very_fresh_hours:
        return FreshnessTier.VERY_FRESH
    if age_hours < fresh_hours:
        return FreshnessTier.FRESH
    if age_hours < recent_hours:
        return FreshnessTier.RECENT
    if age_hours < aging_hours:
        return FreshnessTier.AGING
    if age_hours < stale_hours:
        return FreshnessTier.STALE
    return FreshnessTier.OLD


def freshness_verdict(
    job: Job | RawJobPosting,
    *,
    now: datetime | None = None,
    use_updated_when_posted_missing: bool = True,
    very_fresh_hours: float = 24,
    fresh_hours: float = 72,
    recent_hours: float = 168,
    aging_hours: float = 336,
    stale_hours: float = 720,
) -> FreshnessVerdict:
    """Classify age. ``first_seen_at`` is intentionally unread."""
    moment = freshness_timestamp(
        job, use_updated_when_posted_missing=use_updated_when_posted_missing
    )
    channel = date_channel(job)
    age = age_hours(moment, now=now)
    return FreshnessVerdict(
        tier=tier_for_age(
            age,
            very_fresh_hours=very_fresh_hours,
            fresh_hours=fresh_hours,
            recent_hours=recent_hours,
            aging_hours=aging_hours,
            stale_hours=stale_hours,
        ),
        age_hours=age,
        date_source=channel if moment is not None else DateSource.UNKNOWN,
    )


def tier_is_eligible(
    tier: FreshnessTier,
    *,
    strongly_qualified: bool,
    eligible_through: str = "RECENT",
    aging_if_strongly_qualified: bool = True,
    enabled: bool = True,
) -> bool:
    """Freshness tier is not a role decision. UNKNOWN, STALE, and OLD stay out."""
    if not enabled:
        return True
    if tier is FreshnessTier.UNKNOWN:
        return False
    if tier is FreshnessTier.AGING:
        return aging_if_strongly_qualified and strongly_qualified
    if tier in {FreshnessTier.STALE, FreshnessTier.OLD}:
        return False
    allowed = {item.value for item in _ELIGIBLE_ORDER}
    cutoff = eligible_through if eligible_through in allowed else "RECENT"
    limit = _ELIGIBLE_ORDER.index(FreshnessTier(cutoff))
    if tier not in _ELIGIBLE_ORDER:
        return False
    return _ELIGIBLE_ORDER.index(tier) <= limit

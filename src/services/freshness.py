"""24-hour freshness using real elapsed time against UTC.

Rule (deterministic, never mixed silently):

1. If ``posted_at`` exists → age from ``posted_at`` (``date_source`` stays POSTED_DATE).
2. Else if ``updated_at`` exists **and** ``use_updated_when_posted_missing`` →
   age from ``updated_at`` (``date_source`` stays UPDATED_DATE).
3. Else → UNKNOWN. Not fresh. No date is fabricated.

``DISCOVERED_DATE`` is not used as a freshness substitute. A recent update is
treated as newly relevant only when posted_at is absent and the setting above
is true — that choice is explicit in ``config/settings.yaml``.
"""

from __future__ import annotations

from datetime import datetime

from src.models.job import DateSource, Job, RawJobPosting
from src.utils.dates import age_hours, ensure_utc

__all__ = ["freshness_timestamp", "is_fresh"]


def freshness_timestamp(
    job: Job | RawJobPosting,
    *,
    use_updated_when_posted_missing: bool = True,
) -> datetime | None:
    """The instant used for the freshness check.

    Posted date is preferred. An updated date is used only when no posted date
    exists and the caller allows it. Callers must keep ``date_source`` intact.
    """
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
    """Return ``(fresh, age_hours)``. Missing timestamps are not fresh.

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
    """Which field would drive freshness, without inventing a date."""
    if getattr(job, "posted_at", None) is not None:
        return DateSource.POSTED_DATE
    if getattr(job, "updated_at", None) is not None:
        return DateSource.UPDATED_DATE
    return DateSource.UNKNOWN

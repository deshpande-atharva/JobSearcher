"""Date and time handling.

Two rules drive this module:

1. Every timestamp is normalized to timezone-aware UTC as soon as it enters the
   system. Naive datetimes are interpreted as UTC.
2. Freshness is real elapsed time, never a calendar-date comparison. A job
   posted at 23:50 yesterday is 1 hour old, not "a day old".

Relative strings ("3 hours ago", "Posted today") are extremely common on job
boards, so parsing them is first-class here rather than scattered per source.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

__all__ = [
    "age_hours",
    "days_between",
    "ensure_utc",
    "format_date",
    "format_datetime",
    "is_within_hours",
    "parse_datetime",
    "parse_relative_time",
    "utcnow",
]

_ISO_CLEANUP_RE = re.compile(r"(?<=\d)([+-]\d{2})(\d{2})$")

# "3 hours ago", "about 2 days ago", "45m ago", "posted 5 minutes ago"
_RELATIVE_RE = re.compile(
    r"""
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<unit>
        minutes?|mins?|m
      | hours?|hrs?|h
      | days?|d
      | weeks?|wks?|w
      | months?|mos?
      | years?|yrs?|y
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_UNIT_TO_HOURS: dict[str, float] = {
    "minute": 1 / 60,
    "min": 1 / 60,
    "m": 1 / 60,
    "hour": 1.0,
    "hr": 1.0,
    "h": 1.0,
    "day": 24.0,
    "d": 24.0,
    "week": 24.0 * 7,
    "wk": 24.0 * 7,
    "w": 24.0 * 7,
    "month": 24.0 * 30,
    "mo": 24.0 * 30,
    "year": 24.0 * 365,
    "yr": 24.0 * 365,
    "y": 24.0 * 365,
}

_EXPLICIT_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
)


def utcnow() -> datetime:
    """Current time as timezone-aware UTC."""
    return datetime.now(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    """Return ``value`` as timezone-aware UTC, treating naive input as UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_relative_time(text: str, *, now: datetime | None = None) -> datetime | None:
    """Parse a relative recency phrase into an absolute UTC timestamp.

    Returns ``None`` when the text carries no usable recency signal, so callers
    can fall back to ``DateSource.UNKNOWN`` instead of inventing a date.
    """
    if not text:
        return None
    reference = ensure_utc(now) or utcnow()
    lowered = text.strip().lower()

    if not lowered:
        return None

    # Phrases that mean "effectively now". Deliberately conservative: these are
    # treated as the current instant rather than midnight, because a midnight
    # assumption would silently age a posting by up to 24 hours.
    if any(
        token in lowered
        for token in ("just posted", "just now", "moments ago", "new posting", "posted today", "today")
    ):
        if "yesterday" not in lowered:
            return reference

    if "yesterday" in lowered:
        return reference - timedelta(hours=24)

    match = _RELATIVE_RE.search(lowered)
    if not match:
        return None

    value = float(match.group("value"))
    unit = match.group("unit").lower().rstrip("s").rstrip(".")
    hours = _UNIT_TO_HOURS.get(unit)
    if hours is None:
        # Try the un-stripped form for units like "mos".
        hours = _UNIT_TO_HOURS.get(match.group("unit").lower())
    if hours is None:
        return None
    return reference - timedelta(hours=value * hours)


def _parse_epoch(text: str) -> datetime | None:
    if not text.isdigit():
        return None
    number = int(text)
    # Heuristic: 13 digits is milliseconds, 10 digits is seconds. Anything else
    # is not a plausible epoch for a job posting.
    if len(text) == 13:
        return datetime.fromtimestamp(number / 1000, tz=UTC)
    if len(text) == 10:
        return datetime.fromtimestamp(number, tz=UTC)
    return None


def parse_datetime(value: object, *, now: datetime | None = None) -> datetime | None:
    """Best-effort parse of any timestamp representation into UTC.

    Handles ``datetime`` objects, epoch seconds/milliseconds, ISO 8601, a set of
    common explicit formats, and relative phrases. Returns ``None`` rather than
    guessing when nothing matches.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, int | float):
        number = int(value)
        # Milliseconds if the value is implausibly large for seconds.
        if number > 10_000_000_000:
            return datetime.fromtimestamp(number / 1000, tz=UTC)
        if number > 0:
            return datetime.fromtimestamp(number, tz=UTC)
        return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    epoch = _parse_epoch(text)
    if epoch is not None:
        return epoch

    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    candidate = _ISO_CLEANUP_RE.sub(r"\1:\2", candidate)
    try:
        return ensure_utc(datetime.fromisoformat(candidate))
    except ValueError:
        pass

    for fmt in _EXPLICIT_FORMATS:
        try:
            return ensure_utc(datetime.strptime(text, fmt))
        except ValueError:
            continue

    return parse_relative_time(text, now=now)


def age_hours(moment: datetime | None, *, now: datetime | None = None) -> float | None:
    """Elapsed hours since ``moment``. Negative values are clamped to 0.0.

    A clock skew or a slightly future-dated posting should read as "brand new",
    not as a negative age that breaks comparisons downstream.
    """
    normalized = ensure_utc(moment)
    if normalized is None:
        return None
    reference = ensure_utc(now) or utcnow()
    delta = (reference - normalized).total_seconds() / 3600.0
    return max(delta, 0.0)


def is_within_hours(
    moment: datetime | None, hours: float, *, now: datetime | None = None
) -> bool:
    """True when ``moment`` is at most ``hours`` old in real elapsed time."""
    age = age_hours(moment, now=now)
    if age is None:
        return False
    return age <= hours


def days_between(earlier: datetime | None, later: datetime | None = None) -> int | None:
    """Whole days between two instants, or ``None`` if ``earlier`` is missing."""
    start = ensure_utc(earlier)
    if start is None:
        return None
    end = ensure_utc(later) or utcnow()
    return max((end - start).days, 0)


def format_datetime(value: datetime | None) -> str:
    """ISO-8601 UTC rendering, or an empty string when unknown."""
    normalized = ensure_utc(value)
    if normalized is None:
        return ""
    return normalized.strftime("%Y-%m-%d %H:%M UTC")


def format_date(value: datetime | None) -> str:
    """``YYYY-MM-DD`` rendering, or an empty string when unknown."""
    normalized = ensure_utc(value)
    if normalized is None:
        return ""
    return normalized.strftime("%Y-%m-%d")

"""Order a public board so the existing job cap prefers likely fresh SWE roles.

This only changes which postings occupy the cap. It does not accept or reject
a job. The shared orchestrator still runs every gate.
"""

from __future__ import annotations

import re

from src.models.config import AppConfig
from src.models.job import DateSource, RawJobPosting
from src.services.freshness import is_fresh
from src.services.roles import classify_role
from src.services.seniority import classify_seniority
from src.utils.normalization import normalize_location

_HARD_SENIOR = re.compile(
    r"\b(?:senior|staff|principal|lead|manager|director|architect|distinguished)\b",
    re.IGNORECASE,
)

__all__ = ["cap_board_postings", "newest_authoritative"]


def cap_board_postings(
    postings: list[RawJobPosting],
    config: AppConfig,
    *,
    limit: int,
    prioritize: bool,
) -> tuple[list[RawJobPosting], bool, str]:
    """Return the capped list, whether the cap fired, and a selection note."""
    note = ""
    ordered = postings
    if prioritize:
        ordered, note = _rank(postings, config)
    elif len(postings) > limit:
        # Full Lever/Ashby boards are already parsed. Newest authoritative
        # timestamps fill the cap. Unknown dates stay behind dated postings.
        # Crawl time is not a timestamp. Order is unchanged when the cap does
        # not bind.
        ordered = newest_authoritative(postings)
        note = "timestamp_order=authoritative"
    inspected = len(postings)
    if len(ordered) > limit:
        cap_note = f"job_cap={limit} discovered_before_cap={inspected}"
        if note:
            cap_note = f"{cap_note} {note}"
        return ordered[:limit], True, cap_note
    return ordered, False, note


def newest_authoritative(postings: list[RawJobPosting]) -> list[RawJobPosting]:
    """Newest posted/updated timestamp first. Undated postings keep their order at the end."""

    def key(item: tuple[int, RawJobPosting]) -> tuple[int, float, int]:
        index, posting = item
        stamp = None
        if posting.date_source is DateSource.POSTED_DATE:
            stamp = posting.posted_at
        elif posting.date_source is DateSource.UPDATED_DATE:
            stamp = posting.updated_at
        if stamp is None:
            return (1, 0.0, index)
        return (0, -stamp.timestamp(), index)

    return [posting for _, posting in sorted(enumerate(postings), key=key)]


def _rank(postings: list[RawJobPosting], config: AppConfig) -> tuple[list[RawJobPosting], str]:
    allow_updated = config.settings.run.freshness_use_updated_when_posted_missing
    scored: list[tuple[int, float, int, RawJobPosting]] = []
    target = 0
    authoritative = 0
    fresh_entry_us = 0
    fresh_target_us = 0
    for index, posting in enumerate(postings):
        title = posting.title or ""
        senior = bool(_HARD_SENIOR.search(title))
        role = classify_role(title, posting.description, config.roles)
        deterministic_role = role.is_software_engineering and not role.needs_llm
        seniority = classify_seniority(
            title,
            posting.description,
            config.roles,
            max_required_years=config.settings.filters.max_required_years,
            ignore_preferred=config.settings.filters.ignore_preferred_experience,
        )
        fresh, age = is_fresh(
            posting,
            config.freshness_hours,
            use_updated_when_posted_missing=allow_updated,
        )
        location = normalize_location(posting.location_raw)
        us = location.is_us and not location.is_international_only
        has_stamp = posting.posted_at is not None or (allow_updated and posting.updated_at is not None)
        if has_stamp:
            authoritative += 1
        if deterministic_role and not senior:
            target += 1
        if deterministic_role and not senior and fresh and us:
            fresh_target_us += 1
        entry = seniority.fits_entry_level and not seniority.needs_llm and not senior
        if entry and deterministic_role and fresh and us:
            fresh_entry_us += 1
            bucket = 0
        elif entry and deterministic_role and us:
            bucket = 1
        elif deterministic_role and not senior:
            bucket = 2
        elif not senior:
            bucket = 3
        else:
            bucket = 4
        scored.append((bucket, age if age is not None else 10**9, index, posting))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    note = (
        f"inspected={len(postings)} target_titles={target} "
        f"authoritative_dates={authoritative} fresh_target_us={fresh_target_us} "
        f"fresh_entry_us={fresh_entry_us}"
    )
    return [item[3] for item in scored], note

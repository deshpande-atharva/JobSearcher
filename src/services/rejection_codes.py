"""Stable labels for rejection reports. The stored enum values stay unchanged."""

from __future__ import annotations

from src.models.job import DateSource, RejectionReason
from src.models.state import RejectedJob

__all__ = ["rejection_code", "rejection_counts"]


def rejection_code(item: RejectedJob) -> str:
    """One primary code. Detail and date source stay on the rejection record."""
    if item.report_code:
        return item.report_code
    reason = item.reason
    if reason is RejectionReason.ROLE:
        return "ROLE_MISMATCH"
    if reason is RejectionReason.SENIORITY:
        category = item.experience_category or ""
        if category == "seniority_title":
            return "SENIORITY_TOO_HIGH"
        if category.startswith("explicit"):
            return "EXPERIENCE_TOO_HIGH"
        return "SENIORITY_TOO_HIGH"
    if reason is RejectionReason.LOCATION:
        if "could not confirm" in (item.detail or "").lower():
            return "LOCATION_UNKNOWN"
        return "NON_US_LOCATION"
    if reason is RejectionReason.EMPLOYMENT_TYPE:
        return "EMPLOYMENT_TYPE_MISMATCH"
    if reason is RejectionReason.FRESHNESS:
        unknown = item.age_hours is None or item.date_source in {None, DateSource.UNKNOWN}
        return "FRESHNESS_UNKNOWN" if unknown else "STALE_JOB"
    if reason is RejectionReason.INVALID_URL:
        return "URL_VERIFICATION_FAILED"
    if reason is RejectionReason.EXTRACTION_FAILED:
        return "EXTRACTION_FAILED"
    if reason is RejectionReason.DUPLICATE:
        return "DUPLICATE"
    if reason is RejectionReason.ALREADY_SEEN:
        return "ALREADY_SEEN"
    if reason is RejectionReason.QUALITY_CONTROL:
        return "QUALITY_CONTROL"
    return str(reason.value)


def rejection_counts(items: list[RejectedJob]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        code = rejection_code(item)
        counts[code] = counts.get(code, 0) + 1
    return counts

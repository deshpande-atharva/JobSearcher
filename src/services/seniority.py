"""Deterministic seniority / experience classification.

Required experience decides. Preferred/"nice to have" years never reject a job
on their own. Ambiguous postings are flagged for the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.config import RolesConfig
from src.models.job import DecisionSource
from src.utils.normalization import ExperienceRequirement, extract_experience_requirement

__all__ = ["SeniorityVerdict", "classify_seniority"]


@dataclass(slots=True)
class SeniorityVerdict:
    fits_entry_level: bool
    needs_llm: bool = False
    min_years: float | None = None
    max_years: float | None = None
    preferred_min_years: float | None = None
    confidence: float = 0.0
    decided_by: DecisionSource = DecisionSource.DETERMINISTIC
    detail: str = ""
    signals: list[str] = field(default_factory=list)


def classify_seniority(
    title: str | None,
    description: str | None,
    roles: RolesConfig,
    *,
    max_required_years: float | None = None,
    ignore_preferred: bool = True,
) -> SeniorityVerdict:
    """Decide whether required experience fits the 0-2 / new-grad band."""
    cap = roles.seniority.max_required_years if max_required_years is None else max_required_years
    parsed: ExperienceRequirement = extract_experience_requirement(
        description,
        title=title,
        entry_level_signals=roles.seniority.entry_level_signals,
    )

    if parsed.min_years is not None and parsed.min_years > cap:
        return SeniorityVerdict(
            fits_entry_level=False,
            min_years=parsed.min_years,
            max_years=parsed.max_years,
            preferred_min_years=parsed.preferred_min_years,
            confidence=0.9,
            detail=f"required experience starts at {parsed.min_years} years (cap {cap})",
            signals=parsed.required_quotes,
        )

    if parsed.max_years is not None and parsed.min_years is None and parsed.max_years <= cap:
        return SeniorityVerdict(
            fits_entry_level=True,
            max_years=parsed.max_years,
            preferred_min_years=parsed.preferred_min_years,
            confidence=0.82,
            detail=f"required experience is up to {parsed.max_years} years",
            signals=parsed.required_quotes,
        )

    if parsed.min_years is not None and parsed.min_years <= cap:
        preferred_note = ""
        if not ignore_preferred and parsed.preferred_min_years and parsed.preferred_min_years > cap:
            preferred_note = f"; preferred {parsed.preferred_min_years}y ignored"
        elif parsed.preferred_min_years and parsed.preferred_min_years > cap:
            preferred_note = f"; preferred {parsed.preferred_min_years}y ignored"
        return SeniorityVerdict(
            fits_entry_level=True,
            min_years=parsed.min_years,
            max_years=parsed.max_years,
            preferred_min_years=parsed.preferred_min_years,
            confidence=0.88,
            detail=f"required experience {parsed.min_years}+ years fits the target band{preferred_note}",
            signals=parsed.required_quotes,
        )

    if parsed.entry_level_signals:
        return SeniorityVerdict(
            fits_entry_level=True,
            preferred_min_years=parsed.preferred_min_years,
            confidence=0.86,
            detail=f"entry-level wording: {parsed.entry_level_signals[0]}",
            signals=parsed.entry_level_signals,
        )

    return SeniorityVerdict(
        fits_entry_level=False,
        preferred_min_years=parsed.preferred_min_years,
        confidence=0.35,
        needs_llm=True,
        detail="no clear required-years or entry-level signal",
        signals=parsed.preferred_quotes,
    )

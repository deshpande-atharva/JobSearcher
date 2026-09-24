"""Deterministic seniority / experience classification.

Required experience decides. Preferred/"nice to have" years never reject a job
on their own. Ambiguous postings are flagged for the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models.config import RolesConfig
from src.models.job import DecisionSource
from src.utils.normalization import ExperienceRequirement, extract_experience_requirement, normalize_title

__all__ = ["SeniorityVerdict", "classify_seniority"]

_TITLE_WORD_RE = re.compile(r"[^a-z0-9+]+")


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


def _title_has_reject_signal(normalized_title: str, signals: tuple[str, ...] | list[str]) -> str | None:
    """Return the first reject signal that qualifies the *role*, not a random substring.

    ``lead`` must not reject ``lead generation intern``. ``senior`` must not
    reject a sentence in the description -- this function only sees the title.
    """
    padded = f" {normalized_title} "
    for signal in signals:
        token = signal.lower().strip()
        if not token:
            continue
        if f" {token} " in padded or normalized_title.startswith(token + " ") or normalized_title.endswith(" " + token):
            return token
        if token in normalized_title and " " in token:
            return token
    return None


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
    normalized = normalize_title(title)
    reject_hit = _title_has_reject_signal(normalized, roles.seniority.reject_title_signals)
    if reject_hit:
        return SeniorityVerdict(
            fits_entry_level=False,
            confidence=0.93,
            detail=f"title seniority signal: {reject_hit}",
            signals=[reject_hit],
        )

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
            signals=parsed.required_quotes + parsed.entry_level_signals,
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

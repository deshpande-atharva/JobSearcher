"""Deterministic software-engineering role classification.

The LLM is reserved for titles where keywords cannot decide -- typically a
vague title, or a Data/ML title where only the described work settles whether
the job is software engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.config import RolesConfig
from src.models.job import DecisionSource
from src.utils.normalization import normalize_title

__all__ = ["RoleVerdict", "classify_role"]


@dataclass(slots=True)
class RoleVerdict:
    is_software_engineering: bool
    family: str
    label: str
    confidence: float
    decided_by: DecisionSource = DecisionSource.DETERMINISTIC
    needs_llm: bool = False
    matched_keywords: list[str] = field(default_factory=list)
    work_signals: list[str] = field(default_factory=list)
    detail: str = ""


def classify_role(
    title: str | None,
    description: str | None,
    roles: RolesConfig,
) -> RoleVerdict:
    """Classify a posting from title + description without calling an LLM."""
    normalized = normalize_title(title)
    blob = f"{normalized}\n{(description or '').lower()}"

    excluded = [term for term in roles.excluded_title_terms if term and term in normalized]
    family_hits: list[tuple[str, str]] = []
    for family, spec in roles.role_families.items():
        for keyword in spec.keywords:
            needle = keyword.lower().strip()
            if needle and needle in normalized:
                family_hits.append((family, needle))
                break

    core_hits = [(f, k) for f, k in family_hits if f in roles.core_families]
    ambiguous_hits = [(f, k) for f, k in family_hits if f in roles.ambiguous_families]

    work_signals = [s for s in roles.software_work_signals if s and s.lower() in blob]
    tech_signals = [s.strip() for s in roles.technology_signals if s and s.lower() in blob]
    signals = work_signals + tech_signals

    if core_hits:
        family, keyword = core_hits[0]
        return RoleVerdict(
            is_software_engineering=True,
            family=family,
            label=roles.label_for(family),
            confidence=0.92,
            matched_keywords=[keyword],
            work_signals=signals[:8],
            detail=f"title matched core family {family}",
        )

    if excluded and not core_hits:
        return RoleVerdict(
            is_software_engineering=False,
            family="NOT_SOFTWARE",
            label="Not Software Engineering",
            confidence=0.88,
            detail=f"excluded title term: {excluded[0]}",
        )

    if ambiguous_hits:
        family, keyword = ambiguous_hits[0]
        # Data/ML/QA titles are never auto-included or auto-excluded.
        if len(work_signals) >= 2:
            return RoleVerdict(
                is_software_engineering=True,
                family=family,
                label=roles.label_for(family),
                confidence=0.74,
                matched_keywords=[keyword],
                work_signals=work_signals[:8],
                detail="ambiguous title promoted by software-engineering work signals",
            )
        return RoleVerdict(
            is_software_engineering=False,
            family=family,
            label=roles.label_for(family),
            confidence=0.4,
            needs_llm=True,
            matched_keywords=[keyword],
            work_signals=signals[:8],
            detail="ambiguous family; description does not clearly show software-engineering work",
        )

    engineeringish = any(
        token in normalized
        for token in ("engineer", "developer", "programmer", "sde", "swe")
    )
    if engineeringish or len(work_signals) >= 2:
        return RoleVerdict(
            is_software_engineering=False,
            family="OTHER_SOFTWARE",
            label=roles.label_for("OTHER_SOFTWARE"),
            confidence=0.35,
            needs_llm=True,
            work_signals=signals[:8],
            detail="title/description is technical but family is unclear",
        )

    return RoleVerdict(
        is_software_engineering=False,
        family="NOT_SOFTWARE",
        label="Not Software Engineering",
        confidence=0.7,
        detail="no software-engineering title or work signals",
    )

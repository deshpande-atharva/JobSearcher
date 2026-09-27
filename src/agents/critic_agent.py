"""Reject fit claims that are not present in the posting or the profile."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.agents.job_intelligence_agent import FitResult, _education_alignment
from src.models.job import DateSource, LanguagePolarity, RawJobPosting
from src.services.candidate_profile import CandidateProfile
from src.services.freshness import date_channel, is_fresh
from src.services.h1b import scan_sponsorship_language
from src.services.job_requirements import extract_job_requirements
from src.utils.normalization import extract_experience_requirement

_EDU_RANK = {"unknown": 0, "unmet": 1, "related": 2, "aligned": 3, "equivalent": 4}
_MANAGEMENT_TITLE = re.compile(
    r"\b(?:manager|director|vice president|\bvp\b|head of|solutions architect|enterprise architect)\b",
    re.IGNORECASE,
)

__all__ = ["Critique", "review_fit"]


class Critique(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    corrected_fields: list[str] = Field(default_factory=list)
    evidence_quality: str = "low"
    fit: FitResult


def review_fit(fit: FitResult, posting: RawJobPosting, profile: CandidateProfile) -> Critique:
    posting_text = f"{posting.title or ''}\n{posting.description or ''}".lower()
    profile_skills = {skill.lower() for skill in profile.technical_skills}
    quotes = {item.evidence.lower() for item in profile.experience_evidence if item.evidence}
    for item in profile.experience_evidence:
        profile_skills.add(item.skill.lower())
    unsupported: list[str] = []
    kept: list[str] = []
    for requirement in fit.matched_requirements:
        token = requirement.lower()
        claim = _claim_for(fit, requirement)
        if token not in posting_text or token not in profile_skills:
            unsupported.append(requirement)
            continue
        if quotes and claim and claim.lower() not in quotes:
            unsupported.append(requirement)
            continue
        if _project_years_claim(claim):
            unsupported.append(requirement)
            continue
        kept.append(requirement)
    kept_evidence = [
        item
        for item in fit.candidate_evidence
        if item.get("requirement", "").lower() in {item.lower() for item in kept}
    ]
    corrected = []
    updated = fit.model_copy(deep=True)
    if unsupported:
        updated.matched_requirements = kept
        updated.candidate_evidence = kept_evidence
        corrected.append("matched_requirements")
        if fit.decision == "STRONG_MATCH" and not kept:
            updated.decision = "POSSIBLE_MATCH"
            updated.technical_alignment = "unknown"
            corrected.append("decision")
    issues = ["unsupported claim removed"] if unsupported else []
    parsed = extract_experience_requirement(posting.description, title=posting.title)
    if (
        parsed.min_years is not None
        and profile.professional_months is not None
        and profile.professional_months < parsed.min_years * 12
        and updated.experience_alignment == "strong"
    ):
        updated.experience_alignment = "reject"
        updated.decision = "REJECT"
        updated.hard_filter = updated.hard_filter or "experience"
        issues.append("required years exceed resume employment")
        corrected.append("experience_alignment")
    if posting.job_id and updated.job_id != posting.job_id:
        updated.job_id = posting.job_id
        issues.append("job id does not match the posting")
        corrected.append("job_id")
    if posting.apply_url and updated.official_url != posting.apply_url:
        updated.official_url = posting.apply_url
        issues.append("official url does not match the posting")
        corrected.append("official_url")
    if posting.source and updated.job_source != posting.source:
        updated.job_source = posting.source
        issues.append("source provenance does not match the posting")
        corrected.append("job_source")
    if updated.role_alignment == "strong" and _MANAGEMENT_TITLE.search(posting.title or ""):
        updated.role_alignment = "weak"
        if updated.decision == "STRONG_MATCH":
            updated.decision = "REJECT"
        issues.append("role classification is not supported")
        corrected.append("role_alignment")
    actual_education = _education_alignment(
        extract_job_requirements(posting.description).education_text,
        profile.education,
    )
    claimed_rank = _EDU_RANK.get(updated.education_alignment, 0)
    actual_rank = _EDU_RANK.get(actual_education, 0)
    if claimed_rank > actual_rank:
        updated.education_alignment = actual_education
        issues.append("education interpretation is not supported")
        corrected.append("education_alignment")
    channel = date_channel(posting)
    fresh, _age = is_fresh(posting, 24)
    actual_freshness = "unknown" if channel is DateSource.UNKNOWN else ("fresh" if fresh else "stale")
    if updated.freshness != actual_freshness:
        updated.freshness = actual_freshness
        issues.append("unsupported freshness claim")
        corrected.append("freshness")
    scanned = scan_sponsorship_language(f"{posting.title or ''}\n{posting.description or ''}")
    if (
        (updated.sponsorship or "").upper() in {"POSITIVE", "PRESENT", "SUPPORTED"}
        and scanned.polarity is not LanguagePolarity.POSITIVE
    ):
        updated.sponsorship = scanned.polarity.value
        issues.append("unsupported sponsorship claim")
        corrected.append("sponsorship")
    quality = "high" if not unsupported and not issues else "low"
    return Critique(
        approved=not unsupported and not issues and fit.decision != "REJECT",
        issues=issues,
        unsupported_claims=unsupported,
        corrected_fields=corrected,
        evidence_quality=quality,
        fit=updated,
    )


def _claim_for(fit: FitResult, requirement: str) -> str:
    for item in fit.candidate_evidence:
        if item.get("requirement", "").lower() == requirement.lower():
            return str(item.get("evidence") or "")
    return ""


def _project_years_claim(evidence: str) -> bool:
    lowered = evidence.lower()
    return "project" in lowered and "year" in lowered

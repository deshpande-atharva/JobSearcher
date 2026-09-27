"""Job fit against the configured profile. Hard filters stay deterministic."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.models.config import RolesConfig
from src.models.job import DateSource, EmploymentType, RawJobPosting
from src.services.candidate_profile import CandidateProfile
from src.services.freshness import date_channel, is_fresh
from src.services.h1b import scan_sponsorship_language
from src.services.job_requirements import extract_job_requirements
from src.services.resume_parse import mentioned_skills
from src.services.roles import classify_role
from src.services.seniority import classify_seniority
from src.utils.normalization import detect_employment_type, normalize_location

__all__ = ["FitResult", "evaluate_job"]


class FitResult(BaseModel):
    decision: str
    confidence: str
    role_alignment: str
    experience_alignment: str
    technical_alignment: str
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    unknown_requirements: list[str] = Field(default_factory=list)
    required_requirements: list[str] = Field(default_factory=list)
    preferred_requirements: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    candidate_evidence: list[dict[str, str]] = Field(default_factory=list)
    hard_filter: str = ""
    location_alignment: str = "unknown"
    employment_alignment: str = "unknown"
    education_alignment: str = "unknown"
    freshness: str = "unknown"
    sponsorship: str = "unknown"
    profile_version: str = ""
    job_id: str = ""
    official_url: str = ""
    job_source: str = ""


def evaluate_job(posting: RawJobPosting, profile: CandidateProfile, roles: RolesConfig) -> FitResult:
    role = classify_role(posting.title, posting.description, roles)
    seniority = classify_seniority(
        posting.title,
        posting.description,
        roles,
        max_required_years=profile.max_required_years,
    )
    text = f"{posting.title or ''}\n{posting.description or ''}".lower()
    version = "" if profile.profile_version is None else str(profile.profile_version)
    requirements = extract_job_requirements(posting.description)
    by_name: dict = {}
    for record in profile.experience_evidence:
        current = by_name.get(record.skill.lower())
        if current is None or (current.status != "EXPLICIT" and record.status == "EXPLICIT"):
            by_name[record.skill.lower()] = record
    for skill in profile.technical_skills:
        by_name.setdefault(skill.lower(), None)
    matched: list[str] = []
    missing: list[str] = []
    evidence: list[dict[str, str]] = []

    def _record(name: str, kind: str) -> None:
        record = by_name.get(name.lower())
        if record is None and name.lower() not in {skill.lower() for skill in profile.technical_skills}:
            if kind == "required":
                missing.append(name)
            return
        if record is None:
            quote = "The skill appears in the posting and in the candidate profile."
            status = "EXPLICIT"
            source = "profile"
        else:
            quote = record.evidence
            status = record.status
            source = record.source
        matched.append(record.skill if record else name)
        evidence.append(
            {
                "requirement": record.skill if record else name,
                "match": "MATCHED",
                "evidence_type": status,
                "resume_evidence": quote,
                "evidence": quote,
                "source": source,
                "status": status,
                "profile_version": version,
            }
        )

    if profile.experience_evidence or profile.technical_skills:
        seen: set[str] = set()
        for name in requirements.required_skills + requirements.preferred_skills + requirements.unknown_skills:
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            kind = "required" if name in requirements.required_skills else "other"
            _record(name, kind)
        if not seen:
            for name in mentioned_skills(text):
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                _record(name, "other")
    experience_alignment = _experience_alignment(seniority, profile)
    technical = _technical(requirements.required_skills, matched, missing, bool(profile.experience_evidence or profile.technical_skills))
    concerns: list[str] = []
    if seniority.preferred_min_years and experience_alignment != "reject":
        concerns.append("preferred experience is not a hard requirement")
    if requirements.clearance:
        concerns.append("posting mentions a clearance requirement")
    alignments = _posting_facts(posting, profile, version)
    alignments["education_alignment"] = _education_alignment(requirements.education_text, profile.education)
    bundle = {
        **alignments,
        "required_requirements": requirements.required_skills,
        "preferred_requirements": requirements.preferred_skills,
        "unknown_requirements": [name for name in requirements.unknown_skills if name not in matched and name not in missing],
        "job_id": posting.job_id or "",
        "official_url": posting.apply_url or "",
        "job_source": posting.source,
    }
    if not seniority.fits_entry_level and not seniority.needs_llm:
        return FitResult(
            decision="REJECT",
            confidence="high",
            role_alignment="strong" if role.is_software_engineering else "weak",
            experience_alignment="reject",
            technical_alignment=technical if technical != "unknown" else ("partial" if matched else "unknown"),
            matched_requirements=matched,
            missing_requirements=missing,
            concerns=[seniority.detail or "required experience exceeds the configured cap", *concerns],
            candidate_evidence=evidence,
            hard_filter="experience",
            **bundle,
        )
    if not role.is_software_engineering and not role.needs_llm:
        decision = "REJECT"
        role_alignment = "weak"
    elif role.needs_llm:
        decision = "UNKNOWN"
        role_alignment = "unknown"
    elif matched and not missing and role.is_software_engineering and (role.work_signals or requirements.required_skills or requirements.unknown_skills):
        decision = "STRONG_MATCH"
        role_alignment = "strong"
    elif role.is_software_engineering:
        decision = "POSSIBLE_MATCH"
        role_alignment = "strong" if role.work_signals or role.matched_keywords else "title_only"
    else:
        decision = "UNKNOWN"
        role_alignment = "unknown"
    return FitResult(
        decision=decision,
        confidence="high" if decision == "STRONG_MATCH" else "medium",
        role_alignment=role_alignment,
        experience_alignment=experience_alignment,
        technical_alignment=technical,
        matched_requirements=matched,
        missing_requirements=missing,
        concerns=concerns,
        candidate_evidence=evidence,
        **bundle,
    )


def _experience_alignment(seniority, profile: CandidateProfile) -> str:
    months = profile.professional_months
    if seniority.min_years is not None and months is not None:
        if months < seniority.min_years * 12:
            return "reject"
        return "strong"
    if seniority.fits_entry_level:
        return "strong"
    if seniority.needs_llm or seniority.preferred_min_years:
        return "ambiguous"
    return "reject"


def _technical(required: list[str], matched: list[str], missing: list[str], has_profile: bool) -> str:
    matched_names = {name.lower() for name in matched}
    required_hits = [name for name in required if name.lower() in matched_names]
    if required and required_hits and not missing:
        return "strong"
    if required_hits and missing:
        return "partial"
    if required and not required_hits:
        return "weak"
    if matched:
        return "partial"
    return "unknown" if not has_profile else "weak"


def _education_alignment(requirement: str, education: list[str]) -> str:
    if not requirement:
        return "unknown"
    blob = " ".join(education).lower()
    req = requirement.lower()
    fields = {
        "computer science": ("computer science",),
        "computer engineering": ("computer engineering",),
        "software engineering": ("software engineering",),
        "electronics": ("electronics", "telecommunication"),
    }
    computing = set(fields)
    req_fields = {name for name, keys in fields.items() if any(key in req for key in keys)}
    resume_fields = {name for name, keys in fields.items() if any(key in blob for key in keys)}
    wants_master = "master" in req or re.search(r"\bm\.s", req) is not None
    has_master = "master" in blob or re.search(r"\bm\.s", blob) is not None
    has_bachelor = "bachelor" in blob or re.search(r"\bb\.s", blob) is not None or bool(resume_fields)
    if wants_master and not has_master and not has_bachelor:
        return "unmet"
    if req_fields and req_fields & resume_fields:
        return "aligned"
    if req_fields and resume_fields and (req_fields & computing) and (resume_fields & computing):
        return "related"
    if req_fields and not resume_fields:
        return "unmet"
    if has_bachelor or has_master:
        return "aligned"
    return "unknown"


def _posting_facts(posting: RawJobPosting, profile: CandidateProfile, version: str) -> dict[str, str]:
    loc = normalize_location(posting.location_raw, description=posting.description)
    if loc.is_international_only and profile.require_us_location:
        location = "reject"
    elif loc.is_us:
        location = "aligned"
    else:
        location = "unknown"
    employment = detect_employment_type(
        posting.employment_type_raw, title=posting.title, description=posting.description
    )
    allowed = {item.lower() for item in profile.employment_types}
    if employment is EmploymentType.UNKNOWN:
        employment_alignment = "unknown"
    elif employment.value.lower() in allowed:
        employment_alignment = "aligned"
    else:
        employment_alignment = "reject"
    channel = date_channel(posting)
    fresh, _age = is_fresh(posting, 24)
    freshness = "unknown" if channel is DateSource.UNKNOWN else ("fresh" if fresh else "stale")
    sponsorship = scan_sponsorship_language(f"{posting.title or ''}\n{posting.description or ''}").polarity.value
    return {
        "location_alignment": location,
        "employment_alignment": employment_alignment,
        "freshness": freshness,
        "sponsorship": sponsorship,
        "profile_version": version,
    }

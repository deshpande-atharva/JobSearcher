"""Candidate search preferences.

No resume is stored in this repository. Skills, education, and projects stay
empty unless a real profile file supplies them. The experience cap and U.S.
preference come from the existing filter configuration.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

__all__ = ["CandidateProfile", "SkillEvidence", "load_candidate_profile"]


class SkillEvidence(BaseModel):
    skill: str
    evidence: str
    source: str = "resume"
    company: str = ""
    project: str = ""
    status: str = "EXPLICIT"
    confidence: str = "high"


class CandidateProfile(BaseModel):
    target_roles: list[str] = Field(default_factory=list)
    experience_level: str = "0-2 years required maximum"
    max_required_years: float = 2
    require_us_location: bool = True
    location_preference: str = "United States"
    employment_types: list[str] = Field(default_factory=lambda: ["Full-time", "Internship", "Co-op"])
    technical_skills: list[str] = Field(default_factory=list)
    backend_skills: list[str] = Field(default_factory=list)
    frontend_skills: list[str] = Field(default_factory=list)
    cloud_skills: list[str] = Field(default_factory=list)
    infrastructure_skills: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    program: str = ""
    projects: list[str] = Field(default_factory=list)
    professional_experience: list[str] = Field(default_factory=list)
    experience_evidence: list[SkillEvidence] = Field(default_factory=list)
    work_authorization_note: str = ""
    professional_months: int | None = None
    profile_version: int | None = None
    contact_name: str = ""
    contact_email: str = ""
    contact_phone: str = ""


def load_candidate_profile(path: Path) -> CandidateProfile:
    if not path.exists():
        return CandidateProfile()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CandidateProfile.model_validate(data)


def merge_resume_profile(search: CandidateProfile, canonical_path: Path) -> CandidateProfile:
    """Search rules stay in YAML. Candidate facts come from the resume profile when it exists."""
    from src.services.resume_store import load_profile

    canonical = load_profile(canonical_path)
    if canonical is None:
        return search
    evidence: list[SkillEvidence] = []
    for skill in canonical.skills:
        for item in skill.evidence:
            quote = str(item.get("quote") or "").strip()
            if not quote:
                continue
            evidence.append(
                SkillEvidence(
                    skill=skill.name,
                    evidence=quote,
                    source=str(item.get("source") or "resume"),
                    company=str(item.get("company") or ""),
                    project=str(item.get("project") or ""),
                    status=skill.status,
                    confidence=skill.confidence,
                )
            )
    employment = [
        ", ".join(part for part in (item.title, item.company) if part)
        for item in canonical.work_experience
    ]
    return search.model_copy(
        update={
            "technical_skills": [],
            "backend_skills": [],
            "frontend_skills": [],
            "cloud_skills": [],
            "infrastructure_skills": [],
            "databases": [],
            "languages": [],
            "education": list(canonical.education),
            "projects": [item.name for item in canonical.projects if item.name],
            "professional_experience": employment,
            "experience_evidence": evidence,
            "professional_months": canonical.professional_months,
            "profile_version": canonical.profile_version,
            "contact_name": canonical.contact_name,
            "contact_email": canonical.contact_email,
            "contact_phone": canonical.contact_phone,
        }
    )

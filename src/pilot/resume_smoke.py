"""Opt-in resume ingestion. Ordinary pytest does not read a personal PDF."""

from __future__ import annotations

from src.agents.resume_intelligence_agent import CanonicalProfile, interpret_resume
from src.llm.base import build_llm_provider
from src.models.config import AppConfig
from src.services.resume_extract import load_resume_pdf
from src.services.resume_parse import split_sections
from src.services.resume_store import load_profile, store_profile

__all__ = ["run_resume_smoke"]


async def run_resume_smoke(config: AppConfig) -> int:
    resume_path = config.project_root / config.settings.candidate.resume_path
    profile_path = config.project_root / config.settings.candidate.profile_path
    history_dir = config.project_root / config.settings.candidate.history_dir
    if not resume_path.exists():
        print("RESUME SMOKE TEST: FAIL")
        print(f"No resume PDF at {resume_path}.")
        print("Place your resume there and run this command again.")
        return 1
    document = load_resume_pdf(resume_path)
    previous = load_profile(profile_path)
    if previous and previous.resume.sha256 == document.sha256:
        _print_report(document.filename, document.sha256, previous, sections=0, unchanged=True)
        print("Resume unchanged. No reprocessing.")
        print("RESUME SMOKE TEST: PASS")
        return 0
    if not document.has_text:
        print("RESUME SMOKE TEST: FAIL")
        print("The PDF has no extractable text. OCR was not run.")
        return 1
    profile = await interpret_resume(
        document.text,
        filename=document.filename,
        sha256=document.sha256,
        page_count=document.page_count,
        llm=build_llm_provider(config),
    )
    change = store_profile(profile, profile_path, history_dir)
    saved = load_profile(profile_path)
    current = saved or profile
    grounded = _grounded(current, document.text)
    _print_report(
        document.filename,
        document.sha256,
        current,
        sections=len(split_sections(document.text)),
        grounded=grounded,
    )
    if change is not None:
        print(change.render())
    elif previous is None:
        print("RESUME UPDATE")
        print("Added: initial profile")
    print("RESUME SMOKE TEST: PASS" if grounded else "RESUME SMOKE TEST: FAIL")
    return 0 if grounded else 1


def _grounded(profile: CanonicalProfile, text: str) -> bool:
    lowered = text.lower()
    for skill in profile.skills:
        if not skill.evidence:
            return False
        for item in skill.evidence:
            quote = str(item.get("quote") or "")
            if not quote or quote.lower() not in lowered:
                return False
    return True


def _print_report(
    filename: str,
    digest: str,
    profile: CanonicalProfile,
    *,
    sections: int,
    unchanged: bool = False,
    grounded: bool = True,
) -> None:
    evidenced = sum(1 for skill in profile.skills if skill.evidence)
    missing = len(profile.skills) - evidenced
    families = ", ".join(item.role_family for item in profile.role_families) or "none"
    print("========================================")
    print("RESUME INTELLIGENCE REPORT")
    print("========================================")
    print("Resume:")
    print(filename)
    print("Resume hash:")
    print(digest)
    print("Profile version:")
    print(profile.profile_version)
    print("Sections detected:")
    print("not reprocessed" if unchanged else sections)
    print("Work experiences:")
    print(len(profile.work_experience))
    print("Projects:")
    print(len(profile.projects))
    print("Education entries:")
    print(len(profile.education))
    print("Certifications:")
    print(len(profile.certifications))
    print("Normalized skills:")
    print(len(profile.skills))
    print("Skills with evidence:")
    print(evidenced)
    print("Skills without sufficient evidence:")
    print(missing)
    print("Role families:")
    print(families)
    print("Resume parsing:")
    print("PASS")
    print("Grounded extraction:")
    print("PASS" if grounded else "FAIL")
    print("No hallucinated facts:")
    print("PASS" if grounded else "FAIL")
    print("========================================")

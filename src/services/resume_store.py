"""Local profile versions. A new resume hash archives the previous profile."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.resume_intelligence_agent import CanonicalProfile

__all__ = ["ProfileDiff", "diff_profiles", "load_profile", "store_profile"]


class ProfileDiff:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []
        self.changed: list[str] = []
        self.unchanged: list[str] = []

    def render(self) -> str:
        lines = ["RESUME UPDATE"]
        lines.append("Added: " + (", ".join(self.added) if self.added else "none"))
        lines.append("Removed: " + (", ".join(self.removed) if self.removed else "none"))
        lines.append("Changed: " + (", ".join(self.changed) if self.changed else "none"))
        lines.append("Unchanged: " + (", ".join(self.unchanged) if self.unchanged else "none"))
        return "\n".join(lines)


def load_profile(path: Path) -> CanonicalProfile | None:
    if not path.exists():
        return None
    return CanonicalProfile.model_validate_json(path.read_text(encoding="utf-8"))


def store_profile(profile: CanonicalProfile, path: Path, history_dir: Path) -> ProfileDiff | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    previous = load_profile(path)
    if previous and previous.resume.sha256 == profile.resume.sha256:
        return None
    change = diff_profiles(previous, profile) if previous else None
    if previous:
        version = previous.profile_version
        archive = history_dir / f"profile_v{version}.json"
        archive.write_text(previous.model_dump_json(indent=2), encoding="utf-8")
        profile.profile_version = version + 1
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    if change is not None:
        (history_dir / f"diff_v{profile.profile_version}.json").write_text(
            json.dumps(
                {"added": change.added, "removed": change.removed, "changed": change.changed},
                indent=2,
            ),
            encoding="utf-8",
        )
    return change


def diff_profiles(previous: CanonicalProfile | None, current: CanonicalProfile) -> ProfileDiff:
    change = ProfileDiff()
    if previous is None:
        change.added = [skill.name for skill in current.skills]
        return change
    old_skills = {skill.name for skill in previous.skills}
    new_skills = {skill.name for skill in current.skills}
    full_text = _profile_text(current)
    for name in sorted(new_skills - old_skills):
        change.added.append(name)
    for name in sorted(old_skills - new_skills):
        if name.lower() not in full_text:
            change.removed.append(name)
    old_work = {(item.company, item.title, item.start_date, item.end_date) for item in previous.work_experience}
    new_work = {(item.company, item.title, item.start_date, item.end_date) for item in current.work_experience}
    if old_work != new_work:
        change.changed.append("work experience")
    else:
        change.unchanged.append("work experience")
    if previous.education == current.education:
        change.unchanged.append("education")
    else:
        change.changed.append("education")
    old_certs = set(previous.certifications)
    new_certs = set(current.certifications)
    if new_certs - old_certs:
        change.added.extend(sorted(new_certs - old_certs))
    if old_certs == new_certs:
        change.unchanged.append("certifications")
    return change


def _profile_text(profile: CanonicalProfile) -> str:
    parts = [profile.summary, *profile.education, *profile.certifications]
    for item in profile.work_experience:
        parts.extend([item.company, item.title, *item.responsibilities])
    for project in profile.projects:
        parts.extend([project.name, project.description, *project.evidence])
    for skill in profile.skills:
        for evidence in skill.evidence:
            parts.append(str(evidence.get("quote") or ""))
    return "\n".join(parts).lower()

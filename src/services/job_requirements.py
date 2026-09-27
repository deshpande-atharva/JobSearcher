"""Split posting requirements without treating every mentioned tool as required."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.services.resume_parse import mentioned_skills

__all__ = ["JobRequirements", "extract_job_requirements"]

_PREFERRED_HEADING = re.compile(
    r"^(preferred|nice to have|bonus|what you.d bring|nice-to-have)\b",
    re.I,
)
_REQUIRED_HEADING = re.compile(
    r"^(requirements|qualifications|minimum qualifications|basic qualifications|what you.ll need|what you.ll bring|what you will bring|what we.re looking for|what we are looking for|must have)\b",
    re.I,
)
_BODY_HEADING = re.compile(
    r"^(what you.ll do|what you will do|responsibilities|about the (?:team|company|role)|who we are)\b",
    re.I,
)
_PREFERRED_CLAUSE = re.compile(
    r"\b(?:preferred|nice to have|a plus|plus|bonus|ideally|optional)\b",
    re.I,
)
_REQUIRED_CLAUSE = re.compile(r"\b(?:required|must|minimum)\b", re.I)
_DEGREE = re.compile(
    r"\b(?:bachelor'?s|master'?s|ph\.?d|b\.s\.?|m\.s\.?|b\.e\.?|m\.e\.?|bs|ms|ba|degree)\b[^.\n]{0,120}",
    re.I,
)
_CLEARANCE = re.compile(r"\b(?:security clearance|ts/sci|top secret)\b", re.I)


@dataclass
class JobRequirements:
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    unknown_skills: list[str] = field(default_factory=list)
    education_text: str = ""
    clearance: str = ""


def extract_job_requirements(description: str | None) -> JobRequirements:
    required: list[str] = []
    preferred: list[str] = []
    unknown: list[str] = []
    section = "body"
    for raw in (description or "").splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line:
            continue
        if len(line) <= 80 and _PREFERRED_HEADING.match(line):
            section = "preferred"
            continue
        if len(line) <= 80 and _REQUIRED_HEADING.match(line):
            section = "required"
            continue
        if len(line) <= 80 and _BODY_HEADING.match(line):
            section = "body"
            continue
        for clause in re.split(r"[.;]", line):
            clause = clause.strip()
            if not clause:
                continue
            names = _drop_contained(mentioned_skills(clause))
            if not names:
                continue
            if _PREFERRED_CLAUSE.search(clause) or (section == "preferred" and not _REQUIRED_CLAUSE.search(clause)):
                preferred.extend(names)
            elif _REQUIRED_CLAUSE.search(clause) or section == "required":
                required.extend(names)
            else:
                unknown.extend(names)
    education = _DEGREE.search(description or "")
    clearance = _CLEARANCE.search(description or "")
    return JobRequirements(
        required_skills=_unique(required),
        preferred_skills=_unique(name for name in preferred if name not in set(required)),
        unknown_skills=_unique(name for name in unknown if name not in set(required) and name not in set(preferred)),
        education_text=education.group(0).strip() if education else "",
        clearance=clearance.group(0) if clearance else "",
    )


def _drop_contained(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if not any(other.lower().startswith(name.lower() + " ") for other in names if other != name)
    ]


def _unique(names) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
    return ordered

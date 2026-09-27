"""Deterministic resume structure: sections, dates, and explicit skill mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

__all__ = [
    "ALIASES",
    "EmploymentSpan",
    "SkillHit",
    "canonical_skill",
    "merge_months",
    "parse_spans",
    "scan_skills",
    "mentioned_skills",
    "split_sections",
]

_HEADINGS = {
    "summary": ("summary", "professional summary", "profile"),
    "experience": ("experience", "work experience", "professional experience", "employment"),
    "projects": ("projects", "personal projects", "selected projects"),
    "education": ("education",),
    "skills": ("skills", "technical skills"),
    "certifications": ("certifications", "certificates", "licenses"),
}

# Alias -> canonical. Only names that are the same technology.
ALIASES: dict[str, str] = {
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "react.js": "React",
    "reactjs": "React",
    "golang": "Go",
    "springboot": "Spring Boot",
    "spring boot": "Spring Boot",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "python": "Python",
    "java": "Java",
    "aws": "AWS",
    "terraform": "Terraform",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "rest": "REST",
    "c/c++": "C++",
    "c++": "C++",
    "html": "HTML",
    "css": "CSS",
    "shell scripting": "Shell Scripting",
    "google cloud": "Google Cloud",
    "express": "Express",
    "git": "Git",
    "jira": "JIRA",
    "postman": "Postman",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "linux": "Linux",
    "ubuntu": "Ubuntu",
    "s3": "S3",
    "electron": "Electron",
    "react query": "React Query",
    "trpc": "tRPC",
    "jest": "Jest",
    "azure": "Azure",
    "packer": "Packer",
    "github actions": "GitHub Actions",
    "fastapi": "FastAPI",
    "pytorch": "PyTorch",
    "nginx": "nginx",
    "cloudwatch": "CloudWatch",
    "rds": "RDS",
    "ebs": "EBS",
    "kms": "KMS",
    "secrets manager": "Secrets Manager",
    "kafka": "Kafka",
    "ec2": "EC2",
    "lambda": "Lambda",
    "ecs": "ECS",
    "eks": "EKS",
    "ruby on rails": "Ruby on Rails",
    "ruby": "Ruby",
    "rails": "Ruby on Rails",
    "graphql": "GraphQL",
}

# A mention of the key is enough to record the value as INFERRED, with the same bullet as evidence.
_IMPLY: dict[str, str] = {
    "Spring Boot": "Java",
}

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_DATE_RANGE = re.compile(
    r"(?P<start>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})"
    r"\s*[–—-]\s*"
    r"(?P<end>present|current|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})",
    re.I,
)


@dataclass
class SkillHit:
    name: str
    quote: str
    section: str
    status: str
    company: str = ""
    project: str = ""


@dataclass
class EmploymentSpan:
    title: str
    company: str
    start: date | None
    end: date | None
    raw: str
    kind: str
    bullets: list[str] = field(default_factory=list)
    ambiguous: bool = False


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"preamble": []}
    current = "preamble"
    for line in text.splitlines():
        key = _heading(line)
        if key:
            current = key
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if any(part.strip() for part in lines)}


def _heading(line: str) -> str | None:
    cleaned = re.sub(r"[^a-z ]", "", line.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for name, labels in _HEADINGS.items():
        if cleaned in labels:
            return name
    return None


def canonical_skill(token: str) -> str | None:
    key = token.strip().lower()
    return ALIASES.get(key)


def scan_skills(sections: dict[str, str]) -> list[SkillHit]:
    hits: list[SkillHit] = []
    seen: set[tuple[str, str]] = set()
    for section, body in sections.items():
        company = ""
        project = ""
        for line in body.splitlines():
            raw = line.strip()
            if (
                section == "experience"
                and raw
                and not raw.startswith(("-", "•"))
                and ("," in raw or "|" in raw)
            ):
                _, company = _split_header(raw)
            if section == "projects" and raw and not raw.startswith(("-", "•")):
                project = raw
            quote = raw.lstrip("-• ").strip()
            if not quote:
                continue
            found = _mentions(quote)
            for name in found:
                key = (name, quote.lower())
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    SkillHit(
                        name=name,
                        quote=quote,
                        section=section or "resume",
                        status="EXPLICIT",
                        company=company if section == "experience" else "",
                        project=project if section == "projects" else "",
                    )
                )
            for name in found:
                implied = _IMPLY.get(name)
                if not implied:
                    continue
                key = (implied, quote.lower())
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    SkillHit(
                        name=implied,
                        quote=quote,
                        section=section or "resume",
                        status="INFERRED",
                        company=company if section == "experience" else "",
                        project=project if section == "projects" else "",
                    )
                )
    return hits


def mentioned_skills(text: str) -> list[str]:
    return _mentions(text)


def _mentions(quote: str) -> list[str]:
    lowered = quote.lower()
    found: list[str] = []
    for alias, canonical in sorted(ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
            if canonical not in found:
                found.append(canonical)
    return found


def parse_spans(experience_text: str, *, today: date | None = None) -> list[EmploymentSpan]:
    today = today or date.today()
    lines = [line.strip() for line in experience_text.splitlines() if line.strip()]
    spans: list[EmploymentSpan] = []
    index = 0
    while index < len(lines):
        match = _DATE_RANGE.search(lines[index])
        if match is None:
            if re.search(r"\b(?:summer|fall|spring|winter)\s+\d{4}\b", lines[index], re.I):
                header = lines[index - 1] if index else ""
                title, company = _split_header(header)
                kind = "internship" if re.search(r"\bintern\b|\bco-op\b", header, re.I) else "professional"
                spans.append(
                    EmploymentSpan(
                        title=title,
                        company=company,
                        start=None,
                        end=None,
                        raw=lines[index],
                        kind=kind,
                        ambiguous=True,
                    )
                )
            index += 1
            continue
        prefix = lines[index][: match.start()].strip(" -|")
        header = prefix if prefix else (lines[index - 1] if index else "")
        title, company = _split_header(header)
        start = _parse_point(match.group("start"), today)
        end_raw = match.group("end")
        ambiguous = bool(re.search(r"summer|fall|spring|winter", lines[index], re.I)) and start is None
        if start is None or (end_raw.lower() not in {"present", "current"} and _parse_point(end_raw, today) is None):
            spans.append(
                EmploymentSpan(title, company, None, None, lines[index], _kind(header, lines[index]), ambiguous=True)
            )
            index += 1
            continue
        end = today if end_raw.lower() in {"present", "current"} else _parse_point(end_raw, today)
        bullets: list[str] = []
        index += 1
        while index < len(lines) and _DATE_RANGE.search(lines[index]) is None and _heading(lines[index]) is None:
            if lines[index].startswith(("-", "•")) or not _looks_like_header(lines[index]):
                bullets.append(lines[index].lstrip("-• ").strip())
            else:
                break
            index += 1
        spans.append(EmploymentSpan(title, company, start, end, match.group(0), _kind(header + " " + " ".join(bullets), lines[index - 1] if index else ""), bullets))
    return spans


def _split_header(header: str) -> tuple[str, str]:
    header = _DATE_RANGE.sub("", header).strip(" -|")
    if "|" in header:
        company, title = header.split("|", 1)
        return title.strip(), company.strip()
    if "," in header:
        title, company = header.split(",", 1)
        return title.strip(), company.strip()
    return header.strip(), ""


def _kind(text: str, raw: str) -> str:
    blob = f"{text} {raw}".lower()
    if re.search(r"\bintern(?:ship)?\b|\bco-op\b|\bcoop\b", blob):
        return "internship"
    return "professional"


def _looks_like_header(line: str) -> bool:
    return "," in line and not line.startswith(("-", "•"))


def _parse_point(value: str, today: date) -> date | None:
    token = value.strip().lower().rstrip(".")
    if token in {"present", "current"}:
        return today
    month_match = re.match(r"([a-z]+)\s+(\d{4})", token)
    if month_match:
        month = _MONTHS.get(month_match.group(1)[:4]) or _MONTHS.get(month_match.group(1)[:3])
        if month is None:
            return None
        return date(int(month_match.group(2)), month, 1)
    if re.fullmatch(r"\d{4}", token):
        return date(int(token), 1, 1)
    return None


def merge_months(spans: list[EmploymentSpan], *, kind: str) -> int:
    """Months of employment after merging overlaps. Projects are not included."""
    ranges = [(span.start, span.end) for span in spans if span.kind == kind and span.start and span.end and not span.ambiguous]
    ranges = [(start, end) for start, end in ranges if end >= start]
    if not ranges:
        return 0
    ranges.sort()
    merged: list[tuple[date, date]] = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    months = 0
    for start, end in merged:
        months += (end.year - start.year) * 12 + (end.month - start.month)
    return max(months, 0)

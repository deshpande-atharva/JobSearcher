"""Turn resume text into a versioned profile. Facts must appear in the text."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.services.resume_parse import SkillHit, merge_months, parse_spans, scan_skills, split_sections

__all__ = ["CanonicalProfile", "ResumeFacts", "interpret_resume"]


class ResumeMeta(BaseModel):
    filename: str
    sha256: str
    processed_at: str
    page_count: int = 1


class WorkItem(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    kind: str = "professional"
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    ambiguous_dates: bool = False


class ProjectItem(BaseModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SkillItem(BaseModel):
    name: str
    confidence: str
    status: str
    category: str = "other"
    evidence: list[dict[str, str]] = Field(default_factory=list)


class RoleFamilyItem(BaseModel):
    role_family: str
    confidence: str
    evidence: list[str] = Field(default_factory=list)


class CanonicalProfile(BaseModel):
    profile_version: int = 1
    resume: ResumeMeta
    summary: str = ""
    contact_name: str = ""
    contact_email: str = ""
    contact_phone: str = ""
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    work_experience: list[WorkItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    skills: list[SkillItem] = Field(default_factory=list)
    role_families: list[RoleFamilyItem] = Field(default_factory=list)
    professional_months: int = 0
    internship_months: int = 0


class ResumeFacts(BaseModel):
    """Optional model output. Every quote must already appear in the resume."""

    summary: str = ""
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    skills: list[dict[str, str]] = Field(default_factory=list)


_CATEGORIES = {
    "Java": "languages",
    "Python": "languages",
    "JavaScript": "languages",
    "TypeScript": "languages",
    "Go": "languages",
    "C++": "languages",
    "HTML": "languages",
    "CSS": "languages",
    "Shell Scripting": "languages",
    "Spring Boot": "backend",
    "Node.js": "backend",
    "REST APIs": "backend",
    "REST": "backend",
    "Express": "backend",
    "FastAPI": "backend",
    "React": "frontend",
    "React Query": "frontend",
    "AWS": "cloud",
    "S3": "cloud",
    "Azure": "cloud",
    "Google Cloud": "cloud",
    "CloudWatch": "cloud",
    "RDS": "cloud",
    "EBS": "cloud",
    "KMS": "cloud",
    "Secrets Manager": "cloud",
    "EC2": "cloud",
    "Lambda": "cloud",
    "Terraform": "infrastructure",
    "Kubernetes": "infrastructure",
    "Docker": "infrastructure",
    "Packer": "infrastructure",
    "nginx": "infrastructure",
    "GitHub Actions": "infrastructure",
    "PostgreSQL": "databases",
    "MySQL": "databases",
    "MongoDB": "databases",
    "Redis": "databases",
    "Kafka": "other",
}


_ROLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("backend_engineering", ("spring boot", "rest", "postgresql", "node.js", "django", "express", "fastapi", "golang")),
    ("frontend_engineering", ("react", "html", "css")),
    ("cloud_engineering", ("aws", "terraform", "azure", "google cloud")),
    ("infrastructure_engineering", ("kubernetes", "docker", "terraform")),
    ("software_engineering", ("java", "python", "javascript", "typescript", "golang")),
)


async def interpret_resume(
    text: str,
    *,
    filename: str,
    sha256: str,
    page_count: int,
    llm: Any | None = None,
    today: date | None = None,
) -> CanonicalProfile:
    sections = split_sections(text)
    hits = scan_skills(sections)
    work = _work_items(sections.get("experience", ""), hits, today=today)
    projects = _projects(sections.get("projects", ""), hits)
    if llm is not None and getattr(llm, "available", False):
        extra = await llm.structured(
            prompt=_prompt(sections),
            response_model=ResumeFacts,
            system=(
                "Extract only facts supported by a verbatim quote from the resume. "
                "Do not invent employers, skills, dates, or metrics. "
                "If unsure, omit the item."
            ),
            purpose="resume_interpretation",
        )
        if isinstance(extra, ResumeFacts):
            hits = _merge_quotes(hits, extra, text)
            if extra.summary and extra.summary.lower() in text.lower():
                summary = extra.summary.strip()
            else:
                summary = _first_paragraph(sections.get("summary", ""))
            education = _ground_lines(extra.education, text) or _lines(sections.get("education", ""))
            certifications = _ground_lines(extra.certifications, text) or _lines(sections.get("certifications", ""))
        else:
            summary = _first_paragraph(sections.get("summary", ""))
            education = _lines(sections.get("education", ""))
            certifications = _lines(sections.get("certifications", ""))
    else:
        summary = _first_paragraph(sections.get("summary", ""))
        education = _lines(sections.get("education", ""))
        certifications = _lines(sections.get("certifications", ""))
    skills = _skill_items(hits)
    email, phone, name = _contact(text)
    return CanonicalProfile(
        resume=ResumeMeta(
            filename=filename,
            sha256=sha256,
            processed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            page_count=page_count,
        ),
        summary=summary,
        contact_name=name,
        contact_email=email,
        contact_phone=phone,
        education=education,
        certifications=certifications,
        work_experience=work,
        projects=projects,
        skills=skills,
        role_families=_families(hits, work),
        professional_months=merge_months(parse_spans(sections.get("experience", ""), today=today), kind="professional"),
        internship_months=merge_months(parse_spans(sections.get("experience", ""), today=today), kind="internship"),
    )


def _work_items(text: str, hits: list[SkillHit], *, today: date | None) -> list[WorkItem]:
    items: list[WorkItem] = []
    for span in parse_spans(text, today=today):
        technologies = [hit.name for hit in hits if hit.section == "experience" and hit.quote in span.bullets]
        items.append(
            WorkItem(
                company=span.company,
                title=span.title,
                start_date=span.start.isoformat() if span.start else "",
                end_date=span.end.isoformat() if span.end else "",
                kind=span.kind,
                responsibilities=span.bullets,
                technologies=list(dict.fromkeys(technologies)),
                achievements=[bullet for bullet in span.bullets if any(ch.isdigit() for ch in bullet)],
                ambiguous_dates=span.ambiguous,
            )
        )
    return items


def _projects(text: str, hits: list[SkillHit]) -> list[ProjectItem]:
    projects: list[ProjectItem] = []
    name = ""
    bullets: list[str] = []

    def flush() -> None:
        if not name and not bullets:
            return
        quotes = bullets or ([name] if name else [])
        technologies = [hit.name for hit in hits if hit.quote in quotes]
        projects.append(
            ProjectItem(
                name=name or "Project",
                description=" ".join(bullets),
                technologies=list(dict.fromkeys(technologies)),
                evidence=quotes,
            )
        )

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "•")):
            bullets.append(stripped.lstrip("-• ").strip())
        else:
            flush()
            name = stripped
            bullets = []
    flush()
    return projects


def _skill_items(hits: list[SkillHit]) -> list[SkillItem]:
    grouped: dict[str, SkillItem] = {}
    for hit in hits:
        item = grouped.get(hit.name)
        evidence = {
            "source": hit.section,
            "quote": hit.quote,
            "status": hit.status,
            "company": hit.company,
            "project": hit.project,
        }
        if item is None:
            grouped[hit.name] = SkillItem(
                name=hit.name,
                confidence="high" if hit.status == "EXPLICIT" else "medium",
                status=hit.status,
                category=_CATEGORIES.get(hit.name, "other"),
                evidence=[evidence],
            )
        elif evidence not in item.evidence:
            item.evidence.append(evidence)
            if hit.status == "EXPLICIT":
                item.status = "EXPLICIT"
                item.confidence = "high"
    return list(grouped.values())


def _families(hits: list[SkillHit], work: list[WorkItem]) -> list[RoleFamilyItem]:
    blob = " ".join(hit.quote.lower() for hit in hits)
    families: list[RoleFamilyItem] = []
    for name, needles in _ROLE_RULES:
        evidence = [hit.quote for hit in hits if any(needle in hit.quote.lower() for needle in needles)]
        if evidence:
            families.append(RoleFamilyItem(role_family=name, confidence="high", evidence=list(dict.fromkeys(evidence))[:4]))
    titles = [item.title for item in work if re.search(r"full[\s-]?stack", item.title, re.I)]
    frontend = any(token in blob for token in ("react", "javascript", "typescript", "html", "css"))
    backend = "backend_engineering" in {item.role_family for item in families}
    if titles or (backend and frontend):
        evidence = titles or [
            hit.quote for hit in hits if any(token in hit.quote.lower() for token in ("react", "javascript", "node"))
        ]
        families.append(
            RoleFamilyItem(
                role_family="full_stack_engineering",
                confidence="high" if titles else "medium",
                evidence=list(dict.fromkeys(evidence))[:3],
            )
        )
    return families


def _merge_quotes(hits: list[SkillHit], extra: ResumeFacts, text: str) -> list[SkillHit]:
    lowered = text.lower()
    merged = list(hits)
    for item in extra.skills:
        name = str(item.get("name") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not name or not quote or quote.lower() not in lowered:
            continue
        if name.lower() not in quote.lower() and name.lower() not in lowered:
            continue
        merged.append(SkillHit(name=name, quote=quote, section="resume", status="EXPLICIT"))
    return merged


def _ground_lines(lines: list[str], text: str) -> list[str]:
    lowered = text.lower()
    return [line.strip() for line in lines if line.strip() and line.strip().lower() in lowered]


def _lines(text: str) -> list[str]:
    return [line.strip().lstrip("-• ").strip() for line in text.splitlines() if line.strip()]


def _first_paragraph(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())[:500]


def _contact(text: str) -> tuple[str, str, str]:
    """Only values that appear verbatim. A name is kept from a Name label or an all-caps header line."""
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
    phone_match = re.search(r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}", text)
    name_match = re.search(r"(?im)^\s*name\s*:\s*(.+?)\s*$", text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""
    if name_match:
        name = name_match.group(1).strip()
    else:
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        name = first if re.fullmatch(r"[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,3}", first) else ""
    return email, phone, name


def _prompt(sections: dict[str, str]) -> str:
    chunks = []
    for name, body in sections.items():
        chunks.append(f"[{name}]\n{body[:1500]}")
    return "Resume sections:\n" + "\n\n".join(chunks)

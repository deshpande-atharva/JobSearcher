"""Deterministic Workday job-detail extraction.

Headings stay intact until sections are known. The text handed to Job
Intelligence uses the same heading words the shared requirement parser already
understands. Missing fields stay unknown.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from src.models.job import DateSource
from src.services.h1b import scan_sponsorship_language
from src.services.job_requirements import extract_job_requirements
from src.utils.dates import parse_labeled_job_date
from src.utils.urls import extract_job_id

__all__ = ["WorkdayJobDetail", "parse_workday_detail"]

_RESPONSIBILITY = re.compile(
    r"^(?:what you(?:'ll| will)(?: be)? do(?:ing)?|responsibilities|about the role|role responsibilities)\b",
    re.I,
)
_REQUIRED = re.compile(
    r"^(?:what we need to see|what you'll need|minimum qualifications|basic qualifications|"
    r"required qualifications|requirements|qualifications|must have)\b",
    re.I,
)
_PREFERRED = re.compile(
    r"^(?:ways to stand out|preferred qualifications|preferred requirements|nice to have|bonus)\b",
    re.I,
)
_CHROME = re.compile(
    r"^(?:sign in|search for jobs|similar jobs|apply|skip to main content|follow us|view all.*jobs)\b",
    re.I,
)
_YEARS = re.compile(r"\b\d+\s*(?:\+|-\s*\d+)?\s*(?:years?|yrs?)\b", re.I)
_MULTI_LOCATION = re.compile(r"^\d+\s+locations?$", re.I)


@dataclass
class WorkdayJobDetail:
    job_id: str = ""
    title: str = ""
    company: str = ""
    official_url: str = ""
    location: str = "UNKNOWN"
    description: str = ""
    responsibilities: list[str] = field(default_factory=list)
    required_qualifications: list[str] = field(default_factory=list)
    preferred_qualifications: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    experience_requirements: list[str] = field(default_factory=list)
    education_requirements: str = "UNKNOWN"
    employment_type: str = "UNKNOWN"
    work_arrangement: str = "UNKNOWN"
    clearance_requirements: str = "UNKNOWN"
    sponsorship_language: str = "ABSENT"
    posted_at_raw: str = ""
    updated_at_raw: str = ""
    date_source: DateSource = DateSource.UNKNOWN
    ambiguous: bool = False


def parse_workday_detail(
    description_html: str | None,
    *,
    title: str = "",
    company: str = "",
    official_url: str = "",
    job_id: str = "",
    locations_text: str = "",
    employment_text: str = "",
    posted_text: str = "",
    requisition_text: str = "",
) -> WorkdayJobDetail:
    """Turn one rendered Workday detail fragment into structured posting text."""
    sections = _sections_from_html(description_html or "")
    description = _render(sections)
    requirements = extract_job_requirements(description)
    location = _location(locations_text)
    posted_raw, updated_raw = _split_date_text(posted_text or "")
    posted_at = parse_labeled_job_date(posted_raw) if posted_raw else None
    updated_at = parse_labeled_job_date(updated_raw) if updated_raw else None
    experience = [
        line
        for line in sections.get("required", []) + sections.get("preferred", [])
        if _YEARS.search(line)
    ]
    education = requirements.education_text or "UNKNOWN"
    arrangement = "UNKNOWN"
    if location != "UNKNOWN" and re.search(r"\bremote\b", location, re.I):
        arrangement = "remote"
    elif location != "UNKNOWN" and re.search(r"\bhybrid\b", location, re.I):
        arrangement = "hybrid"
    resolved_id = (job_id or "").strip() or _requisition(requisition_text) or extract_job_id(official_url) or ""
    ambiguous = bool((description_html or "").strip()) and not (
        sections.get("required") or sections.get("responsibilities") or sections.get("preferred")
    )
    return WorkdayJobDetail(
        job_id=resolved_id,
        title=" ".join((title or "").split()),
        company=company,
        official_url=official_url,
        location=location,
        description=description,
        responsibilities=sections.get("responsibilities", []),
        required_qualifications=sections.get("required", []),
        preferred_qualifications=sections.get("preferred", []),
        required_skills=requirements.required_skills,
        preferred_skills=requirements.preferred_skills,
        experience_requirements=experience,
        education_requirements=education,
        employment_type=_employment(employment_text),
        work_arrangement=arrangement,
        clearance_requirements=requirements.clearance or "UNKNOWN",
        sponsorship_language=scan_sponsorship_language(description).polarity.value,
        posted_at_raw=posted_raw,
        updated_at_raw=updated_raw,
        date_source=(
            DateSource.POSTED_DATE
            if posted_at is not None
            else DateSource.UPDATED_DATE
            if updated_at is not None
            else DateSource.UNKNOWN
        ),
        ambiguous=ambiguous,
    )


def _sections_from_html(raw: str) -> dict[str, list[str]]:
    text = raw or ""
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    parser = _DescriptionParser()
    parser.feed(text)
    parser.close()
    sections: dict[str, list[str]] = {
        "intro": [],
        "responsibilities": [],
        "required": [],
        "preferred": [],
    }
    current = "intro"
    for kind, value in parser.blocks:
        if _CHROME.match(value):
            continue
        if kind == "heading":
            mapped = _heading_kind(value)
            if mapped:
                current = mapped
            continue
        if not value:
            continue
        sections[current].append(value)
    return sections


def _split_date_text(text: str) -> tuple[str, str]:
    """Separate posted and updated clauses. A missing clause stays empty."""
    posted: list[str] = []
    updated: list[str] = []
    chunks = re.split(r"[\n|]+|(?=\b(?:posted|updated)\b)", text or "", flags=re.I)
    for chunk in chunks:
        piece = " ".join(chunk.split())
        if not piece:
            continue
        if re.search(r"\bupdated\b", piece, re.I) and not re.search(r"\bposted\b", piece, re.I):
            updated.append(piece)
        else:
            posted.append(piece)
    return " ".join(posted), " ".join(updated)


def _heading_kind(value: str) -> str | None:
    text = value.strip().rstrip(":")
    if _PREFERRED.match(text):
        return "preferred"
    if _REQUIRED.match(text):
        return "required"
    if _RESPONSIBILITY.match(text):
        return "responsibilities"
    return None


def _render(sections: dict[str, list[str]]) -> str:
    chunks: list[str] = []
    if sections["intro"]:
        chunks.extend(sections["intro"])
    mapping = (
        ("responsibilities", "Responsibilities"),
        ("required", "Minimum Qualifications"),
        ("preferred", "Preferred Qualifications"),
    )
    for key, heading in mapping:
        lines = sections.get(key) or []
        if not lines:
            continue
        chunks.append(heading)
        chunks.extend(f"- {line}" for line in lines)
    return "\n".join(chunks).strip()


def _location(raw: str) -> str:
    text = " ".join((raw or "").split())
    text = re.sub(r"^locations?\s+", "", text, flags=re.I).strip()
    text = re.sub(r"\bview all \d+ locations\b", "", text, flags=re.I).strip()
    if not text or _MULTI_LOCATION.match(text):
        return "UNKNOWN"
    return text


def _employment(raw: str) -> str:
    text = " ".join((raw or "").split())
    match = re.search(r"time type\s+(.+)$", text, flags=re.I)
    value = match.group(1).strip() if match else text
    value = re.sub(r"^time type\s+", "", value, flags=re.I).strip()
    return value or "UNKNOWN"


def _requisition(raw: str) -> str:
    match = re.search(r"\b((?:JR|R|REQ)-?\d{3,}(?:-\d+)?)\b", raw or "", flags=re.I)
    return match.group(1) if match else ""


class _DescriptionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._buf: list[str] = []
        self._li = 0
        self._bold = 0
        self._heading = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"br"}:
            self._buf.append(" ")
        if tag == "li":
            self._flush(kind="text")
            self._li += 1
        if tag in {"b", "strong"}:
            self._bold += 1
        if tag in {"p", "div", "h1", "h2", "h3", "h4"} and self._li == 0:
            self._flush(kind="text")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"b", "strong"}:
            self._bold = max(0, self._bold - 1)
            text = _clean("".join(self._buf))
            if text and len(text) <= 90:
                self._heading = text
                self._buf = []
        if tag == "li":
            self._flush(kind="bullet")
            self._li = max(0, self._li - 1)
        elif tag in {"p", "h1", "h2", "h3", "h4"} and self._li == 0:
            text = _clean("".join(self._buf))
            if self._heading and text == self._heading:
                self.blocks.append(("heading", self._heading))
                self._heading = ""
                self._buf = []
            else:
                self._flush(kind="text")
        elif tag == "div" and self._li == 0:
            self._flush(kind="text")

    def handle_data(self, data: str) -> None:
        if data:
            self._buf.append(data)

    def close(self) -> None:
        self._flush(kind="bullet" if self._li else "text")
        super().close()

    def _flush(self, *, kind: str) -> None:
        text = _clean("".join(self._buf))
        self._buf = []
        if self._heading and kind == "text" and not text:
            self.blocks.append(("heading", self._heading))
            self._heading = ""
            return
        if not text:
            return
        if self._heading and text == self._heading:
            self.blocks.append(("heading", text))
            self._heading = ""
            return
        self.blocks.append((kind, text))


def _clean(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())

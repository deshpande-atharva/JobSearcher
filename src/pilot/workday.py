"""Workday browser discovery helpers.

Jobs found in the public UI become ``RawJobPosting`` / ``Job`` records. The
CXS HTTP adapter is not used here and is not labeled as browser discovery.
"""

from __future__ import annotations

import re

from src.agents.extraction_agent import _to_job
from src.browser.page_state import JobCard, PageState
from src.models.job import DateSource, Job, RawJobPosting
from src.services.deduplication import deduplicate
from src.services.url_verification import pick_direct_url
from src.utils.urls import UrlPolicy, extract_job_id

__all__ = [
    "WORKDAY_TENANTS",
    "browser_health",
    "cards_to_postings",
    "canonical_jobs",
    "extract_workday_job_id",
    "official_workday_url",
]

_BLOCKED_HOSTS = ("linkedin.com", "indeed.com", "glassdoor.com", "jobright.ai", "google.com/search")

# One public tenant is the Phase 1 reference. Fallbacks are used only when the
# first site is unreachable. None of these are added to the daily pipeline.
WORKDAY_TENANTS: tuple[tuple[str, str], ...] = (
    ("NVIDIA", "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite"),
    ("Adobe", "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"),
    ("Workday", "https://workday.wd5.myworkdayjobs.com/en-US/Workday"),
)


def extract_workday_job_id(url: str | None, visible: str = "") -> str:
    """Return the public Workday requisition id, or empty when it is not present."""
    from_url = extract_job_id(url)
    if from_url:
        return from_url
    return ""


def official_workday_url(url: str | None) -> tuple[str | None, str]:
    raw = (url or "").strip()
    if not raw.startswith(("https://", "http://")):
        return None, "UNKNOWN"
    lowered = raw.lower()
    if any(host in lowered for host in _BLOCKED_HOSTS):
        return None, "UNKNOWN"
    if "myworkdayjobs.com" in lowered or "myworkdaysite.com" in lowered:
        if "/job/" in lowered or extract_job_id(raw):
            return raw, "verified"
    return None, "UNKNOWN"


def cards_to_postings(page: PageState, company: str) -> list[RawJobPosting]:
    postings: list[RawJobPosting] = []
    for card in page.job_cards:
        postings.append(card_to_posting(card, company, page.url))
    return postings


def _visible_location(raw: str | None) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    compact = " ".join(text.split())
    if re.fullmatch(r"locations?\s+\d+\s+locations?", compact, flags=re.I):
        return None
    cleaned = re.sub(r"^locations?\s+", "", compact, flags=re.I).strip()
    return cleaned or None


def card_to_posting(card: JobCard, company: str, source_url: str) -> RawJobPosting:
    url, status = official_workday_url(card.url)
    job_id = (card.job_id or "").strip() or extract_workday_job_id(card.url)
    location = _visible_location(card.location)
    return RawJobPosting(
        source="workday",
        company_name=card.company or company,
        title=card.title or None,
        location_raw=location,
        job_id=job_id or None,
        apply_url=url,
        date_source=DateSource.UNKNOWN,
        provenance={
            "ats": "workday",
            "discovery_method": "browser",
            "canonical_source": "workday",
            "discovered_from": ["workday"],
            "source_url": source_url,
            "url_status": status,
            "company": card.company or company,
            "job_id": job_id or "",
            "official_url": url or "",
        },
    )


def canonical_jobs(postings: list[RawJobPosting], policy: UrlPolicy | None = None) -> list[Job]:
    """Convert browser postings into the shared Job model and reuse dedup."""
    jobs: list[Job] = []
    for posting in postings:
        if policy is not None and posting.apply_url:
            check = pick_direct_url(posting, policy)
            if not check.accepted:
                posting.apply_url = None
        job = _to_job(posting)
        if job is None:
            continue
        jobs.append(job)
    unique, _duplicates, _historical = deduplicate(jobs)
    return unique


def browser_health(
    *,
    blocked: str = "",
    jobs: int = 0,
    navigated: bool = False,
    parser_ok: bool = True,
) -> str:
    """Map a browser attempt onto the existing source-health vocabulary."""
    if blocked:
        return "BLOCKED"
    if not parser_ok:
        return "ERROR"
    if not navigated:
        return "ERROR"
    if jobs == 0:
        return "EMPTY"
    return "OK"

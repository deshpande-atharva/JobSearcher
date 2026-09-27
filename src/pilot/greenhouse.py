"""Greenhouse pilot: structured board API first, browser navigation only after that fails."""

from __future__ import annotations

from src.browser.page_state import JobCard, PageState
from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.sources.base import SourceContext
from src.sources.greenhouse import GreenhouseSource

__all__ = ["cards_to_postings", "discover_structured", "enrich_browser_jobs"]

_BLOCKED_HOSTS = ("linkedin.com", "indeed.com", "glassdoor.com", "jobright.ai")


async def discover_structured(ctx: SourceContext, company: CompanyConfig) -> list[RawJobPosting]:
    return await GreenhouseSource(ctx).discover(company)


def cards_to_postings(page: PageState, company: str) -> list[RawJobPosting]:
    postings: list[RawJobPosting] = []
    for card in page.job_cards:
        url, status = _official_url(card)
        postings.append(
            RawJobPosting(
                source="greenhouse",
                company_name=card.company or company,
                title=card.title or None,
                location_raw=card.location or None,
                description=card.description or None,
                job_id=card.job_id or None,
                apply_url=url,
                date_source=DateSource.UNKNOWN,
                provenance={
                    "ats": "greenhouse",
                    "discovery_method": "browser",
                    "canonical_source": "greenhouse",
                    "discovered_from": ["greenhouse"],
                    "source_url": page.url,
                    "url_status": status,
                },
            )
        )
    return postings


def _official_url(card: JobCard) -> tuple[str | None, str]:
    url = (card.url or "").strip()
    if not url.startswith(("https://", "http://")):
        return None, "UNKNOWN"
    lowered = url.lower()
    if any(host in lowered for host in _BLOCKED_HOSTS):
        return None, "UNKNOWN"
    if any(token in lowered for token in ("greenhouse.io", "gh_jid=", "/jobs/", "/positions/")):
        return url, "verified"
    return None, "UNKNOWN"


def enrich_browser_jobs(
    browser_jobs: list[RawJobPosting],
    structured_jobs: list[RawJobPosting],
) -> list[RawJobPosting]:
    """Copy dates and descriptions from the public API onto jobs the browser already found.

    Discovery method stays ``browser``. Jobs that were not on the page are not added.
    """
    by_id = {job.job_id: job for job in structured_jobs if job.job_id}
    for job in browser_jobs:
        extra = by_id.get(job.job_id or "")
        job.provenance["discovery_method"] = "browser"
        if extra is None:
            continue
        if job.posted_at is None and extra.posted_at is not None:
            job.posted_at = extra.posted_at
            job.posted_at_raw = extra.posted_at_raw
            job.date_source = extra.date_source
        if job.updated_at is None and extra.updated_at is not None:
            job.updated_at = extra.updated_at
            job.updated_at_raw = extra.updated_at_raw
            if job.posted_at is None:
                job.date_source = extra.date_source
        if extra.description and len(extra.description) > len(job.description or ""):
            job.description = extra.description
        job.location_raw = job.location_raw or extra.location_raw
        job.apply_url = job.apply_url or extra.apply_url
        job.provenance["metadata_source"] = "greenhouse_structured"
        discovered = list(job.provenance.get("discovered_from") or ["greenhouse"])
        if "greenhouse" not in discovered:
            discovered.append("greenhouse")
        job.provenance["discovered_from"] = discovered
    return browser_jobs

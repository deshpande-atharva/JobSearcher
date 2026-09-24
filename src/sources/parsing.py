"""Shared HTML/JSON parsing helpers for discovery adapters.

Most career sites expose one of a handful of machine-readable shapes:

* a ``schema.org/JobPosting`` JSON-LD block (the most reliable, and what Google
  for Jobs consumes),
* a framework hydration payload such as Next.js ``__NEXT_DATA__``,
* or nothing, leaving link extraction as the fallback.

Parsing them here keeps every adapter short and means a fix benefits all of
them. Scraped markup is treated purely as data: nothing is evaluated, and no
embedded script is executed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

from src.models.job import DateSource, RawJobPosting
from src.utils.dates import parse_datetime
from src.utils.logging import get_logger
from src.utils.normalization import clean_text, html_to_text
from src.utils.urls import is_http_url, join_url

__all__ = [
    "extract_job_links",
    "extract_json_ld_job_postings",
    "extract_next_data",
    "iter_dicts",
    "json_ld_to_raw_posting",
    "parse_html",
    "pick",
]

log = get_logger(__name__)

# `html.parser` is stdlib-only, which avoids a compiled lxml dependency in CI.
_PARSER = "html.parser"

_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE
)
_SELF_XHR_JSON_RE = re.compile(
    r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE
)


def parse_html(html: str) -> BeautifulSoup:
    """Parse markup with the stdlib parser."""
    return BeautifulSoup(html or "", _PARSER)


def pick(payload: Any, *keys: str, default: Any = None) -> Any:
    """First non-empty value among ``keys`` in a mapping, case-insensitively."""
    if not isinstance(payload, dict):
        return default
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, "", [], {}):
            return value
    return default


def iter_dicts(node: Any, *, max_depth: int = 12) -> list[dict[str, Any]]:
    """Flatten every nested mapping in a JSON structure.

    Hydration payloads bury job arrays at unpredictable depths; walking the whole
    tree is more robust than guessing a path that changes with each redeploy.
    """
    found: list[dict[str, Any]] = []

    def walk(current: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(current, dict):
            found.append(current)
            for value in current.values():
                walk(value, depth + 1)
        elif isinstance(current, list):
            for item in current:
                walk(item, depth + 1)

    walk(node, 0)
    return found


# ---------------------------------------------------------------------------
# JSON-LD
# ---------------------------------------------------------------------------


def _json_ld_blocks(html: str) -> list[Any]:
    soup = parse_html(html)
    blocks: list[Any] = []
    for tag in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.IGNORECASE)}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            blocks.append(json.loads(raw))
        except (ValueError, TypeError):
            # Some sites emit JSON-LD with trailing commas or embedded newlines
            # inside strings. Salvage what we can and move on.
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                blocks.append(json.loads(cleaned))
            except (ValueError, TypeError):
                log.debug("skipping unparseable JSON-LD block")
    return blocks


def _is_job_posting(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    node_type = node.get("@type") or node.get("type")
    if isinstance(node_type, list):
        return any(str(t).lower() == "jobposting" for t in node_type)
    return str(node_type or "").lower() == "jobposting"


def extract_json_ld_job_postings(html: str) -> list[dict[str, Any]]:
    """Return every ``schema.org/JobPosting`` object found in the markup."""
    postings: list[dict[str, Any]] = []
    for block in _json_ld_blocks(html):
        for candidate in iter_dicts(block):
            if _is_job_posting(candidate):
                postings.append(candidate)
    return postings


def _json_ld_location(payload: dict[str, Any]) -> tuple[str, bool]:
    """Render ``jobLocation`` as text; second value flags remote-only postings."""
    remote = False
    job_type = pick(payload, "jobLocationType")
    if job_type and "telecommute" in str(job_type).lower():
        remote = True

    locations = pick(payload, "jobLocation", default=[])
    if isinstance(locations, dict):
        locations = [locations]
    if not isinstance(locations, list):
        locations = []

    labels: list[str] = []
    for entry in locations:
        if isinstance(entry, str):
            labels.append(entry)
            continue
        address = pick(entry, "address", default={}) if isinstance(entry, dict) else {}
        if isinstance(address, str):
            labels.append(address)
            continue
        city = pick(address, "addressLocality")
        region = pick(address, "addressRegion")
        country = pick(address, "addressCountry")
        if isinstance(country, dict):
            country = pick(country, "name", "identifier")
        parts = [str(p) for p in (city, region, country) if p]
        if parts:
            labels.append(", ".join(parts))

    if not labels:
        applicant_region = pick(payload, "applicantLocationRequirements", default=None)
        if isinstance(applicant_region, dict):
            name = pick(applicant_region, "name")
            if name:
                labels.append(str(name))
        elif isinstance(applicant_region, list):
            labels.extend(
                str(pick(item, "name"))
                for item in applicant_region
                if isinstance(item, dict) and pick(item, "name")
            )

    label = "; ".join(dict.fromkeys(labels))
    if remote and label:
        label = f"Remote - {label}"
    elif remote:
        label = "Remote"
    return label, remote


def json_ld_to_raw_posting(
    payload: dict[str, Any],
    *,
    source: str,
    page_url: str,
    fallback_company: str | None = None,
) -> RawJobPosting | None:
    """Convert a ``JobPosting`` JSON-LD object into a :class:`RawJobPosting`.

    Returns ``None`` when the block lacks both a title and a URL, which means it
    is not describing a real posting. No field is invented: absent data stays
    ``None`` so downstream agents can see it is missing.
    """
    title = clean_text(str(pick(payload, "title", "name") or "")) or None

    hiring_org = pick(payload, "hiringOrganization", default={})
    if isinstance(hiring_org, str):
        company = clean_text(hiring_org)
    else:
        company = clean_text(str(pick(hiring_org, "name", "legalName") or ""))
    company = company or (clean_text(fallback_company or "") or None)

    url_value = pick(payload, "url", "sameAs", "applicationUrl", "directApply")
    apply_url = ""
    if isinstance(url_value, str) and url_value.strip():
        apply_url = join_url(page_url, url_value) if not is_http_url(url_value) else url_value.strip()

    if not title and not apply_url:
        return None

    location, _ = _json_ld_location(payload)
    description = html_to_text(str(pick(payload, "description", default="") or ""))

    posted_raw = pick(payload, "datePosted")
    updated_raw = pick(payload, "dateModified", "dateUpdated")
    posted_at = parse_datetime(posted_raw)
    updated_at = parse_datetime(updated_raw)

    if posted_at is not None:
        date_source = DateSource.POSTED_DATE
    elif updated_at is not None:
        # An update timestamp is never relabelled as the original posting date.
        date_source = DateSource.UPDATED_DATE
    else:
        date_source = DateSource.UNKNOWN

    employment_type = pick(payload, "employmentType")
    if isinstance(employment_type, list):
        employment_type = ", ".join(str(item) for item in employment_type)

    identifier = pick(payload, "identifier", default=None)
    if isinstance(identifier, dict):
        identifier = pick(identifier, "value", "name")
    job_id = str(identifier).strip() if identifier else None

    return RawJobPosting(
        source=source,
        company_name=company,
        title=title,
        location_raw=location or None,
        description=description or None,
        employment_type_raw=str(employment_type) if employment_type else None,
        job_id=job_id,
        apply_url=apply_url or None,
        posted_at_raw=str(posted_raw) if posted_raw else None,
        updated_at_raw=str(updated_raw) if updated_raw else None,
        posted_at=posted_at,
        updated_at=updated_at,
        date_source=date_source,
        provenance={"parser": "json-ld", "page_url": page_url},
    )


# ---------------------------------------------------------------------------
# Framework hydration payloads
# ---------------------------------------------------------------------------


def extract_next_data(html: str) -> Any | None:
    """Return the parsed ``__NEXT_DATA__`` payload, if the page has one."""
    match = _NEXT_DATA_RE.search(html or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (ValueError, TypeError):
        log.debug("__NEXT_DATA__ present but not valid JSON")
        return None


def extract_embedded_json_blobs(html: str, *, limit: int = 20) -> list[Any]:
    """Parse ``<script type="application/json">`` blocks.

    Covers Nuxt, Remix, Astro and hand-rolled hydration without needing a
    per-framework adapter.
    """
    blobs: list[Any] = []
    for match in _SELF_XHR_JSON_RE.finditer(html or ""):
        if len(blobs) >= limit:
            break
        raw = match.group(1).strip()
        if not raw or len(raw) < 32:
            continue
        try:
            blobs.append(json.loads(raw))
        except (ValueError, TypeError):
            continue
    return blobs


# ---------------------------------------------------------------------------
# Link harvesting
# ---------------------------------------------------------------------------

_JOB_LINK_HINTS = (
    "/jobs/",
    "/job/",
    "/careers/",
    "/career/",
    "/opening",
    "/position",
    "/vacancy",
    "gh_jid=",
    "jobid=",
    "requisition",
    "/postings/",
)


def extract_job_links(html: str, page_url: str, *, limit: int = 400) -> list[tuple[str, str]]:
    """Harvest ``(url, link_text)`` pairs that plausibly point at postings.

    Only used when a page offers no structured data. Heuristic by nature, so the
    URL verification agent still has to validate whatever comes back.
    """
    soup = parse_html(html)
    seen: set[str] = set()
    links: list[tuple[str, str]] = []

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = join_url(page_url, href)
        if not is_http_url(absolute):
            continue
        lowered = absolute.lower()
        if not any(hint in lowered for hint in _JOB_LINK_HINTS):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append((absolute, clean_text(anchor.get_text(" ", strip=True))))
        if len(links) >= limit:
            break

    return links

"""Derive a public ATS type and board identifier from a careers URL or page.

Identifiers are extracted from real URLs and markup only. Nothing is guessed.
Authentication walls, CAPTCHAs and blocked pages are reported as a failed
detection, not worked around.

Confidence is not invented. It is assigned from the evidence class:

* ``1.0`` — identifier is in the careers URL itself
* ``0.9`` — identifier is in a redirect / final fetched URL
* ``0.8`` — identifier is in public HTML (canonical link, embed, JSON, page source)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from src.models.config import AtsType
from src.utils.urls import is_http_url

__all__ = [
    "ATSDiscoveryResult",
    "AtsDetection",
    "CONFIDENCE_FROM_HTML",
    "CONFIDENCE_FROM_REDIRECT",
    "CONFIDENCE_FROM_URL",
    "detect_ats",
    "detect_ats_from_html",
    "detect_ats_from_url",
    "is_valid_ats_identifier",
]

CONFIDENCE_FROM_URL = 1.0
CONFIDENCE_FROM_REDIRECT = 0.9
CONFIDENCE_FROM_HTML = 0.8

_GREENHOUSE = re.compile(
    r"https?://(?:job-boards|boards)(?:\.eu)?\.greenhouse\.io/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_GREENHOUSE_FOR = re.compile(
    r"greenhouse\.io/embed/job_board\?for=([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_GREENHOUSE_API = re.compile(
    r"boards-api\.greenhouse\.io/v1/boards/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_GREENHOUSE_RESERVED = frozenset({"embed", "job_board", "jobs", "boards", "v1"})
_LEVER = re.compile(
    r"https?://jobs(?:\.eu)?\.lever\.co/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_LEVER_API = re.compile(
    r"api\.lever\.co/v0/postings/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_ASHBY = re.compile(
    r"https?://jobs\.ashbyhq\.com/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_ASHBY_API = re.compile(
    r"api\.ashbyhq\.com/posting-api/job-board/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_SMART = re.compile(
    r"https?://(?:jobs|careers)\.smartrecruiters\.com/([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
_WORKDAY = re.compile(
    r"https?://[a-z0-9.-]+\.myworkday(?:jobs|site)\.com/[^\s\"'<>]+",
    re.IGNORECASE,
)
_ICIMS = re.compile(
    r"https?://[a-z0-9.-]+\.icims\.com/[^\s\"'<>]*",
    re.IGNORECASE,
)
_INVALID_TOKENS = frozenset(
    {
        "",
        "null",
        "none",
        "auto",
        "unknown",
        "embed",
        "job_board",
        "jobs",
        "boards",
        "v1",
        "postings",
        "job-board",
    }
)


@dataclass(frozen=True, slots=True)
class ATSDiscoveryResult:
    """Structured ATS detection. ``detected=False`` means no reliable identifier."""

    detected: bool
    ats_type: AtsType | None = None
    identifier: str | None = None
    confidence: float | None = None
    source_url: str = ""
    discovery_method: str = "automatic"
    evidence_url: str = ""
    method: str = "none"

    @property
    def ok(self) -> bool:
        return bool(
            self.detected
            and self.ats_type
            and self.identifier
            and is_valid_ats_identifier(self.ats_type, self.identifier)
        )


# Backward-compatible name used by existing tests and callers.
AtsDetection = ATSDiscoveryResult


def is_valid_ats_identifier(ats_type: str | None, identifier: str | None) -> bool:
    """Reject empty, reserved, or structurally impossible identifiers."""
    if not ats_type or not identifier:
        return False
    token = identifier.strip()
    if not token:
        return False
    lowered = token.strip("/").lower()
    if lowered in _INVALID_TOKENS:
        return False
    kind = ats_type.strip().lower()
    if kind == "workday":
        from src.sources.workday import parse_workday_site

        return parse_workday_site(token) is not None
    if kind == "icims":
        return "icims.com" in lowered and is_http_url(token)
    if kind in {"greenhouse", "lever", "ashby", "smartrecruiters"}:
        return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", token.strip("/")))
    return False


def detect_ats_from_url(url: str | None) -> ATSDiscoveryResult | None:
    """Read an ATS type/identifier out of a single URL. No network."""
    hit = detect_ats(url)
    return hit if hit.ok else None


def detect_ats(
    url: str | None,
    *,
    confidence: float = CONFIDENCE_FROM_URL,
    method: str = "url",
) -> ATSDiscoveryResult:
    """Always return a structured result. Identifiers are never guessed."""
    empty = ATSDiscoveryResult(detected=False, source_url=url or "", method=method)
    if not url or not is_http_url(url):
        return empty
    text = url.strip()
    parsed = urlparse(text)

    for_token = parse_qs(parsed.query).get("for", [])
    if "greenhouse.io" in parsed.netloc.lower() and for_token:
        return _hit("greenhouse", for_token[0], method, text, confidence)

    match = _GREENHOUSE.search(text)
    if match and match.group(1).lower() not in _GREENHOUSE_RESERVED:
        return _hit("greenhouse", match.group(1), method, text, confidence)

    match = _LEVER.search(text)
    if match:
        return _hit("lever", match.group(1), method, text, confidence)
    match = _ASHBY.search(text)
    if match:
        return _hit("ashby", match.group(1), method, text, confidence)
    match = _SMART.search(text)
    if match:
        return _hit("smartrecruiters", match.group(1), method, text, confidence)
    match = _WORKDAY.search(text)
    if match:
        identifier = _workday_site_url(match.group(0))
        if identifier:
            return _hit("workday", identifier, method, text, confidence)
    match = _ICIMS.search(text)
    if match:
        return _hit("icims", match.group(0).rstrip("\"'"), method, text, confidence)
    return empty


def detect_ats_from_html(html: str | None, *, page_url: str = "") -> ATSDiscoveryResult | None:
    """Scan public HTML for a concrete ATS board URL. First reliable hit wins."""
    from_page = detect_ats(page_url, confidence=CONFIDENCE_FROM_URL, method="url")
    if from_page.ok:
        return from_page
    if page_url:
        redirected = detect_ats(page_url, confidence=CONFIDENCE_FROM_REDIRECT, method="redirect")
        if redirected.ok and redirected.source_url.rstrip("/") != (page_url or "").rstrip("/"):
            return redirected
    if not html:
        return None

    match = _GREENHOUSE_FOR.search(html)
    if match:
        return _hit("greenhouse", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _GREENHOUSE_API.search(html)
    if match and match.group(1).lower() not in _GREENHOUSE_RESERVED:
        return _hit("greenhouse", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _GREENHOUSE.search(html)
    if match and match.group(1).lower() not in _GREENHOUSE_RESERVED:
        return _hit("greenhouse", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _LEVER.search(html)
    if match:
        return _hit("lever", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _LEVER_API.search(html)
    if match:
        return _hit("lever", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _ASHBY.search(html)
    if match:
        return _hit("ashby", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _ASHBY_API.search(html)
    if match:
        return _hit("ashby", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _SMART.search(html)
    if match:
        return _hit("smartrecruiters", match.group(1), "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _WORKDAY.search(html)
    if match:
        identifier = _workday_site_url(match.group(0))
        if identifier:
            return _hit("workday", identifier, "html", match.group(0), CONFIDENCE_FROM_HTML)
    match = _ICIMS.search(html)
    if match:
        return _hit("icims", match.group(0).rstrip("\"'"), "html", match.group(0), CONFIDENCE_FROM_HTML)
    return None


def _hit(
    ats_type: AtsType,
    identifier: str,
    method: str,
    evidence: str,
    confidence: float,
) -> ATSDiscoveryResult:
    token = identifier.strip()
    if not is_valid_ats_identifier(ats_type, token):
        return ATSDiscoveryResult(detected=False, source_url=evidence, method=method)
    discovery_method = "automatic" if method in {"url", "html", "redirect", "careers_url"} else method
    return ATSDiscoveryResult(
        detected=True,
        ats_type=ats_type,
        identifier=token,
        confidence=confidence,
        source_url=evidence,
        discovery_method=discovery_method,
        evidence_url=evidence,
        method=method,
    )


def _workday_site_url(raw: str) -> str | None:
    """Keep a Workday careers URL that still contains a site path segment."""
    cleaned = raw.rstrip("\"').,; ")
    if not is_http_url(cleaned):
        return None
    from src.sources.workday import parse_workday_site

    return cleaned if parse_workday_site(cleaned) else None

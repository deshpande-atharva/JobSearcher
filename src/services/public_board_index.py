"""Bounded public index of Greenhouse, Ashby, and Workday boards.

Greenhouse and Ashby do not publish a customer directory. Workday does not
publish a universal jobs API. This module reads a small, date-rotated sample
of URLs the Internet Archive has captured, then keeps only tokens or career
boards that match a strict public shape. Coverage is whatever that archive
sample contains.

Common Crawl's CDX API is not queried: its robots.txt disallows that path.
Tokens are not guessed from company names. A host is not a Workday board
just because the URL contains the word "workday".
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlparse

__all__ = [
    "ARCHIVE_CDX",
    "ASHBY_HOST",
    "GREENHOUSE_HOSTS",
    "PublicBoardIndex",
    "archive_query_url",
    "parse_cdx_tokens",
    "parse_cdx_workday",
    "parse_public_workday_board",
    "prefix_for_day",
    "prefixes_for_day",
    "select_board_tokens",
    "select_workday_boards",
    "valid_board_token",
    "workday_archive_query_url",
    "workday_board_identity",
    "workday_clusters_for_run",
]

ASHBY_HOST = "jobs.ashbyhq.com"
GREENHOUSE_HOSTS = frozenset(
    {"boards.greenhouse.io", "job-boards.greenhouse.io", "boards.eu.greenhouse.io"}
)
ARCHIVE_CDX = "https://web.archive.org/cdx/search/cdx"
WORKDAY_DOMAINS = frozenset({"myworkdayjobs.com", "myworkdaysite.com"})
# Production career hosts already used by configured boards. A run with no
# Workday company still has a public cluster to sample. This is not a
# customer list.
_FALLBACK_WORKDAY_CLUSTERS = ("wd1", "wd5", "wd12")
_WD_HOST = re.compile(
    r"^(?P<tenant>[a-z][a-z0-9-]{0,62})\.(?P<cluster>wd\d{1,3})\.(?P<base>myworkdayjobs\.com|myworkdaysite\.com)$"
)
_WD_LOCALE = re.compile(
    r"^(?:en|fr|de|es|zh|ja|ko|pt|it|nl|no|sv|da|fi|pl|cs|hu|ro|tr|th|vi|id|ms|uk|ar|he)(?:-[a-z]{2})?$",
    re.IGNORECASE,
)
_WD_SITE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_WD_SITE_SKIP = frozenset(
    {
        "assets",
        "asset",
        "static",
        "images",
        "javascripts",
        "css",
        "fonts",
        "favicon.ico",
        "robots.txt",
        "sitemap.xml",
        "login",
        "wday",
        "cxs",
        "search",
        "user",
        "jobs",
        "job",
        "details",
        "www",
    }
)
_WD_FILE = re.compile(r"\.(?:html?|js|css|png|jpe?g|gif|svg|ico|xml|json|txt|map)$", re.IGNORECASE)
_PREFIXES = "0123456789abcdefghijklmnopqrstuvwxyz"
_RESERVED = frozenset(
    {
        "www",
        "embed",
        "api",
        "sitemap",
        "sitemap.xml",
        "robots.txt",
        "ads.txt",
        "app-ads.txt",
        "favicon.ico",
        "jobs",
        "job",
    }
)
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_CSS_UNIT = re.compile(r"^\d+(?:vh|vw|px|em|rem|ch|ex)$", re.IGNORECASE)


@dataclass
class PublicBoardIndex:
    greenhouse: list[str] = field(default_factory=list)
    ashby: list[str] = field(default_factory=list)
    workday: list[str] = field(default_factory=list)
    workday_detail: str = ""
    status: str = "unavailable"
    detail: str = ""
    coverage: str = "partial"
    urls_seen: int = 0
    crawl_id: str = ""
    prefix: str = ""
    index_source: str = "internet_archive_cdx"
    index_seconds: float = 0.0


def valid_board_token(token: str) -> bool:
    """Reject path tricks, CSS fragments, and empty labels before any request."""
    text = unquote(token).strip().strip("/")
    if not text or text.lower() in _RESERVED or _CSS_UNIT.fullmatch(text):
        return False
    return _TOKEN.fullmatch(text) is not None


def prefix_for_day(now: datetime | None = None) -> str:
    """One stable character per UTC date. The next day samples a different slice."""
    moment = now or datetime.now(timezone.utc)
    return _PREFIXES[moment.toordinal() % len(_PREFIXES)]


def prefixes_for_day(now: datetime | None = None, count: int = 4) -> tuple[str, ...]:
    """Spaced characters for one run. The same UTC date returns the same set."""
    width = max(1, min(int(count), len(_PREFIXES)))
    start = (now or datetime.now(timezone.utc)).toordinal() % len(_PREFIXES)
    step = max(1, len(_PREFIXES) // width)
    chosen: list[str] = []
    for offset in range(width):
        char = _PREFIXES[(start + offset * step) % len(_PREFIXES)]
        if char not in chosen:
            chosen.append(char)
    return tuple(chosen)


def archive_query_url(host: str, prefix: str, *, limit: int) -> str:
    """One bounded Archive CDX query. ``prefix`` is a single safe character."""
    if host not in GREENHOUSE_HOSTS and host != ASHBY_HOST:
        raise ValueError(f"board host {host!r} is not allow-listed")
    if len(prefix) != 1 or prefix not in _PREFIXES:
        raise ValueError("board index prefix must be one letter or digit")
    target = quote(f"{host}/{prefix}", safe="")
    return (
        f"{ARCHIVE_CDX}?url={target}&matchType=prefix&output=json"
        f"&fl=original,statuscode&filter=statuscode:200&limit={int(limit)}&collapse=urlkey"
    )


def parse_cdx_tokens(body: str, *, kind: str) -> tuple[list[str], int]:
    """Extract unique board tokens from a CDX body.

    Accepts Internet Archive's JSON array and JSON-lines ``{"url", "status"}``.
    Returns ``(tokens, urls_seen)``. Other hosts are skipped.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    urls = 0
    for row in _cdx_rows(body):
        status = str(row.get("status") or row.get("statuscode") or "")
        if status and not status.startswith("2"):
            continue
        url = str(row.get("url") or row.get("original") or "")
        if not url:
            continue
        urls += 1
        token = _token_from_url(url, kind=kind)
        if token is None:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return tokens, urls


def select_board_tokens(tokens: list[str], *, limit: int, now: datetime | None = None) -> list[str]:
    """Stable daily slice. The same UTC date returns the same tokens."""
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not valid_board_token(token):
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
    if limit <= 0:
        return []
    if len(unique) <= limit:
        return unique
    moment = now or datetime.now(timezone.utc)
    start = moment.toordinal() % len(unique)
    rotated = unique[start:] + unique[:start]
    return rotated[:limit]


def _token_from_url(url: str, *, kind: str) -> str | None:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":")[0]
    if kind == "ashby" and host != ASHBY_HOST:
        return None
    if kind == "greenhouse" and host not in GREENHOUSE_HOSTS:
        return None
    if kind not in {"ashby", "greenhouse"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    token = unquote(parts[0]).strip()
    if not valid_board_token(token):
        return None
    return token


def _cdx_rows(body: str) -> list[dict[str, str]]:
    text = body.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list) or not payload:
            return []
        header = payload[0]
        if isinstance(header, list):
            keys = [str(item).lower() for item in header]
            rows: list[dict[str, str]] = []
            for item in payload[1:]:
                if isinstance(item, list):
                    rows.append({key: str(value) for key, value in zip(keys, item, strict=False)})
            return rows
        return [item for item in payload if isinstance(item, dict)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def workday_clusters_for_run(
    clusters: tuple[str, ...] | list[str],
    now: datetime | None = None,
    *,
    count: int = 2,
) -> tuple[str, ...]:
    """Two production ``wdN`` clusters for this UTC date. The next day shifts."""
    ordered: list[str] = []
    for cluster in clusters:
        text = str(cluster).strip().lower()
        if re.fullmatch(r"wd\d{1,3}", text) and text not in ordered:
            ordered.append(text)
    if not ordered:
        ordered = list(_FALLBACK_WORKDAY_CLUSTERS)
    width = max(1, min(int(count), len(ordered)))
    start = (now or datetime.now(timezone.utc)).toordinal() % len(ordered)
    rotated = ordered[start:] + ordered[:start]
    return tuple(rotated[:width])


def workday_archive_query_url(*, domain: str, cluster: str, prefix: str, limit: int) -> str:
    """One bounded Archive CDX page of public Workday career URLs.

    The resume key is the public CDX pagination cursor. It only seeks the
    archive index to ``{cluster}/{prefix}``. It is not a Workday credential
    and it does not select a facet.
    """
    if domain not in WORKDAY_DOMAINS:
        raise ValueError(f"workday index domain {domain!r} is not allow-listed")
    if not re.fullmatch(r"wd\d{1,3}", cluster or ""):
        raise ValueError("workday cluster must look like wd5")
    if len(prefix) != 1 or prefix not in _PREFIXES:
        raise ValueError("workday index prefix must be one letter or digit")
    labels = domain.split(".")
    surt = ",".join(reversed(labels))
    urlkey = f"{surt},{cluster},{prefix}) 00000000000000"
    resume = base64.b64encode(zlib.compress(urlkey.encode("utf-8"))).decode("ascii")
    return (
        f"{ARCHIVE_CDX}?url={quote(domain)}&matchType=domain&output=json"
        f"&fl=original,statuscode&filter=statuscode:200&limit={int(limit)}"
        f"&collapse=urlkey&resumeKey={quote(resume, safe='')}"
    )


def parse_public_workday_board(url: str) -> tuple[str, str, str] | None:
    """Return ``(host, tenant, site)`` for a public career URL.

    The site id is the first path segment after an optional locale, which is
    the career site even when the captured URL is a job-detail page. Hosts
    that merely contain the word workday, sandbox ``impl-wd`` hosts, and
    asset URLs are rejected.
    """
    if not url or not isinstance(url, str):
        return None
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower().split(":")[0]
    if "impl-" in host or "preview" in host:
        return None
    match = _WD_HOST.fullmatch(host)
    if match is None:
        return None
    tenant = match.group("tenant")
    if tenant in {"www", "wday", "login", "auth"}:
        return None
    segments = [unquote(part).strip() for part in parsed.path.split("/") if part]
    if segments and _WD_LOCALE.fullmatch(segments[0]):
        segments = segments[1:]
    if not segments:
        return None
    site = segments[0]
    if not _WD_SITE.fullmatch(site) or site.lower() in _WD_SITE_SKIP or _WD_FILE.search(site):
        return None
    return host, tenant, site


def workday_board_identity(url: str) -> str | None:
    """Canonical board id: ``host|tenant|site``. Company name is not the id."""
    parsed = parse_public_workday_board(url)
    if parsed is None:
        return None
    host, tenant, site = parsed
    return f"{host}|{tenant}|{site.lower()}"


def workday_career_url(url: str) -> str | None:
    parsed = parse_public_workday_board(url)
    if parsed is None:
        return None
    host, _tenant, site = parsed
    return f"https://{host}/{site}"


def parse_cdx_workday(body: str) -> tuple[list[str], int]:
    """Unique public career URLs from a CDX body. Other hosts are ignored."""
    found: list[str] = []
    seen: set[str] = set()
    urls = 0
    for row in _cdx_rows(body):
        status = str(row.get("status") or row.get("statuscode") or "")
        if status and not status.startswith("2"):
            continue
        raw = str(row.get("url") or row.get("original") or "")
        if not raw:
            continue
        urls += 1
        career = workday_career_url(raw)
        identity = workday_board_identity(raw)
        if career is None or identity is None or identity in seen:
            continue
        seen.add(identity)
        found.append(career)
    return found, urls


def select_workday_boards(urls: list[str], *, limit: int, now: datetime | None = None) -> list[str]:
    """Stable daily slice of career URLs. ``limit <= 0`` selects nothing."""
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        identity = workday_board_identity(url)
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        unique.append(workday_career_url(url) or url)
    if limit <= 0:
        return []
    if len(unique) <= limit:
        return unique
    moment = now or datetime.now(timezone.utc)
    start = moment.toordinal() % len(unique)
    rotated = unique[start:] + unique[:start]
    return rotated[:limit]

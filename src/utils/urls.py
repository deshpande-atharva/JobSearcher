"""URL classification, canonicalisation and job-ID extraction.

The project's central URL rule lives here: aggregators are fine for *discovery*
but are never an acceptable final application link. A :class:`UrlPolicy` is
built from ``config/settings.yaml`` and passed in, so the host lists stay data
rather than code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

__all__ = [
    "UrlKind",
    "UrlPolicy",
    "UrlVerdict",
    "canonicalize_url",
    "classify_url",
    "extract_job_id",
    "host_of",
    "is_generic_careers_page",
    "is_http_url",
    "join_url",
    "registrable_domain",
    "unwrap_redirect",
]


class UrlKind(StrEnum):
    """What a URL points at."""

    ATS = "ATS"
    COMPANY_CAREER = "COMPANY_CAREER"
    AGGREGATOR = "AGGREGATOR"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class UrlPolicy:
    """Host lists and canonicalisation rules, sourced from settings."""

    aggregator_hosts: tuple[str, ...] = ()
    # ats_type -> host suffixes
    ats_hosts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    strip_query_params: tuple[str, ...] = ()

    def all_ats_hosts(self) -> tuple[str, ...]:
        return tuple(host for hosts in self.ats_hosts.values() for host in hosts)

    def ats_type_for(self, host: str) -> str | None:
        for ats_type, hosts in self.ats_hosts.items():
            if _host_matches(host, hosts):
                return ats_type
        return None


@dataclass(slots=True)
class UrlVerdict:
    """Outcome of classifying a single URL."""

    url: str
    kind: UrlKind
    host: str = ""
    ats_type: str | None = None
    reason: str = ""

    @property
    def acceptable_as_final(self) -> bool:
        """True only for URLs allowed to become ``direct_application_url``."""
        return self.kind in (UrlKind.ATS, UrlKind.COMPANY_CAREER)


_HOST_PREFIX_RE = re.compile(r"^(?:www|www\d|m|mobile)\.")


def is_http_url(url: str | None) -> bool:
    """True when ``url`` is a syntactically valid absolute http(s) URL."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def host_of(url: str | None) -> str:
    """Lowercase hostname without a leading ``www.``/``m.`` prefix."""
    if not url:
        return ""
    try:
        netloc = urlparse(url.strip()).netloc.lower()
    except ValueError:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    return _HOST_PREFIX_RE.sub("", netloc)


def registrable_domain(url_or_host: str | None) -> str:
    """Approximate registrable domain, e.g. ``jobs.lever.co`` -> ``lever.co``.

    Uses a small list of known multi-part public suffixes instead of pulling in
    a full Public Suffix List dependency; the classification only needs to be
    right for job-board and corporate hosts.
    """
    host = url_or_host or ""
    if "://" in host:
        host = host_of(host)
    host = host.lower().strip(".")
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    two_part_suffixes = {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "com.au",
        "net.au",
        "org.au",
        "co.jp",
        "co.in",
        "com.br",
        "com.mx",
        "co.za",
        "com.sg",
        "com.cn",
    }
    if ".".join(parts[-2:]) in two_part_suffixes:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _host_matches(host: str, suffixes: tuple[str, ...] | list[str]) -> bool:
    """True when ``host`` equals or is a subdomain of any listed suffix."""
    host = host.lower()
    for suffix in suffixes:
        candidate = suffix.lower().lstrip(".")
        if host == candidate or host.endswith(f".{candidate}"):
            return True
    return False


def canonicalize_url(url: str | None, strip_params: tuple[str, ...] | list[str] = ()) -> str:
    """Normalize a URL for comparison and storage.

    Lowercases scheme/host, drops the fragment, removes tracking parameters and
    sorts the remainder, and trims a trailing slash. Deliberately preserves all
    other query parameters -- ``gh_jid``, ``jobId`` and friends *are* the job
    identity on several ATS platforms.
    """
    if not url:
        return ""
    text = url.strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError:
        return text

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[: -len(":443")]
    elif netloc.endswith(":80") and scheme == "http":
        netloc = netloc[: -len(":80")]

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    strip = {p.lower() for p in strip_params}
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in strip
    ]
    query = urlencode(sorted(query_pairs))

    return urlunparse((scheme, netloc, path, "", query, ""))


def join_url(base: str, relative: str | None) -> str:
    """Resolve ``relative`` against ``base``, returning ``""`` on bad input."""
    if not relative:
        return ""
    try:
        return urljoin(base, relative.strip())
    except ValueError:
        return ""


_REDIRECT_PARAMS = ("url", "redirect", "redirect_url", "target", "destination", "u", "link", "apply_url")


def unwrap_redirect(url: str | None, *, max_depth: int = 3) -> str:
    """Pull a real destination out of a wrapper/redirect URL.

    Aggregators frequently link out via ``?url=https%3A%2F%2F...``. Following
    that parameter is just reading the link the page already gave us -- no
    anti-bot control is involved.
    """
    current = (url or "").strip()
    for _ in range(max_depth):
        if not is_http_url(current):
            return current
        parsed = urlparse(current)
        params = dict(parse_qsl(parsed.query, keep_blank_values=False))
        nested = next(
            (
                params[key]
                for key in _REDIRECT_PARAMS
                if key in params and is_http_url(params[key])
            ),
            None,
        )
        if not nested or nested == current:
            return current
        current = nested
    return current


_GENERIC_CAREERS_PATHS = (
    "",
    "/",
    "/careers",
    "/career",
    "/jobs",
    "/job",
    "/join-us",
    "/work-with-us",
    "/opportunities",
    "/careers/",
    "/jobs/search",
    "/careers/search",
    "/search",
    "/search-jobs",
    "/job-search",
    "/en",
    "/en-us",
    "/us/en",
)


def is_generic_careers_page(url: str | None) -> bool:
    """True when the URL is a careers landing/search page, not a posting.

    Such a URL is only acceptable as a final link when no exact posting URL
    could be found.
    """
    if not is_http_url(url):
        return False
    parsed = urlparse(url.strip())
    path = (parsed.path or "").rstrip("/").lower()
    if parsed.query:
        # A query string usually carries the posting identity.
        return False
    return path in {p.rstrip("/") for p in _GENERIC_CAREERS_PATHS}


def classify_url(url: str | None, policy: UrlPolicy) -> UrlVerdict:
    """Classify a URL as ATS, company careers page, aggregator or invalid."""
    if not is_http_url(url):
        return UrlVerdict(url=url or "", kind=UrlKind.INVALID, reason="not a valid http(s) URL")

    assert url is not None
    host = host_of(url)
    if not host:
        return UrlVerdict(url=url, kind=UrlKind.INVALID, host="", reason="missing host")

    ats_type = policy.ats_type_for(host)
    if ats_type:
        return UrlVerdict(
            url=url,
            kind=UrlKind.ATS,
            host=host,
            ats_type=ats_type,
            reason=f"recognised {ats_type} applicant tracking system host",
        )

    if _host_matches(host, policy.aggregator_hosts):
        return UrlVerdict(
            url=url,
            kind=UrlKind.AGGREGATOR,
            host=host,
            reason="job board or aggregator host, not a direct application URL",
        )

    return UrlVerdict(
        url=url,
        kind=UrlKind.COMPANY_CAREER,
        host=host,
        reason="company-hosted careers URL",
    )


# ---------------------------------------------------------------------------
# Job ID extraction
#
# Patterns below reflect the public URL shapes of each ATS. An ID is only
# returned when the URL genuinely contains one -- never synthesised.
# ---------------------------------------------------------------------------

_JOB_ID_QUERY_KEYS = ("gh_jid", "jobid", "job_id", "jobreqid", "reqid", "requisitionid", "id", "posting_id", "pid")

_PATH_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Greenhouse: /boards/acme/jobs/4012345
    re.compile(r"/jobs/(\d{4,})(?:/|$)"),
    # Lever / Ashby: UUID posting ids
    re.compile(r"/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:/|$)", re.IGNORECASE),
    # Workday requisition: ..._R-12345 or ..._JR1234567
    re.compile(r"_((?:R|JR|REQ)-?\d{3,})(?:/|$)", re.IGNORECASE),
    # SmartRecruiters: /Company/743999123456789-title
    re.compile(r"/(\d{12,})-", re.IGNORECASE),
    # iCIMS: /jobs/12345/software-engineer/job
    re.compile(r"/jobs?/(\d{3,})(?:/|$)"),
    # Generic trailing requisition token: /job/R2412345
    re.compile(r"/((?:R|JR|REQ)-?\d{3,})(?:/|$)", re.IGNORECASE),
    # Generic numeric posting at end of path
    re.compile(r"/(\d{5,})(?:/|$)"),
)


def extract_job_id(url: str | None) -> str | None:
    """Pull the posting identifier out of an ATS URL, if one is present.

    Returns ``None`` when the URL carries no identifier. Nothing is fabricated:
    a missing ID makes deduplication fall back to a content fingerprint.
    """
    if not is_http_url(url):
        return None
    assert url is not None
    parsed = urlparse(url.strip())

    params = {k.lower(): v for k, v in parse_qsl(parsed.query, keep_blank_values=False)}
    for key in _JOB_ID_QUERY_KEYS:
        value = params.get(key)
        if value and re.fullmatch(r"[A-Za-z0-9._\-]{3,64}", value):
            return value

    path = parsed.path or ""
    for pattern in _PATH_ID_PATTERNS:
        match = pattern.search(path)
        if match:
            return match.group(1)

    return None

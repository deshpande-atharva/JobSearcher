"""Discovery source interface and the shared HTTP client.

Scraping conduct is enforced here rather than left to each adapter:

* ``robots.txt`` is fetched, cached and honoured when
  ``scraping.respect_robots_txt`` is on.
* Requests are rate limited per host and globally bounded by
  ``run.max_concurrency``.
* Retries use exponential backoff with jitter and respect ``Retry-After``.
* Nothing here bypasses authentication, solves CAPTCHAs, spoofs fingerprints or
  otherwise evades anti-bot controls. A 401/403/429 that survives backoff is
  reported as a failed source, not worked around.
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import BaseModel, Field

from src.models.config import AppConfig, CompanyConfig
from src.models.job import RawJobPosting
from src.utils.logging import JobLogger, get_logger
from src.utils.urls import host_of

__all__ = [
    "DiscoverySource",
    "FetchResult",
    "HttpClient",
    "SOURCE_STATUSES",
    "SourceContext",
    "SourceError",
    "SourceResult",
    "SourceStatus",
    "classify_source_status",
]

log = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 522, 524})

SourceStatus = Literal["OK", "EMPTY", "ERROR", "BLOCKED", "UNSUPPORTED"]
SOURCE_STATUSES: tuple[SourceStatus, ...] = ("OK", "EMPTY", "ERROR", "BLOCKED", "UNSUPPORTED")

_BLOCKED_MARKERS = (
    "robots.txt",
    "disallowed by robots",
    "captcha",
    "access denied",
    "http 401",
    "http 403",
    "blocked",
    "verify you are human",
)
_UNSUPPORTED_MARKERS = (
    "unsupported_configuration",
    "unsupported configuration",
    "not a public myworkdayjobs",
)


def classify_source_status(error: str | None, *, http_status: int | None = None) -> SourceStatus:
    """Map a failure into SOURCE_ERROR / BLOCKED / UNSUPPORTED. Never EMPTY."""
    if http_status in (401, 403):
        return "BLOCKED"
    text = (error or "").lower()
    if any(marker in text for marker in _UNSUPPORTED_MARKERS):
        return "UNSUPPORTED"
    if any(marker in text for marker in _BLOCKED_MARKERS):
        return "BLOCKED"
    return "ERROR"


class SourceError(RuntimeError):
    """A discovery source could not complete. Always caught per-source."""

    def __init__(
        self,
        message: str,
        *,
        status: SourceStatus | None = None,
        http_status: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status: SourceStatus = status or classify_source_status(message, http_status=http_status)
        self.http_status = http_status
        self.diagnostics = diagnostics or {}


class SourceResult(BaseModel):
    """Outcome of one source attempt. Zero jobs is not the same as failure."""

    source_name: str
    company: str
    jobs: list[RawJobPosting] = Field(default_factory=list)
    success: bool = True
    error: str | None = None
    duration_seconds: float | None = None
    discovered_count: int = 0
    status: SourceStatus = "OK"
    http_status: int | None = None
    fallback_used: bool = False
    fallback_status: SourceStatus | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        source_name: str,
        company: str,
        jobs: list[RawJobPosting],
        duration_seconds: float | None = None,
        *,
        http_status: int | None = None,
        fallback_used: bool = False,
        fallback_status: SourceStatus | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> SourceResult:
        return cls(
            source_name=source_name,
            company=company,
            jobs=jobs,
            success=True,
            discovered_count=len(jobs),
            duration_seconds=duration_seconds,
            status="OK" if jobs else "EMPTY",
            http_status=http_status,
            fallback_used=fallback_used,
            fallback_status=fallback_status,
            diagnostics=diagnostics or {},
        )

    @classmethod
    def fail(
        cls,
        source_name: str,
        company: str,
        error: str,
        duration_seconds: float | None = None,
        *,
        status: SourceStatus | None = None,
        http_status: int | None = None,
        fallback_used: bool = False,
        fallback_status: SourceStatus | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> SourceResult:
        classified = status or classify_source_status(error, http_status=http_status)
        return cls(
            source_name=source_name,
            company=company,
            success=False,
            error=error,
            discovered_count=0,
            duration_seconds=duration_seconds,
            status=classified,
            http_status=http_status,
            fallback_used=fallback_used,
            fallback_status=fallback_status,
            diagnostics=diagnostics or {},
        )


@dataclass(slots=True)
class FetchResult:
    """Outcome of one HTTP request."""

    url: str
    status: int | None = None
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    blocked_by_robots: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 300

    @property
    def body_preview(self) -> str:
        """Short response excerpt for diagnostics. Never includes cookies or tokens."""
        text = (self.text or "").replace("\n", " ").strip()
        return text[:200]

    def json(self) -> Any | None:
        """Parse the body as JSON, returning ``None`` when it is not JSON."""
        if not self.text:
            return None
        import json

        try:
            return json.loads(self.text)
        except (ValueError, TypeError):
            return None


class _RobotsCache:
    """Per-host ``robots.txt`` cache.

    A host whose ``robots.txt`` cannot be fetched is treated as allowing access,
    which matches the robots standard: absent rules are not a prohibition.
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, host: str) -> asyncio.Lock:
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    async def allowed(self, url: str) -> bool:
        host = host_of(url)
        if not host:
            return False
        async with self._lock(host):
            if host not in self._parsers:
                self._parsers[host] = await self._load(url)
        parser = self._parsers[host]
        if parser is None:
            return True
        try:
            return parser.can_fetch(self._user_agent, url)
        except Exception:  # a malformed robots file must not block the run
            return True

    async def _load(self, url: str) -> RobotFileParser | None:
        robots_url = urljoin(url, "/robots.txt")
        try:
            response = await self._client.get(robots_url, timeout=10.0)
        except Exception as exc:
            log.debug("robots.txt unavailable", url=robots_url, error=str(exc))
            return None
        if response.status_code >= 400:
            return None
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.parse(response.text.splitlines())
        except Exception:
            return None
        return parser


class _HostThrottle:
    """Enforces a minimum delay between requests to the same host."""

    def __init__(self, min_delay: float) -> None:
        self._min_delay = min_delay
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, host: str) -> None:
        if self._min_delay <= 0 or not host:
            return
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        async with self._locks[host]:
            previous = self._last.get(host)
            now = time.monotonic()
            if previous is not None:
                remaining = self._min_delay - (now - previous)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last[host] = time.monotonic()


class HttpClient:
    """Polite async HTTP client shared by every source."""

    def __init__(self, config: AppConfig) -> None:
        run = config.settings.run
        scraping = config.settings.scraping
        self._config = config
        self._max_bytes = scraping.max_html_bytes
        self._respect_robots = scraping.respect_robots_txt
        self._retries = run.retry_attempts
        self._backoff = run.retry_backoff_seconds
        self._semaphore = asyncio.Semaphore(run.max_concurrency)
        self._throttle = _HostThrottle(scraping.min_delay_seconds)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(run.request_timeout_seconds),
            follow_redirects=True,
            headers={
                "User-Agent": run.user_agent,
                "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            limits=httpx.Limits(
                max_connections=run.max_concurrency * 2,
                max_keepalive_connections=run.max_concurrency,
            ),
        )
        self._robots = _RobotsCache(self._client, run.user_agent)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # --- core request -------------------------------------------------------

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        expect_json: bool = False,
        allow_robots_check: bool = True,
    ) -> FetchResult:
        """Perform a request with throttling, robots checks and backoff.

        Never raises for HTTP or transport problems -- the error is returned on
        the :class:`FetchResult` so one bad source cannot abort a run.
        """
        if allow_robots_check and self._respect_robots and not await self._robots.allowed(url):
            log.info("skipping URL disallowed by robots.txt", url=url)
            return FetchResult(url=url, error="disallowed by robots.txt", blocked_by_robots=True)

        request_headers = dict(headers or {})
        if expect_json:
            request_headers.setdefault("Accept", "application/json")

        host = host_of(url)
        last_error: str | None = None
        last_status: int | None = None

        for attempt in range(self._retries + 1):
            async with self._semaphore:
                await self._throttle.wait(host)
                try:
                    response = await self._client.request(
                        method,
                        url,
                        headers=request_headers,
                        params=params,
                        json=json_body,
                    )
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    last_status = None
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    last_status = None
                else:
                    last_status = response.status_code
                    if response.status_code in _RETRYABLE_STATUS and attempt < self._retries:
                        delay = self._retry_delay(attempt, response.headers.get("Retry-After"))
                        log.debug(
                            "retrying request",
                            url=url,
                            status=response.status_code,
                            attempt=attempt + 1,
                            delay=round(delay, 2),
                        )
                        await asyncio.sleep(delay)
                        continue

                    if response.status_code in (401, 403):
                        # Authentication or an anti-bot wall. We do not attempt
                        # to work around either.
                        return FetchResult(
                            url=str(response.url),
                            status=response.status_code,
                            headers=dict(response.headers),
                            error=(
                                f"access denied (HTTP {response.status_code}); "
                                "source requires authentication or blocks automated access"
                            ),
                        )

                    body = response.text or ""
                    # Truncating JSON makes it unparseable. HTML pages still
                    # honour max_html_bytes; public ATS APIs may exceed that.
                    if not expect_json and len(body) > self._max_bytes:
                        body = body[: self._max_bytes]
                    return FetchResult(
                        url=str(response.url),
                        status=response.status_code,
                        text=body,
                        headers=dict(response.headers),
                        error=None if response.is_success else f"HTTP {response.status_code}",
                    )

            if attempt < self._retries:
                await asyncio.sleep(self._retry_delay(attempt, None))

        return FetchResult(url=url, status=last_status, error=last_error or "request failed")

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        """Exponential backoff with jitter, capped, honouring ``Retry-After``."""
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        base = self._backoff * (2**attempt)
        return min(base + random.uniform(0, 0.4 * base), 30.0)

    # --- convenience wrappers ----------------------------------------------

    async def get_text(self, url: str, **kwargs: Any) -> FetchResult:
        return await self.request("GET", url, **kwargs)

    async def get_json(self, url: str, **kwargs: Any) -> Any | None:
        result = await self.request("GET", url, expect_json=True, **kwargs)
        if result.blocked_by_robots:
            raise SourceError(
                f"GET {url} failed: disallowed by robots.txt",
                status="BLOCKED",
                http_status=result.status,
            )
        if not result.ok:
            raise SourceError(
                f"GET {url} failed: {result.error or result.status}",
                http_status=result.status,
            )
        payload = result.json()
        if payload is None:
            raise SourceError(f"GET {url} did not return JSON", http_status=result.status)
        return payload

    async def post_json(self, url: str, body: Any, **kwargs: Any) -> Any | None:
        result = await self.request("POST", url, json_body=body, expect_json=True, **kwargs)
        if result.blocked_by_robots:
            raise SourceError(
                f"POST {url} failed: disallowed by robots.txt",
                status="BLOCKED",
                http_status=result.status,
            )
        if not result.ok:
            raise SourceError(
                f"POST {url} failed: {result.error or result.status}",
                http_status=result.status,
            )
        payload = result.json()
        if payload is None:
            raise SourceError(f"POST {url} did not return JSON", http_status=result.status)
        return payload

    async def head(self, url: str) -> FetchResult:
        """HEAD request, falling back to a ranged GET.

        Plenty of ATS hosts answer HEAD with 405 while serving GET perfectly
        well, so a HEAD failure alone must not condemn a URL.
        """
        result = await self.request("HEAD", url)
        if result.ok or result.blocked_by_robots:
            return result
        if result.status in (403, 404) or result.status is None:
            return await self.request("GET", url, headers={"Range": "bytes=0-2048"})
        if result.status in (405, 501):
            return await self.request("GET", url, headers={"Range": "bytes=0-2048"})
        return result


# ---------------------------------------------------------------------------
# Source interface
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceContext:
    """Dependencies handed to every discovery source.

    Explicit injection keeps sources trivially testable: a test supplies a
    context with a fake HTTP client or a fixture directory and nothing else
    changes.
    """

    config: AppConfig
    http: HttpClient
    logger: JobLogger

    @property
    def fixture_mode(self) -> bool:
        return self.config.fixture_mode


class DiscoverySource(ABC):
    """A place jobs can be discovered from.

    ``scope`` decides how the discovery agent drives the source:

    * ``"global"`` -- called once per run (e.g. Jobright).
    * ``"company"`` -- called once per configured company (e.g. Greenhouse).
    """

    name: ClassVar[str] = "base"
    scope: ClassVar[Literal["global", "company"]] = "global"
    #: Required ``ats_type`` for company-scoped sources; ``None`` means the
    #: source decides for itself via :meth:`supports`.
    ats_type: ClassVar[str | None] = None

    def __init__(self, ctx: SourceContext) -> None:
        self.ctx = ctx
        self.config = ctx.config
        self.http = ctx.http
        self.log = ctx.logger.bind(source=self.name)

    @property
    def enabled(self) -> bool:
        return self.config.settings.discovery.sources.is_enabled(self.name)

    def supports(self, company: CompanyConfig | None) -> bool:
        """Whether this source can discover jobs for ``company``."""
        if self.scope == "global":
            return company is None
        if company is None:
            return False
        if self.ats_type is None:
            return False
        return company.ats_type == self.ats_type and bool(company.ats_identifier)

    @abstractmethod
    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        """Return everything this source found. May raise :class:`SourceError`."""

    async def discover_result(self, company: CompanyConfig | None = None) -> SourceResult:
        """Run :meth:`discover` and always return a :class:`SourceResult`."""
        label = company.name if company else "*"
        started = time.perf_counter()
        try:
            jobs = await self.discover(company)
        except SourceError as exc:
            return SourceResult.fail(
                self.name,
                label,
                str(exc),
                time.perf_counter() - started,
                status=exc.status,
                http_status=exc.http_status,
                diagnostics=exc.diagnostics,
            )
        except Exception as exc:
            return SourceResult.fail(
                self.name, label, str(exc), time.perf_counter() - started
            )
        return SourceResult.ok(self.name, label, jobs, time.perf_counter() - started)

    def _posting(self, **kwargs: Any) -> RawJobPosting:
        """Build a :class:`RawJobPosting` tagged with this source's name."""
        kwargs.setdefault("source", self.name)
        return RawJobPosting(**kwargs)

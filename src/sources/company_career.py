"""Generic company career-page adapter.

Used when a company has no structured ATS identifier, or as a fallback when
the ATS source failed or returned nothing. HTTP first, Playwright only when
the page is clearly JS-rendered. Job-detail pages are fetched when the listing
page has links but no structured postings.

A generic careers homepage is never treated as the final application URL when
a more specific posting URL exists.
"""

from __future__ import annotations

import re
import time
from typing import ClassVar
from urllib.parse import urlparse

from src.models.config import CompanyConfig
from src.models.job import RawJobPosting
from src.sources.base import DiscoverySource, SourceError
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import (
    extract_job_links,
    extract_json_ld_job_postings,
    json_ld_to_raw_posting,
)
from src.utils.urls import is_generic_careers_page, is_http_url

__all__ = ["CompanyCareerSource"]

# Navigation and login URLs are not job postings. Fetching them is what kept
# pages such as AMD's careers home busy without returning public jobs.
_UNUSABLE_LINK = re.compile(
    r"(?:/login(?:/|$|\?)|/signin(?:/|$|\?)|/sign-in(?:/|$|\?)|"
    r"/privacy(?:/|$|\?|\.html)|/userhome(?:/|$|\?)|/benefits(?:/|$|\.html)|"
    r"/student-programs(?:/|$|\.html)|loginonly=1)",
    re.IGNORECASE,
)


class CompanyCareerSource(DiscoverySource):
    name: ClassVar[str] = "company_career"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = None

    def supports(self, company: CompanyConfig | None) -> bool:
        if company is None:
            return False
        return bool(company.careers_url or company.discovery_urls)

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        if company is None:
            return []
        urls = [u for u in (company.careers_url, *company.discovery_urls) if u]
        if not urls:
            return []

        if self.ctx.fixture_mode:
            return self._from_fixture(company, urls[0])

        started = time.perf_counter()
        budget = self.config.settings.discovery.career_stage_budget_seconds
        self._stage_deadline = started + budget
        self.last_diagnostics = {
            "page_outcome": "OK",
            "http_seconds": 0.0,
            "parse_seconds": 0.0,
            "playwright_seconds": 0.0,
            "detail_seconds": 0.0,
            "detail_requests": 0,
        }

        collected: list[RawJobPosting] = []
        errors: list[str] = []
        first_failure: SourceError | None = None
        for url in urls:
            if self._over_budget():
                self.last_diagnostics["page_outcome"] = "TIMEOUT"
                break
            try:
                collected.extend(await self._fetch_page(url, company))
            except SourceError as exc:
                errors.append(str(exc))
                first_failure = first_failure or exc
                self.log.info("career page failed", company=company.name, url=url, error=str(exc))

        if not collected and errors and first_failure is not None:
            raise SourceError(
                f"{company.name} career pages failed: {errors[0]}",
                status=first_failure.status,
                http_status=first_failure.http_status,
                diagnostics=first_failure.diagnostics or dict(self.last_diagnostics),
            )
        return _dedupe(_drop_generic_homepages(collected))

    def _from_fixture(self, company: CompanyConfig, page_url: str) -> list[RawJobPosting]:
        store = FixtureStore(self.config.fixture_dir)
        html = store.load_text(self.name, f"{slugify(company.name)}.html")
        if html:
            return self._parse_html(html, page_url, company)
        payload = store.load_json(self.name, f"{slugify(company.name)}.json")
        if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
            return [
                p
                for p in (
                    json_ld_to_raw_posting(
                        job, source=self.name, page_url=page_url, fallback_company=company.name
                    )
                    for job in payload["jobs"]
                    if isinstance(job, dict)
                )
                if p
            ]
        return []

    def _over_budget(self) -> bool:
        deadline = getattr(self, "_stage_deadline", None)
        return deadline is not None and time.perf_counter() >= deadline

    def _add_seconds(self, key: str, started: float) -> None:
        diag = getattr(self, "last_diagnostics", None)
        if not isinstance(diag, dict):
            return
        diag[key] = round(float(diag.get(key) or 0) + (time.perf_counter() - started), 3)

    async def _fetch_page(self, url: str, company: CompanyConfig) -> list[RawJobPosting]:
        http_started = time.perf_counter()
        result = await self.http.get_text(url)
        self._add_seconds("http_seconds", http_started)
        if result.blocked_by_robots:
            raise SourceError(
                f"disallowed by robots.txt: {url}",
                diagnostics={"page_outcome": "ROBOTS_DISALLOWED"},
            )
        if not result.ok:
            outcome = _http_outcome(result.status)
            raise SourceError(
                f"GET {url} failed: {result.error or result.status}",
                http_status=result.status,
                diagnostics={"page_outcome": outcome},
            )

        html = result.text or ""
        final_url = result.url or url
        if not html.strip():
            self.last_diagnostics["page_outcome"] = "EMPTY_PAGE"
            return []

        parse_started = time.perf_counter()
        postings = self._parse_html(html, final_url, company)
        self._add_seconds("parse_seconds", parse_started)
        structured = any(p.provenance.get("parser") == "json-ld" for p in postings)
        if structured:
            self.last_diagnostics["page_outcome"] = "OK"
            return postings
        if self._over_budget():
            self.last_diagnostics["page_outcome"] = "TIMEOUT"
            return []

        if not structured and not self._over_budget():
            render_started = time.perf_counter()
            rendered = await self._maybe_render(final_url, html)
            self._add_seconds("playwright_seconds", render_started)
            if rendered:
                html = rendered
                parse_started = time.perf_counter()
                postings = self._parse_html(html, final_url, company)
                self._add_seconds("parse_seconds", parse_started)
                structured = any(p.provenance.get("parser") == "json-ld" for p in postings)

        if structured:
            self.last_diagnostics["page_outcome"] = "OK"
            return postings

        links = [
            link
            for link in extract_job_links(html, final_url, limit=80)
            if _usable_job_link(link[0])
        ]
        if _looks_js_shell(html) and not links:
            self.last_diagnostics["page_outcome"] = "JS_SHELL"
            return []
        if not links:
            self.last_diagnostics["page_outcome"] = "UNSUPPORTED_STRUCTURE"
            return []
        if self._over_budget():
            self.last_diagnostics["page_outcome"] = "TIMEOUT"
            return []
        enriched = await self._enrich_from_detail_pages(links, company, [])
        if self.last_diagnostics.get("page_outcome") == "OK" and enriched:
            self.last_diagnostics["page_outcome"] = "OK"
        return enriched

    async def _maybe_render(self, url: str, html: str) -> str | None:
        if self.ctx.fixture_mode:
            return None
        if not self.config.settings.scraping.playwright.enabled:
            return None
        # Workday listings come from CXS. Rendering the careers host would be
        # browser discovery, which stays disabled.
        host = urlparse(url).netloc.lower()
        if "myworkdayjobs.com" in host or "myworkdaysite.com" in host:
            return None
        if _looks_js_shell(html):
            renderer = self.http.renderer
            if not renderer.enabled:
                return None
            limit = self.config.settings.scraping.playwright.max_renders_per_run
            if renderer.render_count >= limit:
                return None
            self.log.info("rendering career page with Playwright", url=url)
            return await renderer.render(url)
        return None

    async def _enrich_from_detail_pages(
        self,
        links: list[tuple[str, str]],
        company: CompanyConfig,
        existing: list[RawJobPosting],
    ) -> list[RawJobPosting]:
        limit = self.config.settings.discovery.career_detail_fetch_limit
        targets = [
            (href, text)
            for href, text in links
            if is_http_url(href) and not is_generic_careers_page(href)
        ][:limit]
        if not targets:
            return existing or [
                self._posting(
                    company_name=company.name,
                    title=text or None,
                    apply_url=href,
                    provenance={"parser": "career-links"},
                )
                for href, text in links
                if not is_generic_careers_page(href)
            ]

        collected = list(existing)
        consecutive_failures = 0
        for href, text in targets:
            if self._over_budget():
                self.last_diagnostics["page_outcome"] = "TIMEOUT"
                break
            detail_started = time.perf_counter()
            self.last_diagnostics["detail_requests"] = int(
                self.last_diagnostics.get("detail_requests") or 0
            ) + 1
            try:
                page = await self.http.get_text(href)
            except Exception as exc:
                self._add_seconds("detail_seconds", detail_started)
                self.log.debug("job detail fetch failed", url=href, error=str(exc))
                consecutive_failures += 1
                collected.append(
                    self._posting(
                        company_name=company.name,
                        title=text or None,
                        apply_url=href,
                        provenance={"parser": "career-link"},
                    )
                )
                if consecutive_failures >= 2:
                    break
                continue
            self._add_seconds("detail_seconds", detail_started)
            if not page.ok:
                consecutive_failures += 1
                collected.append(
                    self._posting(
                        company_name=company.name,
                        title=text or None,
                        apply_url=href,
                        provenance={"parser": "career-link"},
                    )
                )
                if page.blocked_by_robots or page.status in (401, 403) or consecutive_failures >= 2:
                    break
                continue
            consecutive_failures = 0
            details = self._parse_html(page.text, href, company)
            if details:
                collected.extend(details)
            else:
                collected.append(
                    self._posting(
                        company_name=company.name,
                        title=text or None,
                        apply_url=href,
                        provenance={"parser": "career-link"},
                    )
                )
        return collected

    def _parse_html(self, html: str, page_url: str, company: CompanyConfig) -> list[RawJobPosting]:
        postings: list[RawJobPosting] = []
        for block in extract_json_ld_job_postings(html):
            converted = json_ld_to_raw_posting(
                block, source=self.name, page_url=page_url, fallback_company=company.name
            )
            if converted:
                postings.append(converted)
        if postings:
            return postings
        for href, text in extract_job_links(html, page_url, limit=80):
            if is_generic_careers_page(href):
                continue
            postings.append(
                self._posting(
                    company_name=company.name,
                    title=text or None,
                    apply_url=href,
                    provenance={"parser": "career-links", "page_url": page_url},
                )
            )
        return postings


def _usable_job_link(href: str) -> bool:
    """False for login, privacy, and careers-marketing URLs."""
    return _UNUSABLE_LINK.search(href or "") is None


def _http_outcome(status: int | None) -> str:
    if status == 403:
        return "HTTP_FORBIDDEN"
    if status == 404:
        return "HTTP_NOT_FOUND"
    if status in (401, 407):
        return "HTTP_FORBIDDEN"
    return "HTTP_ERROR"


def _looks_js_shell(html: str) -> bool:
    """True when the document is mostly an empty app shell."""
    if not html or len(html) < 800:
        return True
    lowered = html.lower()
    if "captcha" in lowered or "cf-browser-verification" in lowered:
        return False
    text_len = len(html)
    if text_len < 4000 and ("__next" in lowered or "id=\"root\"" in lowered or "id=\"app\"" in lowered):
        return True
    return "application/ld+json" not in lowered and extract_job_links(html, "https://example.com") == []


def _drop_generic_homepages(postings: list[RawJobPosting]) -> list[RawJobPosting]:
    specific = [p for p in postings if p.apply_url and not is_generic_careers_page(p.apply_url)]
    return specific if specific else postings


def _dedupe(postings: list[RawJobPosting]) -> list[RawJobPosting]:
    seen: set[str] = set()
    unique: list[RawJobPosting] = []
    for posting in postings:
        key = f"{posting.apply_url}|{posting.title}|{posting.job_id}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(posting)
    return unique

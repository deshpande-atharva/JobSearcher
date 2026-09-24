"""Generic company career-page adapter.

Used when a company has no structured ATS identifier, or as a fallback when
the ATS source failed or returned nothing. HTTP first, Playwright only when
the page is clearly JS-rendered. Job-detail pages are fetched when the listing
page has links but no structured postings.

A generic careers homepage is never treated as the final application URL when
a more specific posting URL exists.
"""

from __future__ import annotations

from typing import ClassVar

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

        collected: list[RawJobPosting] = []
        errors: list[str] = []
        for url in urls:
            try:
                collected.extend(await self._fetch_page(url, company))
            except SourceError as exc:
                errors.append(str(exc))
                self.log.info("career page failed", company=company.name, url=url, error=str(exc))

        if not collected and errors:
            raise SourceError(f"{company.name} career pages failed: {errors[0]}")
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

    async def _fetch_page(self, url: str, company: CompanyConfig) -> list[RawJobPosting]:
        result = await self.http.get_text(url)
        if result.blocked_by_robots:
            raise SourceError(f"disallowed by robots.txt: {url}")
        if not result.ok:
            raise SourceError(f"GET {url} failed: {result.error or result.status}")

        html = result.text
        final_url = result.url or url
        postings = self._parse_html(html, final_url, company)
        structured = any(p.provenance.get("parser") == "json-ld" for p in postings)

        if not structured:
            rendered = await self._maybe_render(final_url, html)
            if rendered:
                html = rendered
                postings = self._parse_html(html, final_url, company)
                structured = any(p.provenance.get("parser") == "json-ld" for p in postings)

        if not structured:
            links = extract_job_links(html, final_url, limit=80)
            postings = await self._enrich_from_detail_pages(links, company, postings)
        return postings

    async def _maybe_render(self, url: str, html: str) -> str | None:
        if self.ctx.fixture_mode:
            return None
        if not self.config.settings.scraping.playwright.enabled:
            return None
        if _looks_js_shell(html):
            from src.sources.playwright_renderer import PlaywrightRenderer

            renderer = PlaywrightRenderer(self.config)
            if not renderer.enabled:
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
        for href, text in targets:
            try:
                page = await self.http.get_text(href)
            except Exception as exc:
                self.log.debug("job detail fetch failed", url=href, error=str(exc))
                collected.append(
                    self._posting(
                        company_name=company.name,
                        title=text or None,
                        apply_url=href,
                        provenance={"parser": "career-link"},
                    )
                )
                continue
            if not page.ok:
                collected.append(
                    self._posting(
                        company_name=company.name,
                        title=text or None,
                        apply_url=href,
                        provenance={"parser": "career-link"},
                    )
                )
                continue
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

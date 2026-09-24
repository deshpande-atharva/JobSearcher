"""Jobright discovery adapter.

Primary listing: ``https://jobright.ai/entry-level-jobs``.

Jobright does not publish a documented public API, so this adapter only reads
the public listing page (static HTML first, optional Playwright render second).
Jobright is a *discovery* source: aggregator URLs it produces are never treated
as the final application link.

The scraper is isolated here so a site-structure change cannot leak into the
rest of the pipeline.
"""

from __future__ import annotations

from typing import Any, ClassVar

from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.sources.base import DiscoverySource, SourceError
from src.sources.fixtures import FixtureStore
from src.sources.parsing import (
    extract_embedded_json_blobs,
    extract_job_links,
    extract_json_ld_job_postings,
    extract_next_data,
    iter_dicts,
    json_ld_to_raw_posting,
    pick,
)
from src.utils.dates import parse_datetime
from src.utils.normalization import clean_text
from src.utils.urls import is_http_url

__all__ = ["JobrightSource"]


class JobrightSource(DiscoverySource):
    name: ClassVar[str] = "jobright"
    scope: ClassVar[str] = "global"

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        if company is not None:
            return []

        settings = self.config.settings.discovery.sources.jobright
        if self.ctx.fixture_mode:
            return self._from_fixture()

        html = await self._fetch_listing(settings.entry_url)
        if _looks_blocked(html):
            raise SourceError("Jobright returned a blocked or CAPTCHA page")
        postings = self._parse_listing(html, settings.entry_url)

        if not postings and settings.allow_browser_render:
            rendered = await self._render(settings.entry_url)
            if rendered:
                if _looks_blocked(rendered):
                    raise SourceError(
                        "Jobright Playwright render was blocked or showed a CAPTCHA",
                        status="BLOCKED",
                    )
                postings = self._parse_listing(rendered, settings.entry_url)
            elif not postings:
                self.log.info(
                    "jobright listing has no machine-readable jobs; "
                    "Playwright render unavailable or returned nothing"
                )

        self.log.info("jobright discovery finished", count=len(postings))
        return postings

    def _from_fixture(self) -> list[RawJobPosting]:
        store = FixtureStore(self.config.fixture_dir)
        payload = store.load_first(self.name, ["entry-level.json", "jobs.json"])
        if payload is None:
            self.log.debug("no jobright fixture")
            return []
        entries = payload if isinstance(payload, list) else pick(payload, "jobs", "items", default=[])
        if not isinstance(entries, list):
            raise SourceError("unexpected Jobright fixture shape")
        return [p for p in (self._entry_to_posting(e, page_url="fixture://jobright") for e in entries) if p]

    async def _fetch_listing(self, url: str) -> str:
        result = await self.http.get_text(url)
        if result.blocked_by_robots:
            raise SourceError(
                "Jobright listing is disallowed by robots.txt",
                status="BLOCKED",
                http_status=result.status,
            )
        if not result.ok:
            raise SourceError(
                f"Jobright listing fetch failed: {result.error or result.status}",
                http_status=result.status,
            )
        return result.text

    async def _render(self, url: str) -> str | None:
        from src.sources.playwright_renderer import PlaywrightRenderer

        renderer = PlaywrightRenderer(self.config)
        if not renderer.enabled:
            return None
        self.log.info("rendering Jobright listing with Playwright")
        return await renderer.render(url)

    def _parse_listing(self, html: str, page_url: str) -> list[RawJobPosting]:
        postings: list[RawJobPosting] = []
        seen: set[str] = set()

        def add(posting: RawJobPosting | None) -> None:
            if posting is None:
                return
            key = (posting.apply_url or "") + "|" + (posting.title or "") + "|" + (posting.company_name or "")
            if key in seen:
                return
            seen.add(key)
            postings.append(posting)

        for block in extract_json_ld_job_postings(html):
            add(json_ld_to_raw_posting(block, source=self.name, page_url=page_url))

        next_data = extract_next_data(html)
        if next_data is not None:
            for node in iter_dicts(next_data):
                if _looks_like_job(node):
                    add(self._entry_to_posting(node, page_url=page_url))

        for blob in extract_embedded_json_blobs(html):
            for node in iter_dicts(blob):
                if _looks_like_job(node):
                    add(self._entry_to_posting(node, page_url=page_url))

        if not postings:
            for url, text in extract_job_links(html, page_url):
                add(
                    self._posting(
                        title=text or None,
                        apply_url=url,
                        provenance={"parser": "link-harvest", "page_url": page_url},
                    )
                )
        return postings

    def _entry_to_posting(self, entry: Any, *, page_url: str) -> RawJobPosting | None:
        if not isinstance(entry, dict):
            return None
        title = clean_text(str(pick(entry, "title", "jobTitle", "name", "position") or "")) or None
        company = clean_text(str(pick(entry, "company", "companyName", "employer") or "")) or None
        if isinstance(pick(entry, "company"), dict):
            company = clean_text(str(pick(pick(entry, "company"), "name") or "")) or company

        apply_url = _first_url(entry, "applyUrl", "applicationUrl", "url", "link", "jobUrl", "canonicalUrl")
        alternates = []
        for key in ("sourceUrl", "originalUrl", "redirectUrl", "externalUrl"):
            extra = _first_url(entry, key)
            if extra and extra != apply_url:
                alternates.append(extra)

        if not title and not apply_url:
            return None

        location = pick(entry, "location", "jobLocation", "city")
        if isinstance(location, dict):
            location = pick(location, "name", "city", "label")
        location_raw = clean_text(str(location or "")) or None

        posted_raw = pick(entry, "postedAt", "datePosted", "createdAt", "publishedAt", "postedDate")
        updated_raw = pick(entry, "updatedAt", "dateModified")
        posted_at = parse_datetime(posted_raw)
        updated_at = parse_datetime(updated_raw)
        if posted_at is not None:
            date_source = DateSource.POSTED_DATE
        elif updated_at is not None:
            date_source = DateSource.UPDATED_DATE
        else:
            recency = pick(entry, "postedAgo", "recency", "timeAgo")
            posted_at = parse_datetime(recency)
            date_source = DateSource.POSTED_DATE if posted_at is not None else DateSource.UNKNOWN

        job_id = pick(entry, "jobId", "id", "requisitionId")
        description = clean_text(str(pick(entry, "description", "jobDescription", "snippet") or "")) or None

        return self._posting(
            company_name=company,
            title=title,
            location_raw=location_raw,
            description=description,
            employment_type_raw=str(pick(entry, "employmentType", "jobType") or "") or None,
            remote_type_raw=str(pick(entry, "workplaceType", "remoteType") or "") or None,
            job_id=str(job_id) if job_id else None,
            apply_url=apply_url,
            alternate_urls=alternates,
            posted_at_raw=str(posted_raw) if posted_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance={"parser": "jobright", "page_url": page_url},
        )


def _looks_like_job(node: dict[str, Any]) -> bool:
    keys = {str(k).lower() for k in node}
    title_keys = keys & {"title", "jobtitle", "name", "position"}
    url_keys = keys & {"applyurl", "applicationurl", "url", "joburl", "canonicalurl", "link"}
    id_keys = keys & {"jobid", "requisitionid", "id"}
    if not title_keys:
        return False
    return bool(url_keys or id_keys)


def _first_url(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = pick(entry, key)
        if isinstance(value, str) and is_http_url(value):
            return value.strip()
    return None


def _looks_blocked(html: str) -> bool:
    lowered = (html or "").lower()
    return any(
        token in lowered
        for token in (
            "captcha",
            "cf-browser-verification",
            "access denied",
            "unusual traffic",
            "verify you are human",
        )
    )

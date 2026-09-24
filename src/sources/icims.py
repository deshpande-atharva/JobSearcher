"""iCIMS career-site adapter.

iCIMS does not publish a documented unauthenticated jobs API. Public career
pages typically live on ``*.icims.com`` and expose ``schema.org/JobPosting``
JSON-LD plus ordinary job links. This adapter reads those public pages only.
"""

from __future__ import annotations

from typing import Any, ClassVar

from src.models.config import CompanyConfig
from src.models.job import RawJobPosting
from src.sources.base import DiscoverySource, SourceError
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import (
    extract_job_links,
    extract_json_ld_job_postings,
    json_ld_to_raw_posting,
    pick,
)

__all__ = ["IcimsSource"]


class IcimsSource(DiscoverySource):
    name: ClassVar[str] = "icims"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "icims"

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        if company is None or not company.ats_identifier:
            return []

        if self.ctx.fixture_mode:
            return self._from_fixture(company)

        url = company.ats_identifier.strip()
        result = await self.http.get_text(url)
        if result.blocked_by_robots:
            raise SourceError(f"iCIMS page for {company.name} is disallowed by robots.txt")
        if not result.ok:
            raise SourceError(f"iCIMS fetch failed: {result.error or result.status}")
        return self._parse_html(result.text, url, company)

    def _from_fixture(self, company: CompanyConfig) -> list[RawJobPosting]:
        store = FixtureStore(self.config.fixture_dir)
        payload = store.load_first(self.name, [f"{slugify(company.name)}.json"])
        if isinstance(payload, dict):
            html = pick(payload, "html")
            if isinstance(html, str):
                return self._parse_html(html, company.ats_identifier or "", company)
            jobs = pick(payload, "jobs", default=[])
            if isinstance(jobs, list):
                parsed: list[RawJobPosting] = []
                for job in jobs:
                    if not isinstance(job, dict):
                        continue
                    converted = json_ld_to_raw_posting(
                        job,
                        source=self.name,
                        page_url=company.ats_identifier or "",
                        fallback_company=company.name,
                    )
                    if converted:
                        parsed.append(converted)
                return parsed
        html = store.load_text(self.name, f"{slugify(company.name)}.html")
        if html:
            return self._parse_html(html, company.ats_identifier or "", company)
        return []

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
        for href, text in extract_job_links(html, page_url):
            postings.append(
                self._posting(
                    company_name=company.name,
                    title=text or None,
                    apply_url=href,
                    provenance={"parser": "icims-links", "page_url": page_url},
                )
            )
        return postings

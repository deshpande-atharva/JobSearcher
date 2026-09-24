"""Workday career-site adapter.

Workday does not publish a stable documented public jobs API. Career sites are
public HTML applications whose own frontend loads listings from a CXS path
derived from the site URL:

    https://{host}/wday/cxs/{tenant}/{site}/jobs

That path is not invented: it is the request the public career page itself
makes. This adapter tries it, then falls back to JSON-LD / link harvesting on
the public site. Either path failing is a company-level failure, not a run
failure.

Workday CXS tenants tested in this project reject ``limit`` greater than 20
with HTTP 400 and an empty message. Page size is therefore 20. HTTP 400 is
treated as a malformed request for that payload, not a transient retry.
"""

from __future__ import annotations

import re
import time
from typing import Any, ClassVar
from urllib.parse import urlparse

from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.sources.base import DiscoverySource, SourceError, SourceResult, SourceStatus
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import (
    extract_job_links,
    extract_json_ld_job_postings,
    json_ld_to_raw_posting,
    pick,
)
from src.utils.dates import parse_datetime
from src.utils.normalization import clean_text
from src.utils.urls import is_http_url, join_url

__all__ = ["CXS_PAGE_SIZE", "WorkdaySource", "parse_workday_site"]

_LANG_SEGMENTS = frozenset({"en", "en-us", "en-gb", "fr", "de", "es", "zh", "ja", "ko"})
_PATH_SKIP = frozenset({"job", "jobs", "details", "search"})

# Verified against Adobe, Capital One, Salesforce, NVIDIA, and Workday Inc:
# limit=20 → HTTP 200; limit=32 and limit=50 → HTTP 400 on every tenant.
CXS_PAGE_SIZE = 20
CXS_DEFAULT_MAX_JOBS = 2000


def parse_workday_site(identifier: str) -> tuple[str, str, str] | None:
    """Derive ``(host, tenant, site)`` from a public Workday careers URL."""
    if not identifier or not is_http_url(identifier):
        return None
    parsed = urlparse(identifier.strip())
    host = parsed.netloc.lower()
    if "myworkdayjobs.com" not in host and "myworkdaysite.com" not in host:
        return None
    tenant = host.split(".")[0]
    segments = [s for s in parsed.path.split("/") if s]
    site = next(
        (
            seg
            for seg in reversed(segments)
            if seg.lower() not in _LANG_SEGMENTS and seg.lower() not in _PATH_SKIP
        ),
        None,
    )
    if not tenant or not site:
        return None
    return host, tenant, site


class WorkdaySource(DiscoverySource):
    name: ClassVar[str] = "workday"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "workday"

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        result = await self.discover_result(company)
        if not result.success:
            raise SourceError(
                result.error or "Workday discovery failed",
                status=result.status,
                http_status=result.http_status,
                diagnostics=result.diagnostics,
            )
        return result.jobs

    async def discover_result(self, company: CompanyConfig | None = None) -> SourceResult:
        """CXS and HTML fallback are recorded separately. HTTP 400 is never EMPTY."""
        label = company.name if company else "*"
        started = time.perf_counter()
        if company is None or not company.ats_identifier:
            return SourceResult.ok(self.name, label, [], time.perf_counter() - started)

        if self.ctx.fixture_mode:
            jobs = self._from_fixture(company)
            return SourceResult.ok(self.name, label, jobs, time.perf_counter() - started)

        parsed = parse_workday_site(company.ats_identifier)
        if parsed is None:
            return SourceResult.fail(
                self.name,
                label,
                (
                    f"WORKDAY_UNSUPPORTED_CONFIGURATION: Workday ats_identifier for "
                    f"{company.name!r} is not a public myworkdayjobs.com site URL"
                ),
                time.perf_counter() - started,
                status="UNSUPPORTED",
            )

        host, tenant, site = parsed
        settings = self.config.settings.discovery.sources.workday
        page_size = min(getattr(settings, "page_size", CXS_PAGE_SIZE), CXS_PAGE_SIZE)
        max_jobs = getattr(settings, "max_jobs", CXS_DEFAULT_MAX_JOBS)
        diagnostics: dict[str, Any] = {
            "company": company.name,
            "ats": "workday",
            "tenant": tenant,
            "site": site,
            "endpoint": f"https://{host}/wday/cxs/{tenant}/{site}/jobs",
            "page_size": page_size,
            "pages": 0,
            "http_status": None,
            "fallback_used": False,
            "failure_reason": None,
        }

        try:
            entries, cxs_diag = await self._fetch_cxs(
                host,
                tenant,
                site,
                referer=company.ats_identifier,
                company=company.name,
                page_size=page_size,
                max_jobs=max_jobs,
            )
            diagnostics.update(cxs_diag)
        except SourceError as exc:
            diagnostics.update(exc.diagnostics)
            diagnostics["http_status"] = exc.http_status
            diagnostics["failure_reason"] = str(exc)
            self.log.info(
                "workday CXS unavailable, falling back to HTML",
                company=company.name,
                tenant=tenant,
                site=site,
                endpoint=diagnostics.get("endpoint"),
                http_status=exc.http_status,
                failure_reason=str(exc)[:200],
            )
            try:
                html_jobs = await self._from_html(company.ats_identifier, company)
            except SourceError as html_exc:
                duration = time.perf_counter() - started
                return SourceResult.fail(
                    self.name,
                    label,
                    f"CXS failed ({exc}); HTML fallback failed ({html_exc})",
                    duration,
                    status=exc.status,
                    http_status=exc.http_status,
                    fallback_used=True,
                    fallback_status=html_exc.status,
                    diagnostics={
                        **diagnostics,
                        "fallback_used": True,
                        "fallback_error": str(html_exc)[:200],
                    },
                )
            duration = time.perf_counter() - started
            fallback_status: SourceStatus = "OK" if html_jobs else "EMPTY"
            if html_jobs:
                return SourceResult.ok(
                    self.name,
                    label,
                    html_jobs,
                    duration,
                    http_status=exc.http_status,
                    fallback_used=True,
                    fallback_status=fallback_status,
                    diagnostics={**diagnostics, "fallback_used": True, "fallback_jobs": len(html_jobs)},
                )
            # Structured request failed and HTML found nothing. This is ERROR,
            # not EMPTY — the source did not successfully return an empty board.
            return SourceResult.fail(
                self.name,
                label,
                str(exc),
                duration,
                status=exc.status,
                http_status=exc.http_status,
                fallback_used=True,
                fallback_status=fallback_status,
                diagnostics={**diagnostics, "fallback_used": True, "fallback_jobs": 0},
            )

        postings = [self._to_posting(entry, company, host, site) for entry in entries]
        jobs = [p for p in postings if p is not None]
        return SourceResult.ok(
            self.name,
            label,
            jobs,
            time.perf_counter() - started,
            http_status=diagnostics.get("http_status") or 200,
            diagnostics=diagnostics,
        )

    def _from_fixture(self, company: CompanyConfig) -> list[RawJobPosting]:
        payload = FixtureStore(self.config.fixture_dir).load_first(
            self.name, [f"{slugify(company.name)}.json", "jobs.json"]
        )
        entries = pick(payload, "jobPostings", "jobs", default=None)
        if not isinstance(entries, list):
            return []
        return [p for p in (self._to_posting(e, company, "fixture", "site") for e in entries) if p]

    async def _fetch_cxs(
        self,
        host: str,
        tenant: str,
        site: str,
        *,
        referer: str,
        company: str,
        page_size: int,
        max_jobs: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        url = f"https://{host}/wday/cxs/{tenant}/{site}/jobs"
        collected: list[Any] = []
        pages = 0
        last_status: int | None = None
        reported_total: int | None = None
        source_list_cap_reached = False
        max_offset = max(max_jobs, page_size)

        for offset in range(0, max_offset, page_size):
            body = {"appliedFacets": {}, "limit": page_size, "offset": offset, "searchText": ""}
            result = await self.http.request(
                "POST",
                url,
                json_body=body,
                expect_json=True,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Origin": f"https://{host}",
                    "Referer": referer,
                },
            )
            last_status = result.status
            pages += 1
            self.log.info(
                "workday cxs page",
                company=company,
                ats="workday",
                tenant=tenant,
                site=site,
                endpoint=url,
                http_status=result.status,
                request_attempt=1,
                pagination_offset=offset,
                response_size=len(result.text or ""),
                fallback_used=False,
                failure_reason=None if result.ok else (result.error or f"HTTP {result.status}"),
            )
            if result.blocked_by_robots:
                raise SourceError(
                    f"Workday CXS disallowed by robots.txt: {url}",
                    status="BLOCKED",
                    http_status=result.status,
                    diagnostics={"endpoint": url, "tenant": tenant, "site": site, "pages": pages},
                )
            if not result.ok:
                reason = result.error or f"HTTP {result.status}"
                preview = result.body_preview
                if offset == 0:
                    raise SourceError(
                        f"Workday CXS {reason}",
                        http_status=result.status,
                        diagnostics={
                            "endpoint": url,
                            "tenant": tenant,
                            "site": site,
                            "http_status": result.status,
                            "pagination_offset": offset,
                            "response_size": len(result.text or ""),
                            "response_preview": preview,
                            "pages": pages,
                        },
                    )
                # A later page failed; keep jobs already collected.
                break
            payload = result.json()
            if payload is None:
                if offset == 0:
                    raise SourceError(
                        "Workday CXS did not return JSON",
                        http_status=result.status,
                        diagnostics={
                            "endpoint": url,
                            "tenant": tenant,
                            "site": site,
                            "http_status": result.status,
                            "response_preview": result.body_preview,
                            "pages": pages,
                        },
                    )
                break
            batch = pick(payload, "jobPostings", default=None)
            if not isinstance(batch, list) or not batch:
                break
            collected.extend(batch)
            total = pick(payload, "total", default=None)
            # Later Workday pages often send total=0. Only the first positive
            # total is meaningful.
            if isinstance(total, int) and total > 0 and reported_total is None:
                reported_total = total
            if reported_total is not None and len(collected) >= reported_total:
                # A full last page at a round source total (Workday CXS stops
                # at 2000 for some tenants) means the inventory may continue.
                if len(batch) >= page_size and reported_total >= max_jobs:
                    source_list_cap_reached = True
                break
            if len(batch) < page_size:
                break
            if len(collected) >= max_jobs:
                collected = collected[:max_jobs]
                source_list_cap_reached = True
                break

        if source_list_cap_reached:
            self.log.warning(
                "workday source/list cap may have been reached; inventory may be incomplete",
                company=company,
                discovered=len(collected),
                reported_total=reported_total,
                max_jobs=max_jobs,
            )

        return collected, {
            "endpoint": url,
            "tenant": tenant,
            "site": site,
            "http_status": last_status,
            "pages": pages,
            "reported_total": reported_total,
            "cxs_jobs": len(collected),
            "source_list_cap_reached": source_list_cap_reached,
            "fallback_used": False,
        }

    async def _from_html(self, url: str, company: CompanyConfig) -> list[RawJobPosting]:
        result = await self.http.get_text(url)
        if result.blocked_by_robots:
            raise SourceError(
                f"Workday HTML disallowed by robots.txt: {url}",
                status="BLOCKED",
                http_status=result.status,
            )
        if not result.ok:
            raise SourceError(
                f"Workday HTML fetch failed: {result.error or result.status}",
                http_status=result.status,
            )
        postings: list[RawJobPosting] = []
        for block in extract_json_ld_job_postings(result.text):
            converted = json_ld_to_raw_posting(
                block, source=self.name, page_url=url, fallback_company=company.name
            )
            if converted:
                postings.append(converted)
        if postings:
            return postings
        for href, text in extract_job_links(result.text, url):
            postings.append(
                self._posting(
                    company_name=company.name,
                    title=text or None,
                    apply_url=href,
                    provenance={"parser": "workday-html-links"},
                )
            )
        return postings

    def _to_posting(
        self, entry: Any, company: CompanyConfig, host: str, site: str
    ) -> RawJobPosting | None:
        if not isinstance(entry, dict):
            return None
        title = clean_text(str(pick(entry, "title") or "")) or None
        external_path = pick(entry, "externalPath")
        apply_url = None
        if isinstance(external_path, str) and external_path:
            if is_http_url(external_path):
                apply_url = external_path
            else:
                apply_url = join_url(f"https://{host}/{site}", external_path.lstrip("/"))
        if not title and not apply_url:
            return None

        locations = pick(entry, "locationsText", "location")
        location_raw = clean_text(str(locations or "")) or None
        posted_raw = pick(entry, "postedOn", "startDate")
        posted_at = parse_datetime(posted_raw)
        # Workday's "postedOn" is often "Posted 2 Days Ago" -- parse_datetime
        # handles the relative form. We never invent a calendar date.
        date_source = DateSource.POSTED_DATE if posted_at is not None else DateSource.UNKNOWN
        job_id = pick(entry, "bulletFields")
        if isinstance(job_id, list) and job_id:
            job_id = next((item for item in job_id if isinstance(item, str) and re.search(r"\d", item)), job_id[0])
        if job_id is None:
            job_id = pick(entry, "id")

        return self._posting(
            company_name=company.name,
            title=title,
            location_raw=location_raw,
            remote_type_raw=str(pick(entry, "remoteType", "timeType") or "") or None,
            job_id=str(job_id) if job_id else None,
            apply_url=apply_url,
            posted_at_raw=str(posted_raw) if posted_raw else None,
            posted_at=posted_at,
            date_source=date_source,
            provenance={"ats": "workday", "site": site},
        )

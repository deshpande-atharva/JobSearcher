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
HTTP 404 is not retried. Pagination stops on an empty page, a short page,
the reported total, or ``max_jobs`` (2000). That cap is the configured bound
and, for some tenants, the CXS list limit. Hitting it means the board was
not fully discovered.
"""

from __future__ import annotations

import re
import time
from typing import Any, ClassVar
from urllib.parse import urlparse

from src.models.config import CompanyConfig
from src.services.freshness import is_fresh
from src.models.job import DateSource, RawJobPosting
from src.sources.base import DiscoverySource, SourceError, SourceResult, SourceStatus
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import (
    collect_postings,
    extract_job_links,
    extract_json_ld_job_postings,
    json_ld_to_raw_posting,
    pick,
)
from src.utils.dates import parse_datetime
from src.utils.normalization import clean_text
from src.utils.urls import is_http_url, join_url

__all__ = ["CXS_PAGE_SIZE", "WorkdaySource", "cxs_request_body", "parse_workday_site"]

_CXS_FIELDS = ("appliedFacets", "limit", "offset", "searchText")


def cxs_request_body(*, limit: int, offset: int, search_text: str) -> dict[str, Any]:
    """Public CXS list body. Facet IDs are not guessed, so appliedFacets stays empty."""
    body: dict[str, Any] = {
        "appliedFacets": {},
        "limit": limit,
        "offset": offset,
        "searchText": search_text,
    }
    if tuple(body) != _CXS_FIELDS or body["appliedFacets"] != {}:
        raise SourceError("unsupported Workday filter")
    return body

_LANG_SEGMENTS = frozenset({"en", "en-us", "en-gb", "fr", "de", "es", "zh", "ja", "ko"})
_PATH_SKIP = frozenset({"job", "jobs", "details", "search"})

# Verified against Adobe, Capital One, Salesforce, NVIDIA, and Workday Inc:
# limit=20 → HTTP 200; limit=32 and limit=50 → HTTP 400 on every tenant.
CXS_PAGE_SIZE = 20
CXS_DEFAULT_MAX_JOBS = 2000


def _merge_workday_jobs(
    existing: list[RawJobPosting], incoming: list[RawJobPosting]
) -> tuple[int, int]:
    """Keep one row per Workday job id and record every partition that found it."""
    index: dict[str, int] = {}
    for position, posting in enumerate(existing):
        key = (posting.job_id or "").strip().lower()
        if key:
            index[key] = position
    added = 0
    duplicates = 0
    for posting in incoming:
        key = (posting.job_id or "").strip().lower()
        if key and key in index:
            host = existing[index[key]]
            parts = host.provenance.setdefault("workday_partitions", [])
            for part in posting.provenance.get("workday_partitions") or []:
                if part not in parts:
                    parts.append(part)
            duplicates += 1
            continue
        existing.append(posting)
        if key:
            index[key] = len(existing) - 1
        added += 1
    return added, duplicates


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

    async def discover_result(
        self,
        company: CompanyConfig | None = None,
        *,
        seed_page: dict[str, Any] | None = None,
    ) -> SourceResult:
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
                search_text="",
                seed_page=seed_page,
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

        jobs = collect_postings(
            entries,
            lambda entry: self._to_posting(entry, company, host, site, partition="unfiltered"),
            log=self.log,
        )
        cxs_diag["unfiltered_jobs"] = len(jobs)
        cxs_diag["unfiltered_fresh_jobs"] = self._fresh_count(jobs)
        cxs_diag["estimated_incomplete"] = bool(cxs_diag.get("source_list_cap_reached"))
        cxs_diag["recovered_fresh_jobs"] = 0
        if cxs_diag.get("source_list_cap_reached"):
            jobs, partition_diag = await self._search_partitions(
                host,
                tenant,
                site,
                referer=company.ats_identifier,
                company=company,
                page_size=page_size,
                jobs=jobs,
                pages_used=int(cxs_diag.get("pages") or 0),
            )
            cxs_diag.update(partition_diag)
        diagnostics.update(cxs_diag)
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
        return collect_postings(
            entries,
            lambda entry: self._to_posting(entry, company, "fixture", "site"),
            log=self.log,
        )

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
        search_text: str = "",
        company_pages: list[int] | None = None,
        seed_page: dict[str, Any] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        url = f"https://{host}/wday/cxs/{tenant}/{site}/jobs"
        collected: list[Any] = []
        pages = 0
        last_status: int | None = None
        reported_total: int | None = None
        source_list_cap_reached = False
        max_offset = max(max_jobs, page_size)
        settings = self.config.settings.discovery.sources.workday
        company_pages = company_pages if company_pages is not None else [0]
        # A global board already paid for offset 0 during validation. Reuse that
        # payload as page one. Configured boards pass no seed and POST as before.
        seed = seed_page if isinstance(seed_page, dict) and not search_text else None

        for offset in range(0, max_offset, page_size):
            reused = seed is not None and offset == 0
            if reused:
                payload = seed
                seed = None
                last_status = 200
                pages += 1
                company_pages[0] += 1
            else:
                if not self._workday_request_allowed(settings, company_pages[0]):
                    break
                body = cxs_request_body(limit=page_size, offset=offset, search_text=search_text)
                self._charge_workday_request(company_pages)
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
                    partition=search_text or "unfiltered",
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
                    provenance={
                        "parser": "workday-html-links",
                        "discovery_method": "html",
                        "canonical_source": "workday",
                        "discovered_from": ["workday"],
                        "source_url": href,
                    },
                )
            )
        return postings

    def _workday_request_allowed(self, settings: Any, company_pages: int) -> bool:
        run_count = int(getattr(self.http, "workday_post_count", 0) or 0)
        return (
            run_count < settings.max_requests_per_run
            and company_pages < settings.max_requests_per_company
        )

    def _charge_workday_request(self, company_pages: list[int]) -> None:
        company_pages[0] += 1
        current = int(getattr(self.http, "workday_post_count", 0) or 0)
        try:
            self.http.workday_post_count = current + 1
        except Exception:
            return

    async def _search_partitions(
        self,
        host: str,
        tenant: str,
        site: str,
        *,
        referer: str,
        company: CompanyConfig,
        page_size: int,
        jobs: list[RawJobPosting],
        pages_used: int,
    ) -> tuple[list[RawJobPosting], dict[str, Any]]:
        """Keyword slices of a capped board, using the existing searchText field."""
        settings = self.config.settings.discovery.sources.workday
        diag: dict[str, Any] = {
            "partitions_attempted": 0,
            "partitions_successful": 0,
            "partitions_failed": 0,
            "partition_jobs": 0,
            "partition_duplicates": 0,
            "partition_requests": 0,
            "recovered_fresh_jobs": 0,
        }
        if not settings.partitions_enabled or settings.max_partitions_per_company <= 0:
            return jobs, diag
        texts = [
            text.strip()
            for text in settings.partition_search_texts
            if isinstance(text, str) and text.strip()
        ][: settings.max_partitions_per_company]
        company_pages = [pages_used]
        merged = list(jobs)
        for text in texts:
            if not self._workday_request_allowed(settings, company_pages[0]):
                break
            diag["partitions_attempted"] += 1
            before = company_pages[0]
            try:
                entries, part_diag = await self._fetch_cxs(
                    host,
                    tenant,
                    site,
                    referer=referer,
                    company=company.name,
                    page_size=page_size,
                    max_jobs=settings.max_jobs_per_partition,
                    search_text=text,
                    company_pages=company_pages,
                )
            except SourceError:
                diag["partitions_failed"] += 1
                diag["partition_requests"] += company_pages[0] - before
                continue
            diag["partition_requests"] += int(part_diag.get("pages") or 0)
            if part_diag.get("http_status") and not entries and part_diag.get("pages", 0) == 0:
                diag["partitions_failed"] += 1
                continue
            part_jobs = collect_postings(
                entries,
                lambda entry, label=text: self._to_posting(
                    entry, company, host, site, partition=label
                ),
                log=self.log,
            )
            known_ids = {posting.job_id for posting in merged if posting.job_id}
            diag["recovered_fresh_jobs"] += sum(
                1
                for posting in part_jobs
                if posting.job_id not in known_ids and self._fresh_count([posting])
            )
            added, dupes = _merge_workday_jobs(merged, part_jobs)
            diag["partitions_successful"] += 1
            diag["partition_jobs"] += added
            diag["partition_duplicates"] += dupes
        return merged, diag

    def _fresh_count(self, postings: list[RawJobPosting]) -> int:
        hours = self.config.freshness_hours
        allow_updated = self.config.settings.run.freshness_use_updated_when_posted_missing
        return sum(
            1
            for posting in postings
            if is_fresh(posting, hours, use_updated_when_posted_missing=allow_updated)[0]
        )

    def _to_posting(
        self, entry: Any, company: CompanyConfig, host: str, site: str, partition: str = "unfiltered"
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
            provenance={
                "ats": "workday",
                "site": site,
                "discovery_method": "cxs",
                "canonical_source": "workday",
                "discovered_from": ["workday"],
                "workday_partitions": [partition],
                "source_url": apply_url or "",
                "source_job_id": str(job_id) if job_id else "",
            },
        )

"""Bounded public Workday board sample.

Workday does not publish a directory of customers or a universal jobs API.
This module reads a small Internet Archive CDX sample of public career hosts
(``*.wdN.myworkdayjobs.com`` and ``*.wdN.myworkdaysite.com``), checks that the
public CXS endpoint returns a job-search payload, and then calls the existing
Workday collector. Pagination, the 2,000-job cap, and ``searchText`` partitions
stay in that collector. Coverage is a partial sample, not every Workday job.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from src.models.config import CompanyConfig, CompanyDiscoveryBlock
from src.models.job import RawJobPosting
from src.models.state import PipelineState
from src.services.discovery_learning import assign_strategy, board_selection_kwargs
from src.services.public_board_index import (
    PublicBoardIndex,
    parse_cdx_workday,
    parse_public_workday_board,
    prefixes_for_day,
    select_workday_boards,
    workday_archive_query_url,
    workday_board_identity,
    workday_career_url,
    workday_clusters_for_run,
    workday_snapshot_query_url,
)
from src.sources.base import SourceContext, SourceResult
from src.sources.workday import CXS_PAGE_SIZE, WorkdaySource, cxs_request_body
from src.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "collect_global_workday",
    "configured_workday_clusters",
    "configured_workday_identities",
    "load_workday_archive_sample",
    "load_workday_snapshot_fallback",
]

_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


async def load_workday_archive_sample(
    ctx: SourceContext,
    *,
    clusters: tuple[str, ...],
    limit: int,
    now: datetime | None,
    prefixes_per_run: int,
    snapshot_fallback_enabled: bool = True,
    snapshot_window_days: int = 30,
    snapshot_max_queries: int = 4,
    snapshot_timeout_seconds: float = 15.0,
    snapshot_max_candidates: int = 40,
    snapshot_concurrency: int = 2,
) -> tuple[list[str], str, str]:
    """A few Archive CDX pages. Never raises. An empty sample is not a run failure."""
    chosen = workday_clusters_for_run(clusters, now, count=2)
    prefixes = prefixes_for_day(now, prefixes_per_run)
    # Only query myworkdayjobs.com. myworkdaysite.com CDX returns student portals
    # (e.g. wd1-student.myworkdaysite.com) that have no wdN-based career-board path
    # structure and produce zero valid boards from parse_public_workday_board.
    queries: list[str] = []
    for cluster in chosen:
        for prefix in prefixes:
            queries.append(
                workday_archive_query_url(
                    domain="myworkdayjobs.com",
                    cluster=cluster,
                    prefix=prefix,
                    limit=limit,
                )
            )
    bodies = await asyncio.gather(*(_cdx_body(ctx, url) for url in queries))
    found: list[str] = []
    seen = 0
    for body in bodies:
        urls, count = parse_cdx_workday(body)
        seen += count
        found.extend(urls)
    unique = select_workday_boards(found, limit=len(found) or 0, now=now)
    log.info(
        "cdx_workday_sample_complete",
        service="internet_archive_cdx",
        queries=len(queries),
        urls_seen=seen,
        boards_found=len(unique),
    )
    detail = (
        f"Internet Archive CDX sample for Workday clusters {', '.join(chosen)} "
        f"prefixes {''.join(prefixes)!r}; GLOBAL DISCOVERY: PARTIAL COVERAGE. "
        "This is not every Workday customer or every Workday job."
    )
    if seen == 0:
        detail = (
            "Workday archive sample returned no career URLs; configured boards still run. "
            "GLOBAL DISCOVERY: PARTIAL COVERAGE"
        )
        log.warning("workday public board sample empty")
    if unique:
        return unique, detail, "global_index"
    if not unique and snapshot_fallback_enabled:
        fallback = await load_workday_snapshot_fallback(
            ctx,
            clusters=clusters,
            limit=snapshot_max_candidates,
            now=now,
            window_days=snapshot_window_days,
            max_queries=snapshot_max_queries,
            timeout_seconds=snapshot_timeout_seconds,
            concurrency=snapshot_concurrency,
        )
        if fallback:
            fallback_detail = (
                f"Wayback Snapshot Index fallback (CDX unavailable); "
                f"clusters {', '.join(chosen)}; {len(fallback)} board(s) found. "
                "GLOBAL DISCOVERY: PARTIAL COVERAGE."
            )
            return fallback, fallback_detail, "global_snapshot_index"
    return unique, detail, "global_index"


def configured_workday_clusters(state: PipelineState) -> tuple[str, ...]:
    """``wdN`` suffixes already present on public hosts in the company list.

    The suffix is a Workday data-center label, not a company. Other tenants on
    that public host can be sampled without being added to companies.yaml.
    """
    found: list[str] = []
    for company in state.config.universe.companies:
        for raw in (company.ats_identifier, company.careers_url):
            if not raw:
                continue
            host = urlparse(str(raw).strip()).netloc.lower().split(":")[0]
            match = re.search(r"\.(wd\d{1,3})\.(?:myworkdayjobs|myworkdaysite)\.com$", host)
            if match and match.group(1) not in found:
                found.append(match.group(1))
    return tuple(found)


def configured_workday_identities(state: PipelineState) -> set[str]:
    """Boards already represented in companies.yaml, including auto-detect URLs."""
    identities: set[str] = set()
    for company in state.config.target_companies():
        for raw in (company.ats_identifier, company.careers_url):
            if not raw:
                continue
            identity = workday_board_identity(str(raw))
            if identity:
                identities.add(identity)
    return identities


async def collect_global_workday(
    state: PipelineState,
    ctx: SourceContext,
    *,
    urls: list[str],
    index: PublicBoardIndex,
    limit: int,
    now: datetime | None,
    board_origin: str = "global_index",
) -> tuple[list[RawJobPosting], dict[str, object]]:
    """Validate new boards and collect them with the existing CXS adapter."""
    configured = configured_workday_identities(state)
    tally = {
        "discovered_boards": 0,
        "duplicate_boards": 0,
        "configured_overlap": 0,
        "selected_boards": 0,
        "invalid_boards": 0,
        "failed_boards": 0,
        "empty_boards": 0,
        "complete_boards": 0,
        "partial_boards": 0,
        "valid_boards": 0,
        "jobs": 0,
        "boards_selected_for_collection": 0,
        "unique_boards_collected": 0,
        "duplicate_boards_skipped": 0,
        "index_status": index.status,
        "coverage": "partial",
        "detail": index.workday_detail
        or "GLOBAL DISCOVERY: PARTIAL COVERAGE. Not every Workday customer or job.",
        "crawl_id": index.crawl_id,
        "configured_boards": len(configured),
    }
    fresh: list[str] = []
    seen: set[str] = set()
    for url in urls:
        tally["discovered_boards"] += 1
        identity = workday_board_identity(url)
        if identity is None:
            tally["invalid_boards"] += 1
            continue
        if identity in seen:
            tally["duplicate_boards"] += 1
            continue
        seen.add(identity)
        if identity in configured:
            tally["configured_overlap"] += 1
            continue
        career = workday_career_url(url)
        if career is None:
            tally["invalid_boards"] += 1
            continue
        fresh.append(career)
    chosen = select_workday_boards(
        fresh,
        limit=limit,
        now=now,
        **board_selection_kwargs(state, "workday"),
    )
    tally["selected_boards"] = len(chosen)
    tally["boards_selected_for_collection"] = len(chosen)
    tally["duplicate_boards_skipped"] = int(tally["duplicate_boards"]) + int(
        tally["configured_overlap"]
    )
    if not chosen:
        return [], tally

    timeout = state.config.settings.run.company_timeout_seconds
    concurrency = state.config.settings.discovery.sources.workday.max_concurrency
    semaphore = state.resources.setdefault("workday_semaphore", asyncio.Semaphore(concurrency))
    collected_identities: set[str] = set()

    async def one(url: str):
        async with semaphore:
            started = time.perf_counter()
            try:
                batch, label, result, name = await asyncio.wait_for(
                    _one_board(state, ctx, url, collected_identities, board_origin=board_origin),
                    timeout=timeout,
                )
            except TimeoutError:
                log.warning("global workday board timed out", board=url)
                parsed = parse_public_workday_board(url)
                name = parsed[1] if parsed else url
                result = SourceResult.fail(
                    "workday",
                    name,
                    "global Workday board timed out",
                    time.perf_counter() - started,
                    status="ERROR",
                )
                batch, label = [], "failed"
            return batch, label, result, name, started, time.perf_counter() - started

    batches = await asyncio.gather(*(one(url) for url in chosen))
    spans = [
        (started, started + elapsed)
        for _batch, _label, _result, _name, started, elapsed in batches
    ]
    tally["collection_seconds"] = _union_seconds(spans)
    from src.agents.discovery_agent import _outcome, _record

    posts: list[RawJobPosting] = []
    for batch, label, result, name, started, _elapsed in batches:
        if result is not None:
            _record(state, result, company=name)
            career_url = None
            if result.diagnostics:
                career_url = result.diagnostics.get("career_url")
            state.company_outcomes.append(
                _outcome(
                    result,
                    ats_type="workday",
                    ats_identifier=career_url,
                    detection_method="global_index",
                    started_monotonic=started,
                )
            )
        if label == "invalid":
            tally["invalid_boards"] += 1
        elif label == "failed":
            tally["failed_boards"] += 1
        elif label == "empty":
            tally["empty_boards"] += 1
            tally["unique_boards_collected"] = int(tally["unique_boards_collected"]) + 1
        elif label == "duplicate":
            tally["duplicate_boards"] += 1
            tally["duplicate_boards_skipped"] = int(tally["duplicate_boards_skipped"]) + 1
        elif label == "partial":
            tally["partial_boards"] += 1
            tally["unique_boards_collected"] = int(tally["unique_boards_collected"]) + 1
            posts.extend(batch)
        elif label == "complete":
            tally["complete_boards"] += 1
            tally["unique_boards_collected"] = int(tally["unique_boards_collected"]) + 1
            posts.extend(batch)
    tally["jobs"] = len(posts)
    tally["valid_boards"] = (
        int(tally["complete_boards"]) + int(tally["partial_boards"]) + int(tally["empty_boards"])
    )
    return posts, tally


async def _one_board(
    state: PipelineState,
    ctx: SourceContext,
    url: str,
    collected_identities: set[str],
    board_origin: str = "global_index",
):
    parsed = parse_public_workday_board(url)
    if parsed is None:
        return [], "invalid", None, url
    host, tenant, site = parsed
    identity = f"{host}|{tenant}|{site.lower()}"
    if identity in collected_identities:
        return [], "duplicate", None, tenant
    collected_identities.add(identity)
    career = f"https://{host}/{site}"
    settings = state.config.settings.discovery.sources.workday
    if int(getattr(ctx.http, "workday_post_count", 0) or 0) >= settings.max_requests_per_run:
        result = SourceResult.fail(
            "workday",
            tenant,
            "Workday request budget exhausted before this global board",
            status="ERROR",
            diagnostics={"career_url": career, "tenant": tenant, "site": site},
        )
        return [], "failed", result, tenant

    verdict, http_status, payload = await _probe_cxs(ctx, host, tenant, site, career)
    if verdict == "invalid":
        result = SourceResult.fail(
            "workday",
            _tenant_name(tenant),
            "public CXS response was not a Workday job-search payload",
            status="ERROR",
            http_status=http_status,
            diagnostics=_board_diagnostics(career, host, tenant, site),
        )
        return [], "invalid", result, _tenant_name(tenant)
    if verdict != "ready" or payload is None:
        result = SourceResult.fail(
            "workday",
            _tenant_name(tenant),
            "public CXS request failed",
            status="ERROR",
            http_status=http_status,
            diagnostics=_board_diagnostics(career, host, tenant, site),
        )
        return [], "failed", result, _tenant_name(tenant)

    name = await _display_name(ctx, career, tenant)
    company = CompanyConfig(
        name=name,
        ats_type="workday",
        ats_identifier=career,
        discovery=CompanyDiscoveryBlock(sources=("company_ats",)),
    )
    result = await WorkdaySource(ctx).discover_result(company, seed_page=payload)
    result.diagnostics["career_url"] = career
    if not result.success:
        return [], "failed", result, name
    for posting in result.jobs:
        posting.provenance["board_origin"] = board_origin
        posting.provenance["board_identity"] = f"{host}|{tenant}|{site.lower()}"
        found = posting.provenance.setdefault("discovered_from", [])
        discovered_tag = (
            "wayback_snapshot_index" if board_origin == "global_snapshot_index" else "global_index"
        )
        if discovered_tag not in found:
            found.append(discovered_tag)
        assign_strategy(posting, origin="global_index")  # always "global_index" for learning semantics
    capped = bool(result.diagnostics.get("source_list_cap_reached"))
    incomplete = bool(result.diagnostics.get("estimated_incomplete"))
    if capped or incomplete:
        return list(result.jobs), "partial", result, name
    if not result.jobs:
        return [], "empty", result, name
    return list(result.jobs), "complete", result, name


def _board_diagnostics(career: str, host: str, tenant: str, site: str) -> dict[str, str]:
    return {
        "career_url": career,
        "tenant": tenant,
        "site": site,
        "endpoint": _cxs_url(host, tenant, site),
    }


def _cxs_url(host: str, tenant: str, site: str) -> str:
    return f"https://{host}/wday/cxs/{tenant}/{site}/jobs"


async def _probe_cxs(
    ctx: SourceContext, host: str, tenant: str, site: str, referer: str
) -> tuple[str, int | None, dict | None]:
    """One public CXS POST. ``ready`` means ``jobPostings`` is a list.

    The payload is the first collector page. The collector must not POST it again.
    """
    url = _cxs_url(host, tenant, site)
    body = cxs_request_body(limit=CXS_PAGE_SIZE, offset=0, search_text="")
    _charge(ctx)
    result = await ctx.http.request(
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
    if result.blocked_by_robots or not result.ok:
        return "failed", result.status, None
    payload = result.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("jobPostings"), list):
        return "invalid", result.status, None
    return "ready", result.status, payload


def _charge(ctx: SourceContext) -> None:
    current = int(getattr(ctx.http, "workday_post_count", 0) or 0)
    try:
        ctx.http.workday_post_count = current + 1
    except Exception:
        return


async def _display_name(ctx: SourceContext, career_url: str, tenant: str) -> str:
    try:
        result = await ctx.http.request("GET", career_url, headers={"Range": "bytes=0-8191"})
    except Exception:
        return _tenant_name(tenant)
    if not result.ok or not result.text:
        return _tenant_name(tenant)
    match = _TITLE.search(result.text[:8000])
    if match is None:
        return _tenant_name(tenant)
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    for suffix in (" Careers", " Jobs", " Job Board"):
        if title.endswith(suffix) and len(title) > len(suffix):
            title = title[: -len(suffix)].strip()
    if title.lower().startswith("careers at "):
        title = title[11:].strip()
    if not title or len(title) > 80:
        return _tenant_name(tenant)
    return title


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock span of overlapping board tasks."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for begin, finish in ordered[1:]:
        if begin <= end:
            end = max(end, finish)
        else:
            total += end - start
            start, end = begin, finish
    return round(total + (end - start), 3)


def _tenant_name(tenant: str) -> str:
    return tenant.replace("-", " ").title()


async def _cdx_body(ctx: SourceContext, url: str) -> str:
    try:
        result = await asyncio.wait_for(
            ctx.http.request("GET", url, headers={"Accept": "application/json"}),
            timeout=12,
        )
    except asyncio.TimeoutError:
        log.warning("cdx_timeout", service="internet_archive_cdx", timeout_seconds=12)
        return ""
    except OSError as exc:
        log.warning(
            "cdx_connection_error",
            service="internet_archive_cdx",
            error=type(exc).__name__,
        )
        return ""
    except Exception as exc:
        log.warning(
            "cdx_request_error",
            service="internet_archive_cdx",
            error=type(exc).__name__,
        )
        return ""
    if getattr(result, "blocked_by_robots", False):
        log.warning("cdx_blocked_by_robots", service="internet_archive_cdx")
        return ""
    if not result.ok:
        log.warning(
            "cdx_http_error",
            service="internet_archive_cdx",
            http_status=result.status,
        )
        return ""
    body = result.text or ""
    log.debug(
        "cdx_response_ok",
        service="internet_archive_cdx",
        http_status=result.status,
        response_bytes=len(body),
    )
    return body


async def _snapshot_body(ctx: SourceContext, url: str, timeout_seconds: float) -> str:
    """One Snapshot Index CDX request. Same semantics as _cdx_body, different log events."""
    try:
        result = await asyncio.wait_for(
            ctx.http.request("GET", url, headers={"Accept": "application/json"}),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        log.warning(
            "snapshot_index_timeout",
            service="internet_archive_cdx",
            timeout_seconds=timeout_seconds,
        )
        return ""
    except OSError as exc:
        log.warning(
            "snapshot_index_connection_error",
            service="internet_archive_cdx",
            error=type(exc).__name__,
        )
        return ""
    except Exception as exc:
        log.warning(
            "snapshot_index_request_error",
            service="internet_archive_cdx",
            error=type(exc).__name__,
        )
        return ""
    if getattr(result, "blocked_by_robots", False):
        log.warning("snapshot_index_http_error", service="internet_archive_cdx", http_status="blocked")
        return ""
    if not result.ok:
        log.warning(
            "snapshot_index_http_error",
            service="internet_archive_cdx",
            http_status=result.status,
        )
        return ""
    body = result.text or ""
    log.debug(
        "snapshot_index_response_ok",
        service="internet_archive_cdx",
        response_bytes=len(body),
    )
    return body


async def load_workday_snapshot_fallback(
    ctx: SourceContext,
    *,
    clusters: tuple[str, ...],
    limit: int,
    now: datetime | None,
    window_days: int = 30,
    max_queries: int = 4,
    timeout_seconds: float = 15.0,
    concurrency: int = 2,
) -> list[str]:
    """CDX date-range fallback for when resumeKey CDX returns no boards.

    Queries CDX using per-cluster domain + date-window instead of SURT-position
    resumeKey. Cheaper to serve; different rate-limit surface. Uses the same
    parse_cdx_workday validator as CDX — invalid URLs are rejected identically.

    Never raises. Returns an empty list if all queries fail or find nothing.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    to_date = moment.strftime("%Y%m%d%H%M%S")
    from_date = (moment - timedelta(days=window_days)).strftime("%Y%m%d%H%M%S")
    chosen = workday_clusters_for_run(clusters, now, count=2)
    queries = [
        workday_snapshot_query_url(
            cluster=cluster,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
        )
        for cluster in chosen
    ][:max_queries]
    log.info(
        "snapshot_index_request_started",
        service="internet_archive_cdx",
        clusters=list(chosen),
        window_days=window_days,
        queries=len(queries),
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def one(url: str) -> str:
        async with semaphore:
            return await _snapshot_body(ctx, url, timeout_seconds)

    bodies = await asyncio.gather(*(one(url) for url in queries))
    found: list[str] = []
    seen = 0
    for body in bodies:
        urls, count = parse_cdx_workday(body)
        seen += count
        found.extend(urls)
    unique = select_workday_boards(found, limit=limit, now=now)
    log.info(
        "snapshot_index_candidates_found",
        service="internet_archive_cdx",
        urls_seen=seen,
        boards_found=len(unique),
    )
    if unique:
        log.info(
            "snapshot_index_fallback_used",
            service="internet_archive_cdx",
            boards=len(unique),
        )
    return unique

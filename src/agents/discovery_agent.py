"""Discover jobs from structured ATS boards, career pages, and aggregators.

Jobright is optional. One source or company failing never stops the run.
A successful query that returns zero jobs is recorded separately from a failure.
"""

from __future__ import annotations

import asyncio
import time

from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.models.state import CompanyOutcome, PipelineState
from src.services.ats_discovery import (
    ATSDiscoveryResult,
    detect_ats_from_html,
    detect_ats_from_url,
)
from src.services.ats_registry import AtsRegistry, load_ats_registry
from src.services.discovery_learning import assign_strategy, effort_for_company
from src.services.freshness import is_fresh
from src.sources import COMPANY_SOURCES, GLOBAL_SOURCES
from src.sources.base import (
    DiscoverySource,
    SourceContext,
    SourceError,
    SourceResult,
    begin_effort,
    end_effort,
)
from src.utils.logging import get_logger
from src.utils.normalization import normalize_company_name
from src.utils.urls import canonicalize_url

log = get_logger(__name__)

_ATS_BY_TYPE = {cls.ats_type: cls for cls in COMPANY_SOURCES if cls.ats_type}
_CAREER_SOURCE = next(cls for cls in COMPANY_SOURCES if cls.name == "company_career")


async def run_discovery(state: PipelineState) -> None:
    ctx = _context(state)
    registry = load_ats_registry(state.config)
    discovered: list[RawJobPosting] = []

    discovered.extend(await _run_global_sources(state, ctx))

    companies = list(state.config.target_companies())
    state.summary.companies_attempted = len(companies)
    semaphore = asyncio.Semaphore(state.config.settings.run.max_concurrency)

    timeout = state.config.settings.run.company_timeout_seconds

    async def crawl(company: CompanyConfig) -> tuple[list[RawJobPosting], list[CompanyOutcome]]:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    _discover_company(state, ctx, registry, company),
                    timeout=timeout,
                )
            except TimeoutError:
                raise RuntimeError(
                    f"{company.name}: company discovery timed out after {timeout:.0f}s"
                ) from None

    async def crawl_all():
        return await asyncio.gather(*(crawl(c) for c in companies), return_exceptions=True)

    company_task = None
    if companies:
        company_task = asyncio.create_task(crawl_all())
        # Let configured companies reach the HTTP semaphore before the archive
        # sample queues. The sample then uses slots that open during that crawl.
        await asyncio.sleep(0)
    index_task = _schedule_public_index(state, ctx)
    # Greenhouse, Ashby, and Workday samples share the existing HTTP and Workday
    # limits. They start when the index is ready instead of waiting for every
    # configured company to finish.
    global_task = asyncio.create_task(_collect_indexed_boards(state, ctx, index_task))
    if company_task is not None:
        results = await company_task
        for company, result in zip(companies, results, strict=False):
            if isinstance(result, Exception):
                outcome = CompanyOutcome(
                    company=company.name,
                    source="unknown",
                    succeeded=False,
                    error=str(result),
                )
                state.company_outcomes.append(outcome)
                state.summary.record_source("unknown", success=False, jobs=0, error=str(result), company=company.name)
                continue
            posts, outcomes = result
            discovered.extend(posts)
            state.company_outcomes.extend(outcomes)

    failed = []
    for company in companies:
        company_outcomes = [
            outcome
            for outcome in state.company_outcomes
            if outcome.company == company.name and outcome.detection_method != "global_index"
        ]
        if not company_outcomes or not any(outcome.succeeded for outcome in company_outcomes):
            failed.append(company.name)

    state.summary.companies_succeeded = len(companies) - len(failed)
    state.summary.companies_failed = len(failed)
    state.summary.failed_companies = failed

    discovered.extend(await global_task)

    raw_count = len(discovered)
    discovered = _dedupe_raw(discovered)
    after_dedup = len(discovered)
    discovered, truncated = _apply_safety_cap(
        discovered, state.config.settings.run.max_jobs_per_run
    )
    if truncated:
        log.warning(
            "safety cap applied after dedup",
            cap=state.config.settings.run.max_jobs_per_run,
            deduped=after_dedup,
            processed=len(discovered),
            truncated=truncated,
        )
        state.summary.note(
            f"safety cap truncated {truncated} jobs after dedup "
            f"(processed {len(discovered)} of {after_dedup}; "
            f"max_jobs_per_run={state.config.settings.run.max_jobs_per_run})"
        )

    state.raw_postings = discovered
    state.summary.jobs_discovered = raw_count
    state.summary.jobs_after_cross_source_dedup = after_dedup
    state.summary.jobs_processed = len(discovered)
    state.summary.jobs_truncated = truncated
    state.summary.freshness_hours_used = state.config.freshness_hours
    if (
        state.config.settings.discovery.persist_ats_registry
        and not state.config.fixture_mode
        and not state.config.dry_run
    ):
        registry.save()
    _record_discovery_profile(state, ctx)
    log.info(
        "discovery complete",
        raw=raw_count,
        after_dedup=len(discovered),
        companies_failed=len(failed),
    )


async def _collect_indexed_boards(
    state: PipelineState,
    ctx: SourceContext,
    index_task: asyncio.Task | None,
) -> list[RawJobPosting]:
    """Load the archive sample, then collect new boards. Never raises."""
    if index_task is not None:
        try:
            state.resources["public_board_index"] = await index_task
        except Exception as exc:
            log.warning("public board index failed; continuing", error=type(exc).__name__)
            from src.services.public_board_index import PublicBoardIndex

            state.resources["public_board_index"] = PublicBoardIndex(
                status="unavailable",
                coverage="partial",
                detail=f"public board index unavailable ({type(exc).__name__})",
            )
    try:
        from src.services.global_boards import collect_global_boards

        return await collect_global_boards(
            state,
            ctx,
            index=state.resources.get("public_board_index"),
        )
    except Exception as exc:
        log.warning("global board discovery failed; continuing", error=type(exc).__name__)
        state.summary.note(
            f"global board discovery failed ({type(exc).__name__}); configured companies continued"
        )
        return []


def _schedule_public_index(state: PipelineState, ctx: SourceContext):
    """Start the archive sample unless this run already has one or must stay offline."""
    if state.config.fixture_mode or state.resources.get("public_board_index") is not None:
        return None
    settings = state.config.settings.discovery.global_boards
    if not settings.enabled:
        return None
    from src.services.global_boards import resolve_public_board_index

    return asyncio.create_task(resolve_public_board_index(state, ctx))


async def _run_global_sources(state: PipelineState, ctx: SourceContext) -> list[RawJobPosting]:
    collected: list[RawJobPosting] = []
    for source_cls in GLOBAL_SOURCES:
        source = source_cls(ctx)
        if not source.enabled:
            continue
        state.summary.sources_attempted.append(source.name)
        result = await _measure_effort(source.discover_result(None))
        _record(state, result)
        if result.success:
            collected.extend(result.jobs)
            log.info("global source complete", source=source.name, jobs=result.discovered_count)
        else:
            state.summary.failed_sources.append(source.name)
            log.warning("global source failed; continuing", source=source.name, error=result.error)
    return collected


async def _discover_company(
    state: PipelineState,
    ctx: SourceContext,
    registry: AtsRegistry,
    company: CompanyConfig,
) -> tuple[list[RawJobPosting], list[CompanyOutcome]]:
    posts: list[RawJobPosting] = []
    outcomes: list[CompanyOutcome] = []
    if not company.discovery.enabled:
        return posts, outcomes

    effort = effort_for_company(state, company.name)
    resolved, method = await resolve_company_ats(state, ctx, registry, company)
    ats_jobs: list[RawJobPosting] = []
    ats_failed = False

    if company.wants_source("company_ats") and resolved.has_structured_discovery:
        source_cls = _ATS_BY_TYPE.get(resolved.ats_type)
        if source_cls and source_cls(ctx).enabled:
            started = time.perf_counter()
            if resolved.ats_type == "workday":
                result = await _measure_effort(
                    _discover_workday(
                        state,
                        ctx,
                        source_cls,
                        resolved,
                        skip_partitions=effort == "monitor",
                    )
                )
            else:
                result = await _measure_effort(source_cls(ctx).discover_result(resolved))
            _record(state, result, company=company.name)
            outcomes.append(
                _outcome(
                    result,
                    ats_type=resolved.ats_type,
                    ats_identifier=resolved.ats_identifier,
                    detection_method=method,
                    started_monotonic=started,
                )
            )
            if result.diagnostics.get("source_list_cap_reached"):
                state.summary.note(
                    f"{company.name}: discovered {result.discovered_count}; "
                    f"{result.source_name} list cap may have been reached; "
                    "inventory may be incomplete"
                )
            if result.success:
                ats_jobs = result.jobs
                for posting in result.jobs:
                    assign_strategy(posting, origin="configured")
                posts.extend(result.jobs)
            else:
                ats_failed = True

    want_careers = company.wants_source("company_careers") and _CAREER_SOURCE(ctx).enabled
    # Monitor still fetches the ATS list. It skips the career page when that list
    # already returned jobs. An empty or failed ATS list can still fall back.
    # Career page runs when there is no ATS, ATS failed, or ATS returned nothing.
    fallback = bool(resolved.has_structured_discovery and (ats_failed or not ats_jobs))
    reason = _career_fallback_reason(resolved, method, ats_failed, ats_jobs)
    if effort == "monitor" and ats_jobs:
        reason = None
    if want_careers and reason is not None and _CAREER_SOURCE(ctx).supports(company):
        started = time.perf_counter()
        budget = state.config.settings.discovery.career_stage_budget_seconds
        try:
            result = await asyncio.wait_for(
                _measure_effort(_CAREER_SOURCE(ctx).discover_result(company)),
                timeout=budget,
            )
        except TimeoutError:
            result = SourceResult.fail(
                "company_career",
                company.name,
                f"{company.name}: career page exceeded {budget:.0f}s stage budget",
                budget,
                diagnostics={"page_outcome": "TIMEOUT"},
            )
        result.diagnostics["fallback_reason"] = reason
        _record(state, result, company=company.name)
        outcomes.append(
            _outcome(
                result,
                ats_type=resolved.ats_type,
                ats_identifier=resolved.ats_identifier,
                detection_method=method,
                fallback_used=fallback or reason != "NO_STRUCTURED_ATS",
                started_monotonic=started,
            )
        )
        if result.success:
            for posting in result.jobs:
                assign_strategy(posting, origin="configured")
            posts.extend(result.jobs)

    if not outcomes:
        empty = SourceResult.ok("none", company.name, [])
        empty.error = "no enabled source matched this company"
        _record(state, empty, company=company.name)
        outcomes.append(_outcome(empty))
    return posts, outcomes


async def resolve_company_ats(
    state: PipelineState,
    ctx: SourceContext,
    registry: AtsRegistry,
    company: CompanyConfig,
) -> tuple[CompanyConfig, str]:
    """Manual config wins, then verified registry, then URL/HTML detection.

    Returns ``(resolved_company, detection_method)``.
    """
    if company.ats_discovery_mode == "manual" and company.has_structured_discovery:
        return company, "manual"

    if company.has_structured_discovery:
        return company, "manual"

    cached = registry.get_verified(company.name)
    if cached:
        return (
            company.model_copy(
                update={"ats_type": cached.ats_type, "ats_identifier": cached.ats_identifier}
            ),
            "registry",
        )

    if not state.config.settings.discovery.auto_detect_ats:
        return company, "none"

    detection = detect_ats_from_url(company.careers_url)
    if detection is None:
        for extra in company.discovery_urls:
            detection = detect_ats_from_url(extra)
            if detection:
                break
    if detection is None and company.careers_url and not state.config.fixture_mode:
        detection = await _detect_from_page(ctx, company.careers_url)

    if detection is None or not detection.ok:
        registry.remember_failure(company.name, careers_url=company.careers_url)
        if detection is not None and detection.method == "rejected":
            return company, "uncertain"
        return company, "none"

    resolved = company.model_copy(
        update={"ats_type": detection.ats_type, "ats_identifier": detection.identifier}
    )
    registry.remember(
        company.name,
        ats_type=detection.ats_type,
        ats_identifier=detection.identifier,
        method="automatic",
        careers_url=company.careers_url,
        confidence=detection.confidence,
    )
    log.info(
        "ats detected",
        company=company.name,
        ats_type=detection.ats_type,
        identifier=detection.identifier,
        method=detection.method,
        confidence=detection.confidence,
    )
    return resolved, "automatic"


async def _resolve_ats(
    state: PipelineState,
    ctx: SourceContext,
    registry: AtsRegistry,
    company: CompanyConfig,
) -> CompanyConfig:
    resolved, _method = await resolve_company_ats(state, ctx, registry, company)
    return resolved


async def _detect_from_page(ctx: SourceContext, url: str) -> ATSDiscoveryResult | None:
    result = await ctx.http.get_text(url)
    if not result.ok:
        return detect_ats_from_url(result.url) if result.url else None
    return detect_ats_from_html(result.text, page_url=result.url or url)


async def _discover_workday(
    state: PipelineState,
    ctx: SourceContext,
    source_cls: type[DiscoverySource],
    company: CompanyConfig,
    *,
    skip_partitions: bool = False,
) -> SourceResult:
    """Bound Workday CXS concurrency without serializing other ATS adapters."""
    limit = state.config.settings.discovery.sources.workday.max_concurrency
    semaphore = state.resources.setdefault("workday_semaphore", asyncio.Semaphore(limit))
    async with semaphore:
        if skip_partitions:
            result = await source_cls(ctx).discover_result(company, skip_partitions=True)
        else:
            result = await source_cls(ctx).discover_result(company)
    from src.pilot.workday_target import browser_discovery_enabled, discover_target_jobs, workday_board_url

    if not browser_discovery_enabled(state.config):
        return result
    board = workday_board_url(company)
    if not board:
        return result
    try:
        found = await discover_target_jobs(state.config, board_url=board, company_name=company.name)
    except Exception as exc:
        log.warning("workday browser discovery failed", company=company.name, error=str(exc))
        return result
    if not found.detailed:
        return result
    merged = list(result.jobs) + list(found.detailed)
    return result.model_copy(
        update={
            "jobs": merged,
            "discovered_count": len(merged),
            "status": "OK" if merged else result.status,
        }
    )


async def _measure_effort(awaitable):
    """Attribute HTTP, retry, and browser work to the company task that did it."""
    bucket, token = begin_effort()
    try:
        result = await awaitable
    finally:
        end_effort(token)
    if isinstance(result, SourceResult):
        result.diagnostics["requests"] = bucket["requests"]
        result.diagnostics["retries"] = bucket["retries"]
        result.diagnostics["playwright_renders"] = bucket["playwright_renders"]
    return result


def _record(state: PipelineState, result: SourceResult, company: str | None = None) -> None:
    state.summary.record_source(
        result.source_name,
        success=result.success,
        jobs=result.discovered_count,
        error=result.error,
        company=company or result.company,
        status=result.status,
        http_status=result.http_status,
    )


def _career_fallback_reason(
    resolved: CompanyConfig,
    method: str,
    ats_failed: bool,
    ats_jobs: list[RawJobPosting],
) -> str | None:
    """Why a career page would run. None means the ATS result is sufficient."""
    if resolved.has_structured_discovery and ats_failed:
        return "ATS_FAILED"
    if resolved.has_structured_discovery and not ats_jobs:
        return "ATS_RETURNED_ZERO"
    if method == "uncertain":
        return "ATS_DETECTION_UNCERTAIN"
    if not resolved.has_structured_discovery:
        return "NO_STRUCTURED_ATS"
    return None


def _outcome(
    result: SourceResult,
    *,
    ats_type: str | None = None,
    ats_identifier: str | None = None,
    detection_method: str | None = None,
    fallback_used: bool = False,
    started_monotonic: float | None = None,
) -> CompanyOutcome:
    return CompanyOutcome(
        company=result.company,
        source=result.source_name,
        succeeded=result.success,
        jobs_found=result.discovered_count,
        error=result.error,
        duration_seconds=result.duration_seconds,
        ats_type=ats_type,
        ats_identifier=ats_identifier,
        detection_method=detection_method,
        fallback_used=fallback_used or result.fallback_used,
        started_monotonic=started_monotonic,
        status=result.status,
        http_status=result.http_status,
        fallback_status=result.fallback_status,
        diagnostics=result.diagnostics,
    )


def _apply_safety_cap(
    postings: list[RawJobPosting], cap: int
) -> tuple[list[RawJobPosting], int]:
    """Keep at most ``cap`` jobs with deterministic round-robin by company.

    ``cap <= 0`` means process every job. Truncation never prefers one
    company or source over another.
    """
    if cap <= 0 or len(postings) <= cap:
        return postings, 0
    buckets: dict[str, list[RawJobPosting]] = {}
    order: list[str] = []
    for posting in postings:
        key = normalize_company_name(posting.company_name) or posting.source or "_"
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(posting)
    kept: list[RawJobPosting] = []
    while len(kept) < cap and any(buckets[k] for k in order):
        for key in order:
            if buckets[key]:
                kept.append(buckets[key].pop(0))
                if len(kept) >= cap:
                    break
    return kept, len(postings) - len(kept)


def _dedupe_raw(postings: list[RawJobPosting]) -> list[RawJobPosting]:
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    unique: list[RawJobPosting] = []
    for posting in postings:
        company = normalize_company_name(posting.company_name)
        if company and posting.job_id:
            key = f"{company}|id:{posting.job_id.strip().lower()}"
            if key in seen_ids:
                _remember_source(unique, key, posting)
                continue
            seen_ids.add(key)
        _remember_source_on(posting, posting.source)
        url = canonicalize_url(posting.apply_url)
        if url:
            if url in seen_urls:
                _remember_source(unique, url, posting)
                continue
            seen_urls.add(url)
        unique.append(posting)
    return unique


def _remember_source_on(posting: RawJobPosting, source: str | None) -> None:
    found = posting.provenance.setdefault("discovered_from", [])
    if source and source not in found:
        found.append(source)


def _remember_source(unique: list[RawJobPosting], key: str, incoming: RawJobPosting) -> None:
    from src.utils.normalization import normalize_company_name
    from src.utils.urls import canonicalize_url

    for posting in unique:
        company = normalize_company_name(posting.company_name)
        id_key = (
            f"{company}|id:{posting.job_id.strip().lower()}"
            if company and posting.job_id
            else ""
        )
        url_key = canonicalize_url(posting.apply_url) or ""
        if key in {id_key, url_key}:
            _remember_source_on(posting, posting.source)
            _remember_source_on(posting, incoming.source)
            for item in incoming.provenance.get("discovered_from") or []:
                _remember_source_on(posting, str(item))
            if not posting.provenance.get("strategy_id") and incoming.provenance.get("strategy_id"):
                posting.provenance["strategy_id"] = incoming.provenance["strategy_id"]
            _remember_partitions(posting, incoming)
            _merge_authoritative_timestamp(posting, incoming)
            return


def _merge_authoritative_timestamp(host: RawJobPosting, incoming: RawJobPosting) -> None:
    """Keep the strongest posted timestamp. Never replace one with a weaker copy.

    Posted beats updated. A known posted timestamp is never overwritten by a
    later stale, unknown, or discovery-time value. An unknown host may adopt a
    later source's posted timestamp. ``discovered_at`` is never copied.
    """
    from src.models.job import DateSource

    if host.posted_at is not None:
        if incoming.posted_at is not None and incoming.posted_at > host.posted_at:
            host.posted_at = incoming.posted_at
            host.posted_at_raw = incoming.posted_at_raw
            host.date_source = DateSource.POSTED_DATE
        return
    if incoming.posted_at is not None:
        host.posted_at = incoming.posted_at
        host.posted_at_raw = incoming.posted_at_raw
        host.date_source = DateSource.POSTED_DATE
        return
    if host.updated_at is not None or incoming.updated_at is None:
        return
    host.updated_at = incoming.updated_at
    host.updated_at_raw = incoming.updated_at_raw
    if host.date_source is DateSource.UNKNOWN:
        host.date_source = DateSource.UPDATED_DATE


def _remember_partitions(host: RawJobPosting, incoming: RawJobPosting) -> None:
    extra = incoming.provenance.get("workday_partitions") or []
    if not extra:
        return
    parts = host.provenance.setdefault("workday_partitions", [])
    for part in extra:
        if part not in parts:
            parts.append(part)


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock span of overlapping company tasks. Not the sum of durations."""
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


def _record_discovery_profile(state: PipelineState, ctx: SourceContext) -> None:
    """Aggregate per-source time. Company durations overlap, so their sum is not wall clock."""
    by_source: dict[str, dict[str, float | int]] = {}
    intervals: dict[str, list[tuple[float, float]]] = {}
    fallback_reasons: dict[str, int] = {}
    page_outcomes: dict[str, int] = {}
    partitions = {
        "companies_capped": 0,
        "attempted": 0,
        "successful": 0,
        "failed": 0,
        "jobs": 0,
        "duplicates": 0,
        "requests": 0,
        "recovered_fresh_jobs": 0,
    }
    detections = {"automatic": 0, "uncertain": 0, "rejected_skipped": 0}
    workday_boards: list[dict[str, object]] = []
    for outcome in state.company_outcomes:
        name = outcome.source or "unknown"
        bucket = by_source.setdefault(
            name,
            {
                "seconds_sum": 0.0,
                "wall_seconds": 0.0,
                "jobs": 0,
                "companies": 0,
                "failures": 0,
                "fallbacks": 0,
                "timeouts": 0,
                "http_failures": 0,
                "caps": 0,
                "requests": 0,
            },
        )
        bucket["seconds_sum"] = round(
            float(bucket["seconds_sum"]) + float(outcome.duration_seconds or 0), 3
        )
        bucket["jobs"] = int(bucket["jobs"]) + int(outcome.jobs_found or 0)
        bucket["companies"] = int(bucket["companies"]) + 1
        if not outcome.succeeded:
            bucket["failures"] = int(bucket["failures"]) + 1
        if outcome.fallback_used:
            bucket["fallbacks"] = int(bucket["fallbacks"]) + 1
        if outcome.error and "timed out" in outcome.error:
            bucket["timeouts"] = int(bucket["timeouts"]) + 1
        if outcome.http_status is not None and outcome.http_status >= 400:
            bucket["http_failures"] = int(bucket["http_failures"]) + 1
        diag = outcome.diagnostics or {}
        if diag.get("source_list_cap_reached"):
            bucket["caps"] = int(bucket["caps"]) + 1
            if name == "workday":
                partitions["companies_capped"] += 1
        pages = diag.get("pages")
        if isinstance(pages, int):
            bucket["requests"] = int(bucket["requests"]) + pages
        elif name in {"greenhouse", "lever", "ashby"} and outcome.succeeded:
            bucket["requests"] = int(bucket["requests"]) + 1
        reason = diag.get("fallback_reason")
        if isinstance(reason, str):
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        page_outcome = diag.get("page_outcome")
        if isinstance(page_outcome, str) and page_outcome:
            page_outcomes[page_outcome] = page_outcomes.get(page_outcome, 0) + 1
        if name == "workday":
            partitions["attempted"] += int(diag.get("partitions_attempted") or 0)
            partitions["successful"] += int(diag.get("partitions_successful") or 0)
            partitions["failed"] += int(diag.get("partitions_failed") or 0)
            partitions["jobs"] += int(diag.get("partition_jobs") or 0)
            partitions["duplicates"] += int(diag.get("partition_duplicates") or 0)
            partitions["requests"] += int(diag.get("partition_requests") or 0)
            partitions["recovered_fresh_jobs"] = int(partitions.get("recovered_fresh_jobs") or 0) + int(
                diag.get("recovered_fresh_jobs") or 0
            )
            bucket["requests"] = int(bucket["requests"]) + int(diag.get("partition_requests") or 0)
            if diag.get("source_list_cap_reached"):
                workday_boards.append(
                    {
                        "company": outcome.company,
                        "unfiltered_jobs": int(diag.get("unfiltered_jobs") or diag.get("cxs_jobs") or 0),
                        "reported_total": diag.get("reported_total"),
                        "cap_reached": True,
                        "partitions_attempted": int(diag.get("partitions_attempted") or 0),
                        "partition_requests": int(diag.get("partition_requests") or 0),
                        "partition_jobs": int(diag.get("partition_jobs") or 0),
                        "duplicates": int(diag.get("partition_duplicates") or 0),
                        "unique_count": int(outcome.jobs_found or 0),
                        "estimated_incomplete": True,
                        "recovered_fresh_jobs": int(diag.get("recovered_fresh_jobs") or 0),
                        "unfiltered_fresh_jobs": int(diag.get("unfiltered_fresh_jobs") or 0),
                    }
                )
        if outcome.started_monotonic is not None and outcome.duration_seconds is not None:
            intervals.setdefault(name, []).append(
                (outcome.started_monotonic, outcome.started_monotonic + float(outcome.duration_seconds))
            )
        if outcome.detection_method == "automatic":
            detections["automatic"] += 1
        elif outcome.detection_method == "uncertain":
            detections["uncertain"] += 1
            detections["rejected_skipped"] += 1
    for name, spans in intervals.items():
        if name in by_source:
            by_source[name]["wall_seconds"] = _union_seconds(spans)
    hours = state.config.freshness_hours
    freshness = {"fresh": 0, "stale": 0, "unknown": 0, "by_source": {}}
    attribution: dict[str, dict] = {}
    date_sources = {item.value: 0 for item in DateSource}
    allow_updated = state.config.settings.run.freshness_use_updated_when_posted_missing
    for posting in state.raw_postings:
        fresh, _age = is_fresh(
            posting,
            hours,
            use_updated_when_posted_missing=allow_updated,
        )
        source_name = posting.source or "unknown"
        source_counts = freshness["by_source"].setdefault(
            source_name, {"fresh": 0, "stale": 0, "unknown": 0}
        )
        channel = posting.date_source.value if posting.date_source else DateSource.UNKNOWN.value
        if channel not in date_sources:
            channel = DateSource.UNKNOWN.value
        date_sources[channel] += 1
        row = attribution.setdefault(
            source_name,
            {
                "discovered": 0,
                "posted_known": 0,
                "updated_known": 0,
                "posted_unknown": 0,
                "fresh": 0,
                "stale": 0,
                "unknown": 0,
                "date_sources": {item.value: 0 for item in DateSource},
            },
        )
        row["discovered"] += 1
        if posting.posted_at is not None:
            row["posted_known"] += 1
        else:
            row["posted_unknown"] += 1
        if posting.updated_at is not None:
            row["updated_known"] += 1
        row["date_sources"][channel] += 1
        if posting.posted_at is None and posting.updated_at is None:
            bucket_name = "unknown"
        elif fresh:
            bucket_name = "fresh"
        else:
            bucket_name = "stale"
        freshness[bucket_name] = int(freshness[bucket_name]) + 1
        source_counts[bucket_name] = int(source_counts[bucket_name]) + 1
        row[bucket_name] += 1
    for row in attribution.values():
        discovered = int(row["discovered"] or 0)
        row["fresh_rate"] = round(int(row["fresh"]) / discovered, 4) if discovered else 0.0
        row["unknown_rate"] = round(int(row["unknown"]) / discovered, 4) if discovered else 0.0
    slowest = sorted(
        state.company_outcomes,
        key=lambda item: float(item.duration_seconds or 0),
        reverse=True,
    )[:8]
    http = getattr(ctx, "http", None)
    renderer = getattr(http, "_renderer", None) if http is not None else None
    profile = {
        "by_source": by_source,
        "note": "seconds_sum overlaps; wall_seconds is the union of company intervals",
        "slowest_companies": [
            {
                "company": item.company,
                "source": item.source,
                "seconds": round(float(item.duration_seconds or 0), 3),
                "jobs": item.jobs_found,
                "fallback": item.fallback_used,
                "fallback_reason": (item.diagnostics or {}).get("fallback_reason"),
                "cap": bool((item.diagnostics or {}).get("source_list_cap_reached")),
                "ok": item.succeeded,
                "status": item.status,
                "requests": int((item.diagnostics or {}).get("requests") or 0),
                "retries": int((item.diagnostics or {}).get("retries") or 0),
                "playwright_renders": int((item.diagnostics or {}).get("playwright_renders") or 0),
                "failure": None if item.succeeded else (item.error or "")[:160],
            }
            for item in slowest
        ],
        "http_requests": int(getattr(http, "request_count", 0) or 0),
        "http_cache_hits": int(getattr(http, "cache_hits", 0) or 0),
        "http_retries": int(getattr(http, "retry_count", 0) or 0),
        "workday_posts": int(getattr(http, "workday_post_count", 0) or 0),
        "playwright_renders": int(getattr(renderer, "render_count", 0) or 0),
        "playwright_failures": int(getattr(renderer, "render_failures", 0) or 0),
        "playwright_seconds": round(float(getattr(renderer, "render_seconds", 0.0) or 0.0), 3),
        "fallback_reasons": fallback_reasons,
        "career_page_outcomes": page_outcomes,
        "workday_partitions": partitions,
        "workday_boards": workday_boards,
        "ats_detection": detections,
        "freshness_preview": freshness,
        "freshness_attribution": attribution,
        "date_source_counts": date_sources,
    }
    global_boards = (state.summary.discovery_profile or {}).get("global_boards")
    if global_boards:
        profile["global_boards"] = global_boards
    state.summary.discovery_profile = profile
    log.info(
        "discovery profile",
        http_requests=profile["http_requests"],
        http_cache_hits=profile["http_cache_hits"],
        http_retries=profile["http_retries"],
        playwright_renders=profile["playwright_renders"],
        playwright_failures=profile["playwright_failures"],
        playwright_seconds=profile["playwright_seconds"],
        workday_partitions=partitions,
        fallback_reasons=fallback_reasons,
    )


def _context(state: PipelineState) -> SourceContext:
    http = state.resources.get("http")
    if http is None:
        raise SourceError("HTTP client was not injected into pipeline resources")
    return SourceContext(config=state.config, http=http, logger=log)

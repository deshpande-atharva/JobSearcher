"""Discover jobs from structured ATS boards, career pages, and aggregators.

Jobright is optional. One source or company failing never stops the run.
A successful query that returns zero jobs is recorded separately from a failure.
"""

from __future__ import annotations

import asyncio

from src.models.config import CompanyConfig
from src.models.job import RawJobPosting
from src.models.state import CompanyOutcome, PipelineState
from src.services.ats_discovery import (
    ATSDiscoveryResult,
    detect_ats_from_html,
    detect_ats_from_url,
)
from src.services.ats_registry import AtsRegistry, load_ats_registry
from src.sources import COMPANY_SOURCES, GLOBAL_SOURCES
from src.sources.base import DiscoverySource, SourceContext, SourceError, SourceResult
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

    async def crawl(company: CompanyConfig) -> tuple[list[RawJobPosting], list[CompanyOutcome]]:
        async with semaphore:
            return await _discover_company(state, ctx, registry, company)

    if companies:
        results = await asyncio.gather(*(crawl(c) for c in companies), return_exceptions=True)
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
        company_outcomes = [o for o in state.company_outcomes if o.company == company.name]
        if not company_outcomes or not any(o.succeeded for o in company_outcomes):
            failed.append(company.name)

    state.summary.companies_succeeded = len(companies) - len(failed)
    state.summary.companies_failed = len(failed)
    state.summary.failed_companies = failed

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
    log.info(
        "discovery complete",
        raw=raw_count,
        after_dedup=len(discovered),
        companies_failed=len(failed),
    )


async def _run_global_sources(state: PipelineState, ctx: SourceContext) -> list[RawJobPosting]:
    collected: list[RawJobPosting] = []
    for source_cls in GLOBAL_SOURCES:
        source = source_cls(ctx)
        if not source.enabled:
            continue
        state.summary.sources_attempted.append(source.name)
        result = await source.discover_result(None)
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

    resolved, method = await resolve_company_ats(state, ctx, registry, company)
    ats_jobs: list[RawJobPosting] = []
    ats_failed = False

    if company.wants_source("company_ats") and resolved.has_structured_discovery:
        source_cls = _ATS_BY_TYPE.get(resolved.ats_type)
        if source_cls and source_cls(ctx).enabled:
            if resolved.ats_type == "workday":
                result = await _discover_workday(state, ctx, source_cls, resolved)
            else:
                result = await source_cls(ctx).discover_result(resolved)
            _record(state, result, company=company.name)
            outcomes.append(
                _outcome(
                    result,
                    ats_type=resolved.ats_type,
                    ats_identifier=resolved.ats_identifier,
                    detection_method=method,
                )
            )
            if result.diagnostics.get("source_list_cap_reached"):
                state.summary.note(
                    f"{company.name}: discovered {result.discovered_count}; "
                    "Workday source/list cap may have been reached; "
                    "inventory may be incomplete"
                )
            if result.success:
                ats_jobs = result.jobs
                posts.extend(result.jobs)
            else:
                ats_failed = True

    want_careers = company.wants_source("company_careers") and _CAREER_SOURCE(ctx).enabled
    # Career page runs when there is no ATS, ATS failed, or ATS returned nothing.
    fallback = bool(resolved.has_structured_discovery and (ats_failed or not ats_jobs))
    if want_careers and (not resolved.has_structured_discovery or ats_failed or not ats_jobs):
        if _CAREER_SOURCE(ctx).supports(company):
            result = await _CAREER_SOURCE(ctx).discover_result(company)
            _record(state, result, company=company.name)
            outcomes.append(
                _outcome(
                    result,
                    ats_type=resolved.ats_type,
                    ats_identifier=resolved.ats_identifier,
                    detection_method=method,
                    fallback_used=fallback,
                )
            )
            if result.success:
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
) -> SourceResult:
    """Bound Workday CXS concurrency without serializing other ATS adapters."""
    limit = state.config.settings.discovery.sources.workday.max_concurrency
    semaphore = state.resources.setdefault("workday_semaphore", asyncio.Semaphore(limit))
    async with semaphore:
        return await source_cls(ctx).discover_result(company)


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


def _outcome(
    result: SourceResult,
    *,
    ats_type: str | None = None,
    ats_identifier: str | None = None,
    detection_method: str | None = None,
    fallback_used: bool = False,
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
                continue
            seen_ids.add(key)
        url = canonicalize_url(posting.apply_url)
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        unique.append(posting)
    return unique


def _context(state: PipelineState) -> SourceContext:
    http = state.resources.get("http")
    if http is None:
        raise SourceError("HTTP client was not injected into pipeline resources")
    return SourceContext(config=state.config, http=http, logger=log)

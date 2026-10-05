"""Collect public Greenhouse, Ashby, and Workday boards that are not in companies.yaml.

Configured companies are still collected by the discovery agent. This pass adds
a bounded archive sample. Greenhouse and Ashby tokens are checked on each
vendor's public API. Workday boards are checked on the public CXS endpoint and
then collected by the existing Workday adapter. One bad board does not stop
the others. Career pages stay on the configured-company path. Workday coverage
is a partial sample, not every Workday job.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from src.models.config import CompanyConfig, CompanyDiscoveryBlock
from src.models.job import RawJobPosting
from src.models.state import PipelineState
from src.services.discovery_learning import assign_strategy, board_selection_kwargs
from src.services.public_board_index import (
    ASHBY_HOST,
    PublicBoardIndex,
    archive_query_url,
    parse_cdx_tokens,
    prefixes_for_day,
    select_board_tokens,
    valid_board_token,
)
from src.sources.ashby import AshbySource
from src.sources.base import SourceContext, SourceError
from src.sources.greenhouse import GreenhouseSource
from src.utils.logging import get_logger

log = get_logger(__name__)

_TITLE = re.compile(r"<title>\s*([^<]+?)\s*</title>", re.IGNORECASE)
_GREENHOUSE_BOARD = "https://boards-api.greenhouse.io/v1/boards/{token}"


@dataclass
class _BoardTally:
    discovered: int = 0
    duplicate_boards: int = 0
    configured_overlap: int = 0
    selected: int = 0
    invalid: int = 0
    failed: int = 0
    empty: int = 0
    complete: int = 0
    partial: int = 0
    jobs: int = 0
    index_status: str = "unavailable"
    coverage: str = "partial"
    detail: str = ""
    crawl_id: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "discovered_boards": self.discovered,
            "duplicate_boards": self.duplicate_boards,
            "configured_overlap": self.configured_overlap,
            "selected_boards": self.selected,
            "invalid_boards": self.invalid,
            "failed_boards": self.failed,
            "empty_boards": self.empty,
            "complete_boards": self.complete,
            "partial_boards": self.partial,
            "valid_boards": self.complete + self.partial + self.empty,
            "jobs": self.jobs,
            "index_status": self.index_status,
            "coverage": self.coverage,
            "detail": self.detail,
            "crawl_id": self.crawl_id,
        }


async def resolve_public_board_index(state: PipelineState, ctx: SourceContext) -> PublicBoardIndex:
    """Load the Greenhouse, Ashby, and Workday archive samples.

    Safe to start while configured companies are still running. A failure
    returns an empty index and does not raise.
    """
    settings = state.config.settings.discovery.global_boards
    started = time.monotonic()
    workday_enabled = (
        settings.workday and state.config.settings.discovery.sources.workday.enabled
    )
    workday_task = None
    if workday_enabled:
        from src.services.global_workday import (
            configured_workday_clusters,
            load_workday_archive_sample,
        )

        workday_task = asyncio.create_task(
            load_workday_archive_sample(
                ctx,
                clusters=configured_workday_clusters(state),
                limit=settings.max_index_urls,
                now=state.summary.run_started_at,
                prefixes_per_run=settings.prefixes_per_run,
                snapshot_fallback_enabled=settings.workday_snapshot_fallback_enabled,
                snapshot_window_days=settings.workday_snapshot_window_days,
                snapshot_max_queries=settings.workday_snapshot_max_queries,
                snapshot_timeout_seconds=settings.workday_snapshot_timeout_seconds,
                snapshot_max_candidates=settings.workday_snapshot_max_candidates,
                snapshot_concurrency=settings.workday_snapshot_concurrency,
            )
        )
    try:
        index = await asyncio.wait_for(
            load_public_board_index(
                ctx,
                limit=settings.max_index_urls,
                now=state.summary.run_started_at,
                prefixes_per_run=settings.prefixes_per_run,
            ),
            timeout=settings.index_timeout_seconds,
        )
    except TimeoutError:
        index = PublicBoardIndex(
            status="unavailable",
            coverage="partial",
            detail="public board index timed out; configured companies still run",
        )
    except Exception as exc:
        index = PublicBoardIndex(
            status="unavailable",
            coverage="partial",
            detail=f"public board index unavailable ({type(exc).__name__})",
        )
    if workday_task is not None:
        remaining = settings.index_timeout_seconds - (time.monotonic() - started)
        try:
            urls, detail, board_origin = await asyncio.wait_for(workday_task, timeout=max(remaining, 0.1))
            index.workday = urls
            index.workday_detail = detail
            index.workday_board_origin = board_origin
        except TimeoutError:
            workday_task.cancel()
            index.workday = []
            index.workday_detail = (
                "Workday archive sample timed out; configured boards still run. "
                "GLOBAL DISCOVERY: PARTIAL COVERAGE"
            )
            log.warning("workday public board sample timed out")
        except Exception as exc:
            workday_task.cancel()
            index.workday = []
            index.workday_detail = (
                f"Workday archive sample unavailable ({type(exc).__name__}); "
                "configured boards still run. GLOBAL DISCOVERY: PARTIAL COVERAGE"
            )
            log.warning("workday public board sample failed", error=type(exc).__name__)
    index.index_seconds = round(time.monotonic() - started, 3)
    return index


async def collect_global_boards(
    state: PipelineState,
    ctx: SourceContext,
    *,
    index: PublicBoardIndex | None = None,
) -> list[RawJobPosting]:
    """Return postings from validated boards. Failures stay inside the tally."""
    settings = state.config.settings.discovery.global_boards
    profile = state.summary.discovery_profile.setdefault("global_boards", {})
    if not settings.enabled:
        profile["status"] = "disabled"
        return []
    if index is None and state.config.fixture_mode:
        profile["status"] = "fixture_skipped"
        profile["coverage"] = "not_queried"
        profile["detail"] = "fixture mode does not query the public board index"
        return []
    if index is None:
        index = await resolve_public_board_index(state, ctx)

    collected: list[RawJobPosting] = []
    now = state.summary.run_started_at
    workday_enabled = (
        settings.workday and state.config.settings.discovery.sources.workday.enabled
    )
    if settings.greenhouse and state.config.settings.discovery.sources.greenhouse.enabled:
        posts, tally = await _collect_kind(
            state,
            ctx,
            kind="greenhouse",
            tokens=index.greenhouse,
            index=index,
            limit=settings.max_boards_per_run,
            now=now,
        )
        collected.extend(posts)
        profile["greenhouse"] = tally.as_dict()
    if settings.ashby and state.config.settings.discovery.sources.ashby.enabled:
        posts, tally = await _collect_kind(
            state,
            ctx,
            kind="ashby",
            tokens=index.ashby,
            index=index,
            limit=settings.max_boards_per_run,
            now=now,
        )
        collected.extend(posts)
        profile["ashby"] = tally.as_dict()
    if workday_enabled:
        from src.services.global_workday import collect_global_workday

        posts, workday_tally = await collect_global_workday(
            state,
            ctx,
            urls=index.workday,
            index=index,
            limit=settings.max_boards_per_run,
            now=now,
            board_origin=index.workday_board_origin,
        )
        collected.extend(posts)
        profile["workday"] = workday_tally
    else:
        profile["workday"] = {
            "status": "disabled",
            "coverage": "not_queried",
            "configured_boards": 0,
        }
    profile["status"] = index.status
    profile["coverage"] = "partial"
    profile["index_seconds"] = round(float(getattr(index, "index_seconds", 0.0) or 0.0), 3)
    return collected


async def load_public_board_index(
    ctx: SourceContext,
    *,
    limit: int,
    now=None,
    prefixes_per_run: int = 4,
) -> PublicBoardIndex:
    """A few Archive CDX pages per public board host. Never raises.

    The sample is several letters or digits, spaced across the alphabet and
    rotated by UTC date. A missing index does not fail the daily run.
    """
    index = PublicBoardIndex(coverage="partial", index_source="internet_archive_cdx")
    prefixes = prefixes_for_day(now, prefixes_per_run)
    index.prefix = "".join(prefixes)
    index.crawl_id = "archive:" + index.prefix
    hosts = (
        ("greenhouse", "job-boards.greenhouse.io"),
        ("greenhouse", "boards.greenhouse.io"),
        ("ashby", ASHBY_HOST),
    )
    queries = [
        (kind, archive_query_url(host, prefix, limit=limit))
        for prefix in prefixes
        for kind, host in hosts
    ]
    bodies = await asyncio.gather(*(_cdx_body(ctx, url) for _kind, url in queries))
    greenhouse_tokens: list[str] = []
    ashby_tokens: list[str] = []
    urls = 0
    for (_kind, _url), body in zip(queries, bodies, strict=False):
        tokens, seen = parse_cdx_tokens(body, kind=_kind)
        urls += seen
        if _kind == "ashby":
            ashby_tokens.extend(tokens)
        else:
            greenhouse_tokens.extend(tokens)
    index.greenhouse = _unique(greenhouse_tokens)
    index.ashby = _unique(ashby_tokens)
    index.urls_seen = urls
    log.info(
        "cdx_board_index_sample_complete",
        service="internet_archive_cdx",
        queries=len(queries),
        urls_seen=urls,
        greenhouse_found=len(index.greenhouse),
        ashby_found=len(index.ashby),
    )
    if not any(bodies):
        index.status = "unavailable"
        index.detail = "Internet Archive CDX returned no rows; configured companies still run"
        log.warning("public board index unavailable")
        return index
    index.status = "ok"
    index.detail = (
        f"Internet Archive CDX sample for prefixes {index.prefix!r}; "
        "not a complete Greenhouse or Ashby directory"
    )
    return index


def _unique(tokens: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
    return unique


async def _collect_kind(
    state: PipelineState,
    ctx: SourceContext,
    *,
    kind: str,
    tokens: list[str],
    index: PublicBoardIndex,
    limit: int,
    now,
) -> tuple[list[RawJobPosting], _BoardTally]:
    tally = _BoardTally(
        index_status=index.status,
        coverage=index.coverage,
        detail=index.detail,
        crawl_id=index.crawl_id,
    )
    configured = _configured_tokens(state, kind)
    fresh: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        tally.discovered += 1
        if not valid_board_token(token):
            tally.invalid += 1
            continue
        key = token.lower()
        if key in seen:
            tally.duplicate_boards += 1
            continue
        seen.add(key)
        if key in configured:
            tally.configured_overlap += 1
            continue
        fresh.append(token)
    chosen = select_board_tokens(
        fresh,
        limit=limit,
        now=now,
        **board_selection_kwargs(state, kind),
    )
    tally.selected = len(chosen)
    if not chosen:
        return [], tally

    timeout = state.config.settings.discovery.global_boards.board_timeout_seconds
    semaphore = asyncio.Semaphore(state.config.settings.run.max_concurrency)

    async def one(token: str):
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    _one_board(ctx, kind, token),
                    timeout=timeout,
                )
            except TimeoutError:
                log.warning("global board timed out", source=kind, board=token)
                return [], "failed", None, token

    batches = await asyncio.gather(*(one(token) for token in chosen))
    from src.agents.discovery_agent import _record

    posts: list[RawJobPosting] = []
    for batch, label, result, name in batches:
        if result is not None:
            _record(state, result, company=name)
        if label == "invalid":
            tally.invalid += 1
        elif label == "failed":
            tally.failed += 1
        elif label == "empty":
            tally.empty += 1
        elif label == "partial":
            tally.partial += 1
            posts.extend(batch)
        elif label == "complete":
            tally.complete += 1
            posts.extend(batch)
    tally.jobs = len(posts)
    return posts, tally


async def _one_board(ctx: SourceContext, kind: str, token: str):
    try:
        name = await _board_name(ctx, kind, token)
    except SourceError as exc:
        label = "invalid" if exc.http_status == 404 else "failed"
        log.info("global board rejected", source=kind, board=token, http_status=exc.http_status)
        return [], label, None, token
    company = CompanyConfig(
        name=name,
        ats_type=kind,  # type: ignore[arg-type]
        ats_identifier=token,
        discovery=CompanyDiscoveryBlock(sources=("company_ats",)),
    )
    source_cls = GreenhouseSource if kind == "greenhouse" else AshbySource
    result = await source_cls(ctx).discover_result(company)
    if not result.success:
        return [], "invalid" if result.http_status == 404 else "failed", result, name
    for posting in result.jobs:
        posting.provenance["board_origin"] = "global_index"
        posting.provenance["board_token"] = token
        found = posting.provenance.setdefault("discovered_from", [])
        if "global_index" not in found:
            found.append("global_index")
        assign_strategy(posting, origin="global_index")
    if not result.jobs:
        return [], "empty", result, name
    if result.diagnostics.get("cap_reached") or result.diagnostics.get("source_list_cap_reached"):
        return list(result.jobs), "partial", result, name
    return list(result.jobs), "complete", result, name


async def _board_name(ctx: SourceContext, kind: str, token: str) -> str:
    if kind == "greenhouse":
        payload = await ctx.http.get_json(_GREENHOUSE_BOARD.format(token=token))
        if isinstance(payload, dict):
            name = payload.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return token
    result = await ctx.http.request(
        "GET",
        f"https://{ASHBY_HOST}/{token}",
        headers={"Range": "bytes=0-8191"},
    )
    if result.status == 404:
        raise SourceError(f"Ashby board {token} was not found", http_status=404)
    title = _company_name_from_html(result.text or "")
    return title or token


def _company_name_from_html(html: str) -> str | None:
    match = _TITLE.search(html[:8000])
    if match is None:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    for suffix in (" Jobs", " Careers", " Job Board"):
        if title.endswith(suffix) and len(title) > len(suffix):
            return title[: -len(suffix)].strip()
    return None


def _configured_tokens(state: PipelineState, kind: str) -> set[str]:
    from urllib.parse import urlparse

    tokens: set[str] = set()
    for company in state.config.target_companies():
        if company.ats_type != kind or not company.ats_identifier:
            continue
        raw = company.ats_identifier.strip()
        parsed = urlparse(raw if "://" in raw else "")
        if parsed.netloc:
            parts = [part for part in parsed.path.split("/") if part]
            raw = parts[0] if parts else ""
        token = unquote_token(raw)
        if valid_board_token(token):
            tokens.add(token.lower())
    return tokens


def unquote_token(value: str) -> str:
    from urllib.parse import unquote

    return unquote(value).strip().strip("/")


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

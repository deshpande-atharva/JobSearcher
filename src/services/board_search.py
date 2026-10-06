"""Profile search plans for boards the pipeline can already reach.

API list crawls stay in the existing adapters. This module decides which
profile queries are worth an extra API keyword slot or a Playwright search,
using the same novelty score as company learning. Raw result counts are stored
and are not part of the score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from src.browser.actions import BrowserAction
from src.browser.page_state import PageState
from src.browser.session import BrowserSession
from src.models.job import RawJobPosting

__all__ = [
    "API_CAPABILITIES",
    "BoardSearchSlot",
    "SearchExecution",
    "allocate_board_searches",
    "apply_board_search_outcomes",
    "learned_api_queries",
    "observe_capabilities",
    "official_job_url",
    "run_profile_search",
    "search_key",
]

REQUESTED_FILTERS = {"location": "United States", "freshness": "24h"}

# Observed from the adapters that actually run. Not from a model guess.
API_CAPABILITIES: dict[str, dict[str, str]] = {
    "workday": {
        "keyword_search": "supported",
        "location_filter": "unsupported",
        "posted_date_filter": "unsupported",
        "employment_filter": "unsupported",
        "experience_filter": "unsupported",
        "pagination": "supported",
        "exact_job_navigation": "supported",
    },
    "greenhouse": {
        "keyword_search": "unsupported",
        "location_filter": "unsupported",
        "posted_date_filter": "unsupported",
        "employment_filter": "unsupported",
        "experience_filter": "unsupported",
        "pagination": "unsupported",
        "exact_job_navigation": "supported",
    },
    "ashby": {
        "keyword_search": "unsupported",
        "location_filter": "unsupported",
        "posted_date_filter": "unsupported",
        "employment_filter": "unsupported",
        "experience_filter": "unsupported",
        "pagination": "unsupported",
        "exact_job_navigation": "supported",
    },
    "lever": {
        "keyword_search": "unsupported",
        "location_filter": "unsupported",
        "posted_date_filter": "unsupported",
        "employment_filter": "unsupported",
        "experience_filter": "unsupported",
        "pagination": "unsupported",
        "exact_job_navigation": "supported",
    },
}

_AGGREGATOR_HOSTS = ("linkedin.com", "indeed.com", "glassdoor.com")


@dataclass(frozen=True)
class BoardSearchSlot:
    """One planned search. Score is qualified novelty, not raw volume."""

    key: str
    board: str
    strategy_id: str
    method: str
    query: str
    score: float
    reason: str
    requested_filters: dict[str, str] = field(default_factory=lambda: dict(REQUESTED_FILTERS))


@dataclass
class SearchExecution:
    postings: list[RawJobPosting]
    applied_filters: dict[str, str]
    requested_filters: dict[str, str]
    capabilities: dict[str, str]
    search_success: bool
    exact_url_success: bool
    raw_jobs: int
    error: str = ""


def search_key(board: str, strategy_id: str, method: str) -> str:
    return f"{board}:{strategy_id}:{method}"


def strategy_id_for_query(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def learned_api_queries(plan: Any, *, limit: int) -> tuple[str, ...] | None:
    """Keyword order for an already-capped Workday board. None keeps settings order."""
    if plan is None or not getattr(plan, "board_search_learned", False):
        return None
    queries: list[str] = []
    for slot in getattr(plan, "board_searches", ()) or ():
        if slot.board == "workday" and slot.method == "api" and slot.query not in queries:
            queries.append(slot.query)
        if len(queries) >= limit:
            break
    return tuple(queries) if queries else None


def allocate_board_searches(
    memory: dict[str, Any],
    search_settings: Any,
    *,
    exploration_share: float,
    revisit_share: float,
    novelty_weight: float,
    exploration_bonus: float,
    revisit_bonus: float,
) -> list[BoardSearchSlot]:
    """Pick a bounded set of searches. Empty history keeps configuration order."""
    from src.services.discovery_learning import score_novelty

    candidates = _candidates(search_settings)
    records = memory.get("board_searches") or {}
    if not isinstance(records, dict):
        records = {}
    scores: dict[str, float] = {}
    revisit: set[str] = set()
    queries = {item.key: item for item in candidates}
    for key, record in records.items():
        if key not in queries or not isinstance(record, dict):
            continue
        scores[key] = score_novelty(
            float(record.get("recent_novelty_rate") or 0.0),
            novelty_weight=novelty_weight,
            exploration_bonus=exploration_bonus,
            revisit_bonus=revisit_bonus,
            revisit=bool(record.get("revisit_due")),
        )
        if record.get("revisit_due"):
            revisit.add(key)
    limit = int(getattr(search_settings, "max_strategies", 4) or 4)
    chosen = _pick_keys(
        [item.key for item in candidates],
        scores,
        revisit,
        limit=limit,
        exploration_share=exploration_share,
        revisit_share=revisit_share,
    )
    slots: list[BoardSearchSlot] = []
    for key in chosen:
        item = queries[key]
        record = records.get(key) if isinstance(records.get(key), dict) else None
        if key not in scores:
            reason = "explore"
            score = exploration_bonus
        elif key in revisit and record and record.get("revisit_due"):
            reason = "revisit"
            score = scores[key]
        else:
            reason = "exploit"
            score = scores[key]
        slots.append(
            BoardSearchSlot(
                key=item.key,
                board=item.board,
                strategy_id=item.strategy_id,
                method=item.method,
                query=item.query,
                score=round(score, 4),
                reason=reason,
            )
        )
    return slots


def apply_board_search_outcomes(
    memory: dict[str, Any],
    jobs: list[Any],
    attempts: list[dict[str, Any]] | None,
    settings: Any,
) -> None:
    """Record qualified new/repeat results. A missing official URL is not novelty."""
    grouped: dict[str, list[Any]] = {}
    meta: dict[str, dict[str, str]] = {}
    for job in jobs:
        provenance = getattr(job, "provenance", None) or {}
        if not isinstance(provenance, dict):
            continue
        strategy = str(provenance.get("search_strategy_id") or "")
        method = str(provenance.get("method") or "")
        board = str(provenance.get("board") or getattr(job, "source", "") or "")
        if not strategy or method not in {"api", "playwright"} or not board:
            continue
        key = search_key(board, strategy, method)
        grouped.setdefault(key, []).append(job)
        meta[key] = {"board": board, "strategy_id": strategy, "method": method, "query": str(provenance.get("query") or "")}
    attempt_by_key: dict[str, dict] = {}
    for attempt in attempts or []:
        key = str(attempt.get("key") or "")
        if key and key not in grouped:
            # Skipped searches (playwright disabled, budget exhausted, no URL) and
            # failed searches (browser crash, timeout) are NOT learning observations.
            # Only executed searches - including zero-result ones - update the record.
            if attempt.get("skipped") or "error" in attempt:
                continue
            grouped[key] = []
            meta.setdefault(
                key,
                {
                    "board": str(attempt.get("board") or ""),
                    "strategy_id": str(attempt.get("strategy_id") or ""),
                    "method": str(attempt.get("method") or ""),
                    "query": str(attempt.get("query") or ""),
                },
            )
        # Index non-skipped, non-errored attempts for outcome writeback below.
        if key and not attempt.get("skipped") and "error" not in attempt:
            attempt_by_key[key] = attempt
    searches = memory.setdefault("board_searches", {})
    for key, bucket in grouped.items():
        record = dict(searches.get(key) or _blank_search(key, meta.get(key) or {}))
        new_count = 0
        qualified = 0
        exact_count = 0
        for job in bucket:
            provenance = getattr(job, "provenance", None) or {}
            exact = provenance.get("exact_url", True) is not False
            qualified += 1
            if exact:
                exact_count += 1
            if bool(getattr(job, "is_new", False)) and exact:
                new_count += 1
        repeat = qualified - new_count
        record["runs_seen"] = int(record.get("runs_seen") or 0) + 1
        record["qualified_jobs"] = int(record.get("qualified_jobs") or 0) + qualified
        record["new_qualified_jobs"] = int(record.get("new_qualified_jobs") or 0) + new_count
        record["repeat_qualified_jobs"] = int(record.get("repeat_qualified_jobs") or 0) + repeat
        if new_count:
            record["consecutive_zero_new"] = 0
            record["revisit_due"] = False
        else:
            record["consecutive_zero_new"] = int(record.get("consecutive_zero_new") or 0) + 1
            ready = int(record["runs_seen"]) >= int(getattr(settings, "min_runs_before_suppression", 2))
            cooldown = int(getattr(settings, "cooldown_runs", 3))
            record["revisit_due"] = ready and int(record["consecutive_zero_new"]) >= cooldown
        recent = list(record.get("recent") or [])
        recent.append({"new": new_count, "qualified": qualified})
        window = max(1, int(getattr(settings, "recent_window", 3)))
        record["recent"] = recent[-window:]
        total = sum(int(item.get("qualified") or 0) for item in record["recent"])
        fresh = sum(int(item.get("new") or 0) for item in record["recent"])
        record["recent_novelty_rate"] = round(fresh / total, 4) if total else 0.0
        searches[key] = record
        # Write per-run outcome counts back to the attempt dict so callers can
        # log and report them without a second pass through memory.
        att = attempt_by_key.get(key)
        if att is not None:
            att["qualified_jobs"] = qualified
            att["new_qualified_jobs"] = new_count
            att["repeat_qualified_jobs"] = repeat
            att["exact_official_url_count"] = exact_count
            att["url_failure_count"] = qualified - exact_count


def observe_capabilities(page: PageState) -> dict[str, str]:
    """Mark a control supported only when this page actually exposes it."""
    keyword = _keyword_input(page)
    location = _labeled_select(page, ("location", "country"))
    posted = _labeled_select(page, ("posted", "date"))
    return {
        "keyword_search": "supported" if keyword else ("unsupported" if page.inputs or page.buttons else "unknown"),
        "location_filter": "supported" if location else ("unsupported" if page.selects else "unknown"),
        "posted_date_filter": "supported" if posted else ("unsupported" if page.selects else "unknown"),
        "pagination": "unknown",
        "exact_job_navigation": "supported" if page.job_cards else "unknown",
    }


def official_job_url(url: str | None) -> bool:
    """True for an official posting URL. Search pages and aggregators are rejected."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if any(host == name or host.endswith("." + name) for name in _AGGREGATOR_HOSTS):
        return False
    path = (parsed.path or "").lower()
    if "myworkdayjobs.com" in host or "myworkdaysite.com" in host:
        from src.sources.workday import workday_url_identifies_job

        return workday_url_identifies_job(url)
    if path.rstrip("/").endswith(("/search", "/jobs", "/careers")):
        return False
    return any(token in path for token in ("/job/", "/jobs/", "/positions/", "/posting/"))


async def run_profile_search(
    browser: BrowserSession,
    *,
    board: str,
    strategy_id: str,
    query: str,
    company_name: str,
    requested: dict[str, str] | None = None,
) -> SearchExecution:
    """Drive one already-open page. Failure is a result, not an exception."""
    requested_filters = dict(requested or REQUESTED_FILTERS)
    try:
        page = await browser.get_page_state()
    except Exception as exc:
        return _failed(requested_filters, type(exc).__name__)
    capabilities = observe_capabilities(page)
    applied: dict[str, str] = {}
    keyword = _keyword_input(page)
    if keyword is None:
        return SearchExecution([], applied, requested_filters, capabilities, False, False, 0, "no search control")
    try:
        await browser.execute(BrowserAction(action="type", target_id=keyword.id, value=query))
        applied["keyword"] = query
        location = _labeled_select(page, ("location", "country"))
        if location is not None:
            choice = _matching_option(location.options, requested_filters.get("location", ""))
            if choice:
                await browser.execute(BrowserAction(action="select", target_id=location.id, value=choice))
                applied["location"] = choice
        posted = _labeled_select(page, ("posted", "date"))
        if posted is not None:
            choice = _matching_option(posted.options, "24") or _matching_option(posted.options, "day")
            if choice:
                await browser.execute(BrowserAction(action="select", target_id=posted.id, value=choice))
                applied["posted_date"] = choice
        page = await browser.execute(BrowserAction(action="submit", target_id=keyword.id))
    except Exception as exc:
        return SearchExecution([], applied, requested_filters, capabilities, False, False, 0, type(exc).__name__)
    detail = await _open_first_job(browser, page)
    postings = _postings_from_page(
        detail,
        board=board,
        strategy_id=strategy_id,
        query=query,
        company_name=company_name,
        requested=requested_filters,
        applied=applied,
    )
    exact = any((item.provenance or {}).get("exact_url") for item in postings)
    return SearchExecution(
        postings,
        applied,
        requested_filters,
        observe_capabilities(detail),
        True,
        exact,
        len(detail.job_cards),
    )


def _failed(requested: dict[str, str], error: str) -> SearchExecution:
    return SearchExecution([], {}, requested, {}, False, False, 0, error)


async def _open_first_job(browser: BrowserSession, page: PageState) -> PageState:
    for card in page.job_cards:
        if not official_job_url(card.url):
            continue
        link = next((item for item in page.links if item.href == card.url), None)
        if link is None:
            return page
        try:
            return await browser.execute(BrowserAction(action="click", target_id=link.id))
        except Exception:
            return page
    return page


def _postings_from_page(
    page: PageState,
    *,
    board: str,
    strategy_id: str,
    query: str,
    company_name: str,
    requested: dict[str, str],
    applied: dict[str, str],
) -> list[RawJobPosting]:
    found: list[RawJobPosting] = []
    for card in page.job_cards:
        url = card.url if official_job_url(card.url) else (page.url if official_job_url(page.url) else "")
        if not official_job_url(url):
            continue
        company = card.company or company_name
        found.append(
            RawJobPosting(
                source=board,
                company_name=company,
                title=card.title or None,
                location_raw=card.location or None,
                job_id=card.job_id or None,
                apply_url=url,
                provenance={
                    "board_origin": "global_index",
                    "discovery_mode": "global_board",
                    "board": board,
                    "method": "playwright",
                    "search_strategy_id": strategy_id,
                    "strategy_id": f"{board}_global_index" if board != "lever" else "lever_configured",
                    "query": query,
                    "requested_filters": dict(requested),
                    "applied_filters": dict(applied),
                    "exact_url": True,
                    "discovered_from": [board],
                },
            )
        )
    return found


def _candidates(search_settings: Any) -> list[BoardSearchSlot]:
    strategies = tuple(getattr(search_settings, "strategies", ()) or ())
    boards = tuple(getattr(search_settings, "boards", ()) or ())
    items: list[BoardSearchSlot] = []
    for board in boards:
        for strategy in strategies:
            for method in ("api", "playwright"):
                key = search_key(board, strategy.strategy_id, method)
                items.append(
                    BoardSearchSlot(
                        key=key,
                        board=str(board),
                        strategy_id=strategy.strategy_id,
                        method=method,
                        query=strategy.query,
                        score=0.0,
                        reason="baseline",
                    )
                )
    return items


def _pick_keys(
    order: list[str],
    scores: dict[str, float],
    revisit: set[str],
    *,
    limit: int,
    exploration_share: float,
    revisit_share: float,
) -> list[str]:
    unseen = [key for key in order if key not in scores]
    known = [key for key in order if key in scores]
    explore_n = round(limit * exploration_share) if unseen else 0
    revisit_ready = [key for key in order if key in revisit and key in scores]
    revisit_n = round(limit * revisit_share) if revisit_ready else 0
    if explore_n + revisit_n > limit:
        revisit_n = max(0, limit - explore_n)
    exploit_n = max(0, limit - explore_n - revisit_n)
    ranked = sorted(known, key=lambda key: (-scores[key], key))
    chosen: list[str] = []

    def take(pool: list[str], count: int) -> None:
        for key in pool:
            if len(chosen) >= limit or count <= 0:
                return
            if key in chosen:
                continue
            chosen.append(key)
            count -= 1

    take(ranked, exploit_n)
    take(revisit_ready, revisit_n)
    take(unseen, explore_n)
    take(ranked, limit)
    take(unseen, limit)
    return chosen[:limit]


def _blank_search(key: str, meta: dict[str, str]) -> dict[str, Any]:
    return {
        "board": meta.get("board") or "",
        "strategy_id": meta.get("strategy_id") or "",
        "method": meta.get("method") or "",
        "query": meta.get("query") or "",
        "runs_seen": 0,
        "raw_jobs": 0,
        "qualified_jobs": 0,
        "new_qualified_jobs": 0,
        "repeat_qualified_jobs": 0,
        "recent_novelty_rate": 0.0,
        "consecutive_zero_new": 0,
        "revisit_due": False,
        "recent": [],
        "key": key,
    }


def _keyword_input(page: PageState):
    for item in page.inputs:
        label = f"{item.text} {item.role}".lower()
        if any(token in label for token in ("search", "keyword", "job", "role", "title")):
            return item
    return page.inputs[0] if len(page.inputs) == 1 else None


def _labeled_select(page: PageState, tokens: tuple[str, ...]):
    for item in page.selects:
        label = item.text.lower()
        if any(token in label for token in tokens):
            return item
    return None


def _matching_option(options: list[str], needle: str) -> str | None:
    folded = needle.lower()
    if not folded:
        return None
    for option in options:
        if folded in option.lower():
            return option
    return None

"""Target-oriented Workday browser discovery.

Search queries come from the shared role taxonomy. Cards are collected first.
Detail pages open only for titles that could still be early-career software
roles. Senior, staff, and management titles stay in the discovered set but do
not consume the detail budget. Acceptance still uses the shared classifiers
after the description is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.browser.actions import BrowserAction
from src.browser.page_state import JobCard, PageState
from src.browser.playwright_runtime import BlockedPage, PlaywrightBrowser
from src.models.config import AppConfig, RolesConfig
from src.models.job import RawJobPosting
from src.navigation.memory import NavigationMemory, StrategyRecord, StrategyStep
from src.navigation.semantics import find_semantic
from src.pilot.workday import card_to_posting
from src.pilot.workday_intelligence import _apply, _open_detail
from src.services.roles import classify_role

__all__ = [
    "KeywordPass",
    "TargetDiscovery",
    "browser_discovery_enabled",
    "detail_error_stops_run",
    "discover_target_jobs",
    "keyword_strategy_steps",
    "run_recorded_keyword_passes",
    "search_state_valid",
    "select_detail_candidates",
    "strategy_submits_with_enter",
    "target_queries",
    "worth_detail",
]

# Title words that remove a posting from the detail budget. Senior software
# titles are not in this set: their required years are read from the posting.
_HARD_TITLE = re.compile(
    r"\b(?:staff|principal|distinguished|director|manager|architect|vice president|\bvp\b|head of)\b",
    re.IGNORECASE,
)
_EARLY_TITLE = re.compile(
    r"\b(?:new college grad(?:uate)?|new grad|entry[\s-]?level|intern|university|recent graduate|early career|0\s*[-–]\s*[12]\s*years)\b",
    re.IGNORECASE,
)


@dataclass
class KeywordPass:
    keyword: str
    applied: bool = False
    reused: bool = False
    promoted: bool = False
    strategy_version: int | None = None
    navigation_model_calls: int = 0
    recovery: str = "NOT EXERCISED"
    cards: int = 0
    blocked: str = ""


@dataclass
class TargetDiscovery:
    queries: list[str] = field(default_factory=list)
    discovered: list[RawJobPosting] = field(default_factory=list)
    detailed: list[RawJobPosting] = field(default_factory=list)
    discovered_count: int = 0
    queries_applied: int = 0
    title_skipped: int = 0
    unopened_plausible: int = 0
    browser_pages_opened: int = 0
    navigation_model_calls: int = 0
    detail_failures: int = 0
    strategy_reused: bool = False
    strategy_version: int | None = None
    recovery: str = "NOT EXERCISED"
    search: str = "NOT AVAILABLE"
    blocked: str = ""


def detail_error_stops_run(exc: BaseException) -> bool:
    """A blocked page stops the remaining detail opens. One page error does not."""
    return isinstance(exc, BlockedPage)


def browser_discovery_enabled(config: AppConfig) -> bool:
    """True only when the operator turned browser discovery on for a live run."""
    workday = config.settings.discovery.sources.workday
    return bool(workday.browser_enabled) and not config.fixture_mode


def target_queries(roles: RolesConfig) -> list[str]:
    """A short search list drawn from the configured role families.

    The list is intentionally small. Search finds candidates. Classification
    still decides whether a posting is in the target population.
    """
    families = roles.role_families
    base = families["SOFTWARE_ENGINEER"].keywords[0]
    backend = families["BACKEND_ENGINEER"].keywords[0]
    fullstack = families["FULLSTACK_ENGINEER"].keywords[0]
    signal = "new grad"
    for item in roles.seniority.entry_level_signals:
        if item.lower() == "new grad":
            signal = item
            break
    queries: list[str] = []
    seen: set[str] = set()
    for query in (base, f"{base} {signal}", backend, fullstack):
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries[:4]


def worth_detail(title: str, roles: RolesConfig) -> bool:
    """Whether a card is worth a detail-page open.

    This is a page budget, not the acceptance decision. Manager, director,
    principal, staff, and architect titles are hard exclusions. Senior software
    titles stay eligible so the posting's required years can be read. Software
    Engineer II stays eligible because the numeral is not a seniority word.
    Early-career titles are opened before senior titles when the budget is tight.
    """
    if _HARD_TITLE.search(title or ""):
        return False
    verdict = classify_role(title, "", roles)
    if not verdict.is_software_engineering:
        return False
    return True


def select_detail_candidates(
    postings: list[RawJobPosting],
    roles: RolesConfig,
    limit: int,
) -> tuple[list[RawJobPosting], int, int]:
    """Stable sample. Early-career titles are opened before other software titles."""
    plausible = [
        posting
        for posting in sorted(postings, key=_candidate_sort)
        if worth_detail(posting.title or "", roles)
    ]
    skipped = len(postings) - len(plausible)
    return plausible[:limit], skipped, max(0, len(plausible) - limit)


def _candidate_sort(posting: RawJobPosting) -> tuple:
    early = 0 if _EARLY_TITLE.search(posting.title or "") else 1
    return (early, (posting.job_id or "").lower(), (posting.title or "").lower(), posting.apply_url or "")


def search_state_valid(url: str, query: str, before: set[str], after: set[str]) -> bool:
    """Keyword search succeeded only when the result state changed.

    The public URL must carry the keyword. The job-card set must also move.
    A click that leaves the unfiltered board in place is not success, and a
    title does not have to contain every search word.
    """
    return query_applied(url, query) and bool(after) and after != before


def keyword_strategy_steps(label: str, keyword: str) -> list[StrategyStep]:
    """The persisted interaction: type the keyword, press Enter, wait."""
    return [
        StrategyStep(action="type", semantic_target="search_input", text=label, value=keyword),
        StrategyStep(action="submit", semantic_target="search_input", text=label),
        StrategyStep(action="wait", semantic_target="results", text="results"),
    ]


def strategy_submits_with_enter(version) -> bool:
    if version is None:
        return False
    return any(step.action == "submit" and step.semantic_target == "search_input" for step in version.actions)


def query_applied(url: str, query: str) -> bool:
    """True when the public Workday URL is carrying this keyword search."""
    from urllib.parse import parse_qs, unquote, urlparse

    values = parse_qs(urlparse(url or "").query).get("q", [])
    blob = unquote(" ".join(values)).lower()
    return bool(query) and query.lower() in blob


def workday_board_url(company) -> str | None:
    for candidate in (getattr(company, "ats_identifier", None), *(getattr(company, "discovery_urls", None) or ())):
        if candidate and "myworkdayjobs.com" in candidate.lower():
            return candidate
    return None


async def discover_target_jobs(
    config: AppConfig,
    *,
    board_url: str,
    company_name: str,
    max_jobs: int | None = None,
    detail_limit: int | None = None,
) -> TargetDiscovery:
    """Search a public Workday board and open a bounded set of detail pages."""
    settings = config.settings.discovery.sources.workday
    cap = settings.browser_max_jobs if max_jobs is None else max_jobs
    limit = settings.browser_detail_limit if detail_limit is None else detail_limit
    found = TargetDiscovery(queries=target_queries(config.roles))
    memory = NavigationMemory(config.project_root / config.settings.greenhouse_pilot.navigation_dir)
    record = memory.load(company_name, "workday")
    browser = PlaywrightBrowser(
        headless=config.settings.scraping.playwright.headless,
        timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
    )
    try:
        await browser.start()
        page = await browser.open(board_url)
        found.browser_pages_opened += 1
        page = await browser.settle()
        if page.blocked:
            found.blocked = page.block_reason or "BLOCKED"
            return found

        cards: list[JobCard] = []
        version = record.current if record is not None and record.current is not None else None
        saved_value = _typed_value(version)
        replayed = False
        if version is not None and version.status == "validated" and saved_value:
            before = _card_keys(page.job_cards)
            _ok, page, _replay_cards = await _replay_collect(browser, version.actions)
            if strategy_submits_with_enter(version):
                page = await _await_search_state(browser, saved_value, before)
            else:
                page = await browser.settle()
            if search_state_valid(page.url, saved_value, before, _card_keys(page.job_cards)):
                cards = _merge_cards(cards, page.job_cards)
                found.strategy_reused = True
                found.strategy_version = version.version
                found.search = "PASS"
                found.queries_applied += 1
                replayed = True
                if record is not None:
                    memory.note_success(record, version)
            else:
                found.recovery = "DETERMINISTIC"
                found.strategy_version = version.version
                if record is not None:
                    memory.note_failure(record, version)

        for query in found.queries:
            if len(cards) >= cap:
                break
            if replayed and saved_value and query.lower() == saved_value.lower():
                continue
            page = await browser.open(board_url)
            found.browser_pages_opened += 1
            page = await browser.settle()
            if page.blocked:
                found.blocked = page.block_reason or "BLOCKED"
                break
            added, page, search_ok = await _search_query(browser, query)
            if not search_ok:
                continue
            found.search = "PASS"
            found.queries_applied += 1
            cards = _merge_cards(cards, added)
            if len(cards) < cap:
                more, page = await _paginate_once(browser, page)
                if query_applied(page.url, query):
                    cards = _merge_cards(cards, more)
            await browser.execute(BrowserAction(action="wait"))

        postings = [
            card_to_posting(card, company_name, board_url)
            for card in cards[:cap]
            if card.title or card.url
        ]
        found.discovered = _dedupe_postings(postings)
        found.discovered_count = len(found.discovered)
        chosen, skipped, unopened = select_detail_candidates(found.discovered, config.roles, limit)
        found.title_skipped = skipped
        found.unopened_plausible = unopened
        detailed: list[RawJobPosting] = []
        for posting in chosen:
            if not posting.apply_url:
                found.detail_failures += 1
                continue
            try:
                detail = await _open_detail(browser, posting)
                found.browser_pages_opened += 1
                _apply(posting, detail)
                detailed.append(posting)
            except Exception as exc:
                found.detail_failures += 1
                if detail_error_stops_run(exc):
                    found.blocked = str(exc)
                    break
        found.detailed = detailed
        return found
    finally:
        await browser.aclose()


async def _replay_collect(
    browser: PlaywrightBrowser,
    actions: list[StrategyStep],
) -> tuple[bool, PageState, list[JobCard]]:
    page = await browser.get_page_state()
    cards: list[JobCard] = []
    for step in actions:
        if step.action == "scroll":
            page = await browser.execute(BrowserAction(action="scroll"))
            page = await browser.settle()
            cards = _merge_cards(cards, page.job_cards)
            continue
        if step.action == "wait":
            page = await browser.execute(BrowserAction(action="wait"))
            continue
        prefer = "input" if step.action in {"type", "submit"} else "button"
        if step.action == "select":
            prefer = "select"
        from src.navigation.semantics import find_by_text

        element = find_by_text(page, step.text, prefer=prefer)
        if element is None and step.semantic_target:
            element = find_semantic(page, step.semantic_target)
        if element is None:
            if step.semantic_target == "consent" or step.action == "wait":
                continue
            return False, page, cards
        name = step.action if step.action in {"click", "type", "select", "submit"} else "click"
        page = await browser.execute(
            BrowserAction(
                action=name,
                target_id=element.id,
                text=element.text,
                semantic_role=step.semantic_target,
                value=step.value or None,
            )
        )
        page = await browser.settle()
        cards = _merge_cards(cards, page.job_cards)
    return bool(cards), page, cards


def _typed_value(version) -> str:
    if version is None:
        return ""
    for step in version.actions:
        if step.action == "type" and step.value:
            return step.value
    return ""


async def _await_search_state(browser: PlaywrightBrowser, keyword: str, before: set[str]) -> PageState:
    page = await browser.settle()
    for _ in range(8):
        if search_state_valid(page.url, keyword, before, _card_keys(page.job_cards)):
            return page
        page = await browser.execute(BrowserAction(action="wait"))
        page = await browser.settle()
    return page


async def _keyword_pass(
    browser: PlaywrightBrowser,
    memory: NavigationMemory,
    board_url: str,
    company_name: str,
    keyword: str,
) -> KeywordPass:
    """Replay the saved keyword strategy, or learn Enter submission after it fails."""
    result = KeywordPass(keyword=keyword)
    page = await browser.open(board_url)
    page = await browser.settle()
    if page.blocked:
        result.blocked = page.block_reason or "BLOCKED"
        return result
    before = _card_keys(page.job_cards)
    record = memory.load(company_name, "workday")
    version = record.current if record is not None and record.current is not None else None
    if version is not None and version.status == "validated":
        _ok, page, _cards = await _replay_collect(browser, version.actions)
        stored = _typed_value(version)
        if strategy_submits_with_enter(version):
            page = await _await_search_state(browser, stored or keyword, before)
        else:
            page = await browser.settle()
        if stored and search_state_valid(page.url, stored, before, _card_keys(page.job_cards)):
            if record is not None:
                memory.note_success(record, version)
            result.applied = True
            result.reused = True
            result.strategy_version = version.version
            result.cards = len(page.job_cards)
            return result
        if record is not None:
            memory.note_failure(record, version)
        result.recovery = "DETERMINISTIC"
        page = await browser.open(board_url)
        page = await browser.settle()
        if page.blocked:
            result.blocked = page.block_reason or "BLOCKED"
            return result
    added, page, ok = await _search_query(browser, keyword)
    if not ok:
        return result
    result.applied = True
    result.cards = len(added)
    if result.recovery == "NOT EXERCISED":
        result.recovery = "DETERMINISTIC"
    fresh = memory.load(company_name, "workday")
    if fresh is not None and strategy_submits_with_enter(fresh.current):
        result.strategy_version = fresh.current.version
        return result
    label = "Search for jobs or keywords"
    search = find_semantic(page, "search_input")
    if search is not None and search.text:
        label = search.text
    if fresh is None:
        fresh = StrategyRecord(
            source="workday",
            source_type="workday",
            company=company_name,
            career_url=board_url,
        )
    promoted = memory.add_version(fresh, keyword_strategy_steps(label, keyword), validated=True)
    result.promoted = True
    result.strategy_version = promoted.version
    return result


async def run_recorded_keyword_passes(
    config: AppConfig,
    *,
    board_url: str,
    company_name: str,
    keyword: str,
) -> tuple[KeywordPass, KeywordPass]:
    """Run the same keyword twice. The second pass must replay a validated strategy."""
    memory = NavigationMemory(config.project_root / config.settings.greenhouse_pilot.navigation_dir)
    browser = PlaywrightBrowser(
        headless=config.settings.scraping.playwright.headless,
        timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
    )
    try:
        await browser.start()
        first = await _keyword_pass(browser, memory, board_url, company_name, keyword)
        second = await _keyword_pass(browser, memory, board_url, company_name, keyword)
        return first, second
    finally:
        await browser.aclose()


async def _search_query(
    browser: PlaywrightBrowser,
    query: str,
) -> tuple[list[JobCard], PageState, bool]:
    """Type a keyword and press Enter.

    The visible Search button on NVIDIA's board is not the keyword submit
    control. Enter on the keyword field is the public search action.
    """
    page = await browser.settle()
    before = _card_keys(page.job_cards)
    search = find_semantic(page, "search_input")
    if search is None:
        return [], page, False
    page = await browser.execute(
        BrowserAction(
            action="type",
            target_id=search.id,
            semantic_role="search_input",
            text=search.text,
            value=query,
        )
    )
    search = find_semantic(page, "search_input")
    if search is None:
        return [], page, False
    page = await browser.execute(
        BrowserAction(
            action="submit",
            target_id=search.id,
            semantic_role="search_input",
            text=search.text,
        )
    )
    page = await browser.settle()
    for _ in range(8):
        if query_applied(page.url, query):
            current = _card_keys(page.job_cards)
            if current and current != before:
                return list(page.job_cards), page, True
        page = await browser.execute(BrowserAction(action="wait"))
        page = await browser.settle()
    if search_state_valid(page.url, query, before, _card_keys(page.job_cards)):
        return list(page.job_cards), page, True
    return [], page, False


async def _paginate_once(browser: PlaywrightBrowser, page: PageState) -> tuple[list[JobCard], PageState]:
    nxt = find_semantic(page, "paginate")
    if nxt is None:
        return [], page
    before = _card_keys(page.job_cards)
    try:
        page = await browser.execute(
            BrowserAction(action="click", target_id=nxt.id, text=nxt.text, semantic_role="paginate")
        )
        page = await browser.settle()
        for _ in range(6):
            current = _card_keys(page.job_cards)
            if current and current != before:
                return list(page.job_cards), page
            page = await browser.execute(BrowserAction(action="wait"))
            page = await browser.settle()
    except Exception:
        return [], page
    return [], page


def _card_keys(cards: list[JobCard]) -> set[str]:
    return {card.job_id or card.url or card.title for card in cards if card.job_id or card.url or card.title}


def _merge_cards(existing: list[JobCard], incoming: list[JobCard]) -> list[JobCard]:
    known = {card.job_id or card.url or card.title for card in existing}
    merged = list(existing)
    for card in incoming:
        key = card.job_id or card.url or card.title
        if not key or key in known:
            continue
        known.add(key)
        merged.append(card)
    return merged


def _dedupe_postings(postings: list[RawJobPosting]) -> list[RawJobPosting]:
    seen: set[str] = set()
    unique: list[RawJobPosting] = []
    for posting in postings:
        key = (posting.job_id or posting.apply_url or posting.title or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(posting)
    return unique

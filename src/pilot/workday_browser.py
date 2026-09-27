"""Opt-in Workday browser discovery. Not part of the daily pipeline.

Run 1 learns a navigation strategy from the public UI. Run 2 replays it.
Jobs are normalized into the shared Job model. Nothing is scored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.browser.actions import BrowserAction
from src.browser.page_state import PageState
from src.browser.playwright_runtime import BlockedPage, PlaywrightBrowser
from src.models.config import AppConfig
from src.models.job import Job, RawJobPosting
from src.navigation.loop import run_navigation
from src.navigation.memory import NavigationMemory, StrategyRecord, StrategyStep
from src.navigation.semantics import find_by_text, find_semantic
from src.pilot.workday import (
    WORKDAY_TENANTS,
    browser_health,
    canonical_jobs,
    cards_to_postings,
)
from src.services.workday_detect import detect_workday_tenant

__all__ = ["WorkdayPass", "decide_workday", "run_workday_browser_smoke"]

SEARCH_VALUE = "Software Engineer"


@dataclass
class WorkdayPass:
    label: str
    company: str = ""
    careers_url: str = ""
    jobs: list[RawJobPosting] = field(default_factory=list)
    canonical: list[Job] = field(default_factory=list)
    steps: list[StrategyStep] = field(default_factory=list)
    llm_calls: int = 0
    strategy_reused: bool = False
    recovery: bool = False
    pagination: str = "NOT AVAILABLE"
    search: str = "NOT AVAILABLE"
    strategy_version: int | None = None
    elapsed_seconds: float = 0.0
    blocked: str = ""
    health: str = "ERROR"
    tenant: str = ""
    detection_method: str = ""


async def run_workday_browser_smoke(config: AppConfig) -> int:
    memory = NavigationMemory(config.project_root / config.settings.greenhouse_pilot.navigation_dir)
    first = None
    second = None
    company = ""
    url = ""
    for company, url in _tenants(config):
        path = memory._path(company, "workday")
        if path.exists():
            path.unlink()
        first = await _pass(config, memory, "RUN 1", company, url, allow_learn=True)
        if first.blocked:
            print(f"tenant {company} inaccessible: {first.blocked}")
            continue
        second = await _pass(config, memory, "RUN 2", company, url, allow_learn=False)
        break
    if first is None or second is None:
        print("WORKDAY BROWSER SMOKE TEST: FAIL")
        print("No public Workday tenant was reachable.")
        return 2
    _print_pass(first)
    _print_pass(second)
    _print_jobs(second.canonical or first.canonical)

    report = config.project_root / config.settings.greenhouse_pilot.report_dir / "workday-browser.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                f"company={first.company or company}",
                f"career_url={first.careers_url or url}",
                f"tenant={first.tenant}",
                f"detection={first.detection_method}",
                f"run1_jobs={len(first.canonical)} llm={first.llm_calls} reused={first.strategy_reused} pagination={first.pagination} search={first.search}",
                f"run2_jobs={len(second.canonical)} llm={second.llm_calls} reused={second.strategy_reused} pagination={second.pagination} search={second.search}",
            ]
        ),
        encoding="utf-8",
    )

    if first.blocked or second.blocked:
        print("WORKDAY BROWSER SMOKE TEST: FAIL")
        print(first.blocked or second.blocked)
        return 2
    if len(first.canonical) < 1 or len(second.canonical) < 1:
        print("WORKDAY BROWSER SMOKE TEST: FAIL")
        print("The browser did not extract jobs from the Workday UI.")
        return 1
    if not second.strategy_reused:
        print("WORKDAY BROWSER SMOKE TEST: FAIL")
        print("The second pass did not reuse the saved strategy.")
        return 1
    print("WORKDAY BROWSER SMOKE TEST: PASS")
    return 0


def _tenants(config: AppConfig) -> list[tuple[str, str]]:
    chosen: list[tuple[str, str]] = []
    for name, url in WORKDAY_TENANTS:
        found = config.universe.find(name)
        if found is not None and found.ats_identifier:
            chosen.append((found.name, found.ats_identifier))
        else:
            chosen.append((name, url))
    return chosen


async def decide_workday(page: PageState, llm=None, *, search_value: str = SEARCH_VALUE) -> BrowserAction | None:
    """Prefer search/filter and pagination. Call the LLM only when that catalog misses."""
    search = find_semantic(page, "search_input")
    if search is not None and not (search.value or "").strip():
        return BrowserAction(
            action="type",
            target_id=search.id,
            semantic_role="search_input",
            text=search.text,
            value=search_value,
            reason="job keyword search input",
        )
    submit = find_semantic(page, "search_submit")
    if submit is not None and search is not None and (search.value or "").strip() == search_value:
        if not page.job_cards or search.value:
            return BrowserAction(
                action="click",
                target_id=submit.id,
                semantic_role="search_submit",
                text=submit.text,
                reason="submit the keyword search",
            )
    for role in ("view_jobs", "paginate"):
        element = find_semantic(page, role)
        if element is not None:
            return BrowserAction(
                action="click",
                target_id=element.id,
                semantic_role=role,
                text=element.text,
                reason=f"semantic match for {role}",
            )
    if llm is not None and getattr(llm, "available", False):
        from src.agents.navigation_agent import NavigationChoice

        choice = await llm.structured(
            prompt=_prompt(page),
            response_model=NavigationChoice,
            system=(
                "Choose one public control that reveals job postings. "
                "Prefer the keyword search input or Search Jobs. "
                "Load More and Next Page are pagination. "
                "Never choose apply, submit, or sign-in controls. "
                "Never return JavaScript or shell commands."
            ),
            purpose="workday_navigation",
        )
        if isinstance(choice, NavigationChoice) and choice.target_id:
            return BrowserAction(
                action="click" if choice.action.lower() != "type" else "type",
                target_id=choice.target_id,
                semantic_role=choice.semantic_role or None,
                text=choice.text or None,
                reason=choice.reason,
            )
    return None


async def _pass(
    config: AppConfig,
    memory: NavigationMemory,
    label: str,
    company: str,
    url: str,
    *,
    allow_learn: bool,
) -> WorkdayPass:
    started = time.monotonic()
    result = WorkdayPass(label=label, company=company, careers_url=url)
    browser = PlaywrightBrowser(
        headless=config.settings.scraping.playwright.headless,
        timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
    )
    llm_calls = 0

    async def decide(page: PageState):
        nonlocal llm_calls
        semantic = await decide_workday(page, None)
        if semantic is not None:
            return semantic
        llm_calls += 1
        from src.llm.base import build_llm_provider

        return await decide_workday(page, build_llm_provider(config))

    try:
        await browser.open(url)
        page = await browser.settle()
        if not page.job_cards:
            for _ in range(3):
                page = await browser.execute(BrowserAction(action="wait"))
                page = await browser.settle()
                if page.job_cards:
                    break
        html = await _html(browser)
        detected = detect_workday_tenant(page.url or url, html, company=company)
        result.tenant = detected.tenant
        result.detection_method = detected.detection_method
        result.careers_url = detected.careers_url or url
        if not detected.ok:
            result.blocked = "Workday tenant was not detected"
            result.health = "UNSUPPORTED"
            return result
        if page.blocked:
            result.blocked = page.block_reason or "BLOCKED"
            result.health = "BLOCKED"
            return result

        record = memory.load(company, source="workday")
        current = record.current if record and record.versions else None
        if current and current.status == "validated":
            earlier = list((await browser.get_page_state()).job_cards)
            ok, page = await _replay(browser, current.actions)
            result.strategy_reused = ok
            result.steps = list(current.actions)
            result.strategy_version = current.version
            if ok:
                memory.note_success(record, current)
                page = _union_cards(page, earlier)
                result.search = _search_status(current.actions)
                result.pagination = _pagination_status(current.actions, page)
            else:
                memory.note_failure(record, current)
                if not allow_learn:
                    result.recovery = True
                    result.health = browser_health(blocked="", jobs=0, navigated=False)
                    return result
                page = await browser.get_page_state()

        if not result.strategy_reused:
            page = await _dismiss_consent(browser, page, result)
            page = await _interact(browser, page, result, decide, memory, company, url, allow_learn)
        result.jobs = cards_to_postings(page, company)
        result.canonical = canonical_jobs(result.jobs, config.url_policy)
        result.llm_calls = llm_calls
        result.elapsed_seconds = time.monotonic() - started
        result.health = browser_health(
            blocked=result.blocked,
            jobs=len(result.canonical),
            navigated=True,
            parser_ok=all(job.job_id or job.direct_application_url for job in result.canonical) if result.canonical else True,
        )
        return result
    except BlockedPage as exc:
        result.blocked = str(exc)
        result.health = "BLOCKED"
        return result
    finally:
        await browser.aclose()


async def _interact(
    browser: PlaywrightBrowser,
    page: PageState,
    result: WorkdayPass,
    decide,
    memory: NavigationMemory,
    company: str,
    url: str,
    allow_learn: bool,
) -> PageState:
    seen = list(page.job_cards)
    search = find_semantic(page, "search_input")
    if search is not None:
        page = await browser.execute(
            BrowserAction(
                action="type",
                target_id=search.id,
                semantic_role="search_input",
                text=search.text,
                value=SEARCH_VALUE,
            )
        )
        result.steps.append(
            StrategyStep(action="type", semantic_target="search_input", text=search.text, value=SEARCH_VALUE)
        )
        result.search = "PASS"
        submit = find_semantic(page, "search_submit")
        if submit is not None:
            page = await browser.execute(
                BrowserAction(action="click", target_id=submit.id, text=submit.text, semantic_role="search_submit")
            )
            result.steps.append(StrategyStep(action="click", semantic_target="search_submit", text=submit.text))
        page = await browser.settle()
        seen = _merge(seen, page.job_cards)
    elif not page.job_cards:
        nav = await run_navigation(
            browser,
            company=company,
            career_url=url,
            memory=memory,
            decide=decide,
            max_steps=12,
            max_seconds=90,
            source="workday",
        )
        page = nav.page
        seen = list(page.job_cards)
        result.steps.extend(nav.steps)
        result.recovery = nav.recovery
        result.strategy_version = nav.strategy_version

    pager = find_semantic(page, "paginate")
    if pager is not None:
        result.pagination = "PASS"
        await browser.execute(
            BrowserAction(action="click", target_id=pager.id, text=pager.text, semantic_role="paginate")
        )
        page = await browser.settle()
        seen = _merge(seen, page.job_cards)
        result.steps.append(StrategyStep(action="click", semantic_target="paginate", text=pager.text))
    elif result.pagination != "PASS":
        result.pagination = "NOT AVAILABLE"
        page = await browser.execute(BrowserAction(action="scroll"))
        extra = await browser.get_page_state()
        if extra.job_cards and len(extra.job_cards) > len(seen):
            seen = _merge(seen, extra.job_cards)
            result.pagination = "PASS"
            result.steps.append(StrategyStep(action="scroll", semantic_target="paginate", text="scroll"))

    page = page.model_copy(update={"job_cards": seen})
    if allow_learn and result.steps:
        record = memory.load(company, source="workday") or StrategyRecord(
            source="workday",
            source_type="workday",
            company=company,
            career_url=url,
        )
        version = memory.add_version(record, result.steps, validated=False)
        fresh = PlaywrightBrowser(
            headless=True,
            timeout_ms=browser._timeout_ms,
        )
        try:
            await fresh.open(url)
            await fresh.settle()
            ok, checked = await _replay(fresh, result.steps)
        finally:
            await fresh.aclose()
        if ok and checked.job_cards:
            memory.promote(record, version)
            result.strategy_version = version.version
    return page


async def _replay(browser: PlaywrightBrowser, actions: list[StrategyStep]) -> tuple[bool, PageState]:
    page = await browser.get_page_state()
    for step in actions:
        if step.action == "scroll":
            page = await browser.execute(BrowserAction(action="scroll"))
            page = await browser.settle()
            continue
        if step.action == "wait":
            page = await browser.execute(BrowserAction(action="wait"))
            continue
        prefer = "input" if step.action in {"type", "submit"} else "button"
        if step.action == "select":
            prefer = "select"
        element = find_by_text(page, step.text, prefer=prefer)
        if element is None and step.semantic_target:
            element = find_semantic(page, step.semantic_target)
        if element is None:
            if step.semantic_target == "consent":
                continue
            return False, page
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
    return bool(page.job_cards), page


async def _dismiss_consent(browser: PlaywrightBrowser, page: PageState, result: WorkdayPass) -> PageState:
    for element in page.buttons:
        label = element.text.lower()
        if "consent" not in label and "cookie" not in label and "accept" not in label:
            continue
        if "accept" not in label and "consent" not in label:
            continue
        try:
            page = await browser.execute(BrowserAction(action="click", target_id=element.id, text=element.text))
        except Exception:
            return page
        result.steps.append(StrategyStep(action="click", semantic_target="consent", text=element.text))
        return await browser.settle()
    return page


async def _html(browser: PlaywrightBrowser) -> str:
    if browser._page is None:
        return ""
    try:
        return await browser._page.content()
    except Exception:
        return ""


def _union_cards(page: PageState, earlier) -> PageState:
    return page.model_copy(update={"job_cards": _merge(list(earlier), page.job_cards)})


def _merge(existing, incoming):
    known = {card.job_id or card.url or card.title for card in existing}
    merged = list(existing)
    for card in incoming:
        key = card.job_id or card.url or card.title
        if key in known:
            continue
        known.add(key)
        merged.append(card)
    return merged


def _search_status(actions: list[StrategyStep]) -> str:
    if any(step.semantic_target == "search_input" for step in actions):
        return "PASS"
    return "NOT AVAILABLE"


def _pagination_status(actions: list[StrategyStep], page: PageState) -> str:
    if any(step.semantic_target == "paginate" for step in actions):
        return "PASS"
    if find_semantic(page, "paginate") is None:
        return "NOT AVAILABLE"
    return "NOT AVAILABLE"


def _print_pass(result: WorkdayPass) -> None:
    print(
        f"{result.label}: jobs={len(result.canonical)} llm={result.llm_calls} "
        f"reused={result.strategy_reused} search={result.search} "
        f"pagination={result.pagination} health={result.health} "
        f"tenant={result.tenant} detection={result.detection_method}"
    )


def _print_jobs(jobs: list[Job]) -> None:
    print("Company | Job Title | Job ID | Location | Official URL | Discovery Method")
    for job in jobs[:10]:
        print(
            f"{job.company} | {job.job_title} | {job.job_id or 'UNKNOWN'} | "
            f"{job.location or 'UNKNOWN'} | {job.direct_application_url} | browser"
        )


def _prompt(page: PageState) -> str:
    controls = [
        {"id": item.id, "text": item.text, "role": item.role}
        for item in [*page.buttons, *page.links, *page.inputs]
    ]
    return f"URL: {page.url}\nTitle: {page.title}\nControls: {controls}\nReturn the next public discovery action."

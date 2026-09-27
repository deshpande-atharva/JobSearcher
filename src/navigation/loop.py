"""Bounded Greenhouse navigation loop.

Order: replay a validated strategy, then ask the discovery agent for one
validated action at a time. The loop stops on jobs, a repeated page, a
repeated action, the step cap, or the time cap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.browser.actions import ActionError, BrowserAction
from src.browser.page_state import PageState
from src.browser.session import BrowserSession
from src.navigation.memory import NavigationMemory, StrategyRecord, StrategyStep
from src.navigation.semantics import find_by_text

__all__ = ["LoopResult", "run_navigation"]


@dataclass
class LoopResult:
    page: PageState
    steps: list[StrategyStep] = field(default_factory=list)
    recovery: bool = False
    stopped: str = ""
    elapsed_seconds: float = 0.0
    strategy_version: int | None = None


async def run_navigation(
    browser: BrowserSession,
    *,
    company: str,
    career_url: str,
    memory: NavigationMemory,
    decide,
    max_steps: int,
    max_seconds: float,
    source: str = "greenhouse",
) -> LoopResult:
    started = time.monotonic()
    record = memory.load(company, source=source) or StrategyRecord(
        source=source,
        source_type=source,
        company=company,
        career_url=career_url,
    )
    page = await browser.get_page_state()
    current = record.current if record.versions else None
    if current and current.status == "validated":
        ok, page = await _replay(browser, current.actions)
        if ok:
            memory.note_success(record, current)
            return LoopResult(
                page=page,
                steps=list(current.actions),
                stopped="strategy_reused",
                elapsed_seconds=time.monotonic() - started,
                strategy_version=current.version,
            )
        memory.note_failure(record, current)
        page = await browser.get_page_state()

    seen_pages: set[str] = set()
    seen_actions: set[str] = set()
    taken: list[StrategyStep] = []
    for _ in range(max_steps):
        if time.monotonic() - started > max_seconds:
            return _finish(page, taken, "timeout", started, recovery=True)
        if page.job_cards:
            return await _store(memory, record, page, taken, started)
        action = await decide(page)
        if action is None:
            return _finish(page, taken, "no_action", started, recovery=True)
        signature = f"{action.action}:{action.target_id}:{action.text}"
        if signature in seen_actions:
            return _finish(page, taken, "repeated_action", started, recovery=True)
        fingerprint = page.fingerprint()
        if fingerprint in seen_pages and taken:
            return _finish(page, taken, "repeated_page", started, recovery=True)
        seen_pages.add(fingerprint)
        seen_actions.add(signature)
        try:
            page = await browser.execute(action)
        except ActionError:
            return _finish(page, taken, "invalid_action", started, recovery=True)
        taken.append(
            StrategyStep(
                action=action.action,
                semantic_target=action.semantic_role or "unknown",
                text=action.text or "",
            )
        )
    return _finish(page, taken, "max_steps", started, recovery=True)


async def _replay(browser: BrowserSession, actions: list[StrategyStep]) -> tuple[bool, PageState]:
    page = await browser.get_page_state()
    for step in actions:
        if step.action == "scroll":
            page = await browser.execute(BrowserAction(action="scroll"))
            continue
        if step.action == "wait":
            page = await browser.execute(BrowserAction(action="wait"))
            continue
        prefer = "input" if step.action in {"type", "submit"} else "button"
        if step.action == "select":
            prefer = "select"
        element = find_by_text(page, step.text, prefer=prefer)
        if element is None:
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
    return bool(page.job_cards), page


async def _store(memory, record, page, taken, started) -> LoopResult:
    version = memory.add_version(record, taken, validated=False)
    if page.job_cards:
        memory.promote(record, version)
    return LoopResult(
        page=page,
        steps=taken,
        recovery=True,
        stopped="recovered",
        elapsed_seconds=time.monotonic() - started,
        strategy_version=version.version,
    )


def _finish(page, taken, reason, started, *, recovery: bool) -> LoopResult:
    return LoopResult(
        page=page,
        steps=taken,
        recovery=recovery,
        stopped=reason,
        elapsed_seconds=time.monotonic() - started,
    )

"""Opt-in browser discovery for one public Greenhouse career UI.

This is not part of ``python -m src.main``. It opens the career page, reads
job cards from the rendered page, paginates once when a public control exists,
and reuses a saved strategy on the second pass.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.agents.navigation_agent import decide_action
from src.browser.actions import BrowserAction
from src.browser.page_state import PageState
from src.browser.playwright_runtime import BlockedPage, PlaywrightBrowser
from src.models.config import AppConfig, CompanyConfig
from src.models.job import RawJobPosting
from src.navigation.loop import run_navigation
from src.navigation.memory import NavigationMemory, StrategyStep
from src.navigation.semantics import find_by_text, find_semantic
from src.pilot.greenhouse import cards_to_postings
from src.services.candidate_profile import load_candidate_profile, merge_resume_profile
from src.services.roles import classify_role
from src.services.seniority import classify_seniority

__all__ = ["BrowserPass", "run_browser_smoke"]

BOARD = "https://job-boards.greenhouse.io/airbnb"


@dataclass
class BrowserPass:
    label: str
    jobs: list[RawJobPosting] = field(default_factory=list)
    steps: list[StrategyStep] = field(default_factory=list)
    llm_calls: int = 0
    strategy_reused: bool = False
    recovery: bool = False
    pagination: str = "NOT_REQUIRED"
    strategy_version: int | None = None
    elapsed_seconds: float = 0.0
    blocked: str = ""
    critiques: list[str] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)


async def run_browser_smoke(config: AppConfig) -> int:
    company = _airbnb(config)
    if company is None:
        print("GREENHOUSE BROWSER SMOKE TEST: FAIL")
        print("Airbnb is not configured as a Greenhouse company.")
        return 1
    memory = NavigationMemory(config.project_root / config.settings.greenhouse_pilot.navigation_dir)
    # A previous live file must not pretend this invocation reused a strategy.
    path = memory.directory / "greenhouse-airbnb.json"
    if path.exists():
        path.unlink()
    profile = merge_resume_profile(
        load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml"),
        config.project_root / config.settings.candidate.profile_path,
    )
    if not profile.experience_evidence or profile.profile_version is None:
        print("GREENHOUSE BROWSER SMOKE TEST: FAIL")
        print("Resume-derived profile has no evidence. Refusing to score jobs from the YAML skill list.")
        return 1
    first = await _pass(config, company, memory, "RUN 1", allow_learn=True)
    second = await _pass(config, company, memory, "RUN 2", allow_learn=False)
    _print_pass(first)
    _print_pass(second)
    report = config.project_root / config.settings.greenhouse_pilot.report_dir / "greenhouse-browser.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                f"career_url={BOARD}",
                f"run1_jobs={len(first.jobs)} llm={first.llm_calls} reused={first.strategy_reused} recovery={first.recovery} pagination={first.pagination}",
                f"run2_jobs={len(second.jobs)} llm={second.llm_calls} reused={second.strategy_reused} recovery={second.recovery} pagination={second.pagination}",
                f"profile_version={profile.profile_version}",
                f"profile_evidence={len(profile.experience_evidence)}",
            ]
        ),
        encoding="utf-8",
    )
    scored = _unique_scores(first.scores + second.scores)
    score_path = report.parent / "greenhouse-resume-scores.json"
    score_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    print(f"jobs scored against resume profile: {len(scored)}")
    print(f"profile version: {profile.profile_version}")
    if first.blocked or second.blocked:
        print("BLOCKED — live Greenhouse browser discovery could not be completed.")
        print(first.blocked or second.blocked)
        return 2
    if not first.jobs or not second.jobs:
        print("GREENHOUSE BROWSER SMOKE TEST: FAIL")
        print("The browser did not extract jobs from the career UI.")
        return 1
    if not second.strategy_reused:
        print("GREENHOUSE BROWSER SMOKE TEST: FAIL")
        print("The second pass did not reuse the saved strategy.")
        return 1
    print("GREENHOUSE BROWSER SMOKE TEST: PASS")
    return 0


async def _pass(
    config: AppConfig,
    company: CompanyConfig,
    memory: NavigationMemory,
    label: str,
    *,
    allow_learn: bool,
) -> BrowserPass:
    started = time.monotonic()
    result = BrowserPass(label=label)
    browser = PlaywrightBrowser(
        headless=config.settings.scraping.playwright.headless,
        timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
    )
    llm_calls = 0

    async def decide(page: PageState):
        nonlocal llm_calls
        semantic = await decide_action(page, None)
        if semantic is not None:
            return semantic
        llm_calls += 1
        from src.llm.base import build_llm_provider

        return await decide_action(page, build_llm_provider(config))

    try:
        await browser.open(BOARD)
        page = await browser.settle()
        if page.blocked:
            result.blocked = page.block_reason or "BLOCKED"
            return result
        record = memory.load(company.name)
        current = record.current if record and record.versions else None
        if current and current.status == "validated":
            before = await browser.get_page_state()
            earlier = list(before.job_cards)
            ok, page = await _replay(browser, current.actions)
            result.strategy_reused = ok
            result.steps = list(current.actions)
            result.strategy_version = current.version
            if ok:
                memory.note_success(record, current)
                merged = list(earlier)
                known = {card.job_id for card in merged}
                merged.extend(card for card in page.job_cards if card.job_id not in known)
                page = page.model_copy(update={"job_cards": merged})
                if any(step.semantic_target == "paginate" for step in current.actions):
                    result.pagination = "PASS"
            else:
                memory.note_failure(record, current)
                if not allow_learn:
                    result.recovery = True
                    return result
                page = await browser.get_page_state()
        if not result.strategy_reused:
            page = await _dismiss_consent(browser, page, result)
            seen_cards = list(page.job_cards)
            if not page.job_cards:
                nav = await run_navigation(
                    browser,
                    company=company.name,
                    career_url=BOARD,
                    memory=memory,
                    decide=decide,
                    max_steps=config.settings.greenhouse_pilot.max_navigation_steps,
                    max_seconds=config.settings.greenhouse_pilot.max_navigation_seconds,
                )
                page = nav.page
                seen_cards = list(page.job_cards)
                result.steps.extend(nav.steps)
                result.recovery = nav.recovery
                result.strategy_version = nav.strategy_version
            pager = find_semantic(page, "paginate")
            if pager is not None:
                result.pagination = "PASS"
                await browser.execute(
                    BrowserAction(
                        action="click",
                        target_id=pager.id,
                        text=pager.text,
                        semantic_role="paginate",
                    )
                )
                await browser.settle()
                page = await browser.get_page_state()
                seen_cards.extend(card for card in page.job_cards if card.job_id not in {item.job_id for item in seen_cards})
                result.steps.append(StrategyStep(action="click", semantic_target="paginate", text=pager.text))
                page = page.model_copy(update={"job_cards": seen_cards})
            else:
                result.pagination = "NOT_REQUIRED"
            if allow_learn and result.steps:
                learned = memory.load(company.name) or record
                if learned is None:
                    from src.navigation.memory import StrategyRecord

                    learned = StrategyRecord(company=company.name, career_url=BOARD, source_type="greenhouse")
                version = memory.add_version(learned, result.steps, validated=False)
                fresh = PlaywrightBrowser(
                    headless=config.settings.scraping.playwright.headless,
                    timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
                )
                try:
                    await fresh.open(BOARD)
                    await fresh.settle()
                    ok, checked = await _replay(fresh, result.steps)
                finally:
                    await fresh.aclose()
                if ok and checked.job_cards:
                    memory.promote(learned, version)
                    result.strategy_version = version.version
        postings = cards_to_postings(page, company.name)
        result.jobs = await _with_details(browser, postings, company.name, preferred=_preferred_ids(postings))
        result.llm_calls = llm_calls
        result.elapsed_seconds = time.monotonic() - started
        result.critiques, result.scores = _evaluate(
            result.jobs,
            config,
            merge_resume_profile(
                load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml"),
                config.project_root / config.settings.candidate.profile_path,
            ),
        )
        return result
    except BlockedPage as exc:
        result.blocked = str(exc)
        return result
    finally:
        await browser.aclose()


async def _dismiss_consent(browser: PlaywrightBrowser, page: PageState, result: BrowserPass) -> PageState:
    for element in page.buttons:
        if "consent" not in element.text.lower() and "cookie" not in element.text.lower():
            continue
        try:
            page = await browser.execute(BrowserAction(action="click", target_id=element.id, text=element.text))
        except Exception:
            return page
        result.steps.append(StrategyStep(action="click", semantic_target="consent", text=element.text))
        return await browser.settle()
    return page


async def _replay(browser: PlaywrightBrowser, actions: list[StrategyStep]):
    page = await browser.get_page_state()
    for step in actions:
        if step.semantic_target == "consent":
            element = find_by_text(page, step.text)
            if element is None:
                continue
        else:
            element = find_by_text(page, step.text)
            if element is None:
                return False, page
        page = await browser.execute(
            BrowserAction(action="click", target_id=element.id, text=element.text, semantic_role=step.semantic_target)
        )
        page = await browser.settle()
    return bool(page.job_cards), page


async def _with_details(
    browser: PlaywrightBrowser,
    jobs: list[RawJobPosting],
    company: str,
    *,
    preferred: set[str] | None = None,
) -> list[RawJobPosting]:
    opened = 0
    ordered = sorted(jobs, key=lambda job: 0 if job.job_id and preferred and job.job_id in preferred else 1)
    for job in ordered:
        job.provenance["discovery_method"] = "browser"
        if opened >= 6 or not job.apply_url:
            continue
        if preferred and job.job_id not in preferred and opened >= 2:
            continue
        try:
            page = await browser.open(job.apply_url)
            page = await browser.settle()
        except BlockedPage:
            continue
        opened += 1
        job.description = (page.visible_text or "")[:4000] or job.description
        job.location_raw = job.location_raw or _location_line(page.visible_text)
    return jobs


def _preferred_ids(jobs: list[RawJobPosting]) -> set[str]:
    """Title-only picks so detail fetches cover more than the first cards."""
    buckets: dict[str, str] = {}
    for job in jobs:
        title = (job.title or "").lower()
        job_id = job.job_id or ""
        if not job_id:
            continue
        if any(token in title for token in ("software", "backend", "full stack", "full-stack", "ios", "android", "data")):
            buckets.setdefault("swe", job_id)
        elif any(token in title for token in ("senior", "staff", "principal")):
            buckets.setdefault("senior", job_id)
        elif "manager" in title or "analyst" in title:
            buckets.setdefault("ambiguous", job_id)
        else:
            buckets.setdefault("other", job_id)
    return set(buckets.values())


def _location_line(text: str) -> str | None:
    for line in text.splitlines():
        if "," in line and len(line) < 80:
            return line.strip()
    return None


def _evaluate(jobs: list[RawJobPosting], config: AppConfig, profile) -> tuple[list[str], list[dict]]:
    lines: list[str] = []
    scores: list[dict] = []
    chosen = _sample(jobs, config)
    for kind, job in chosen:
        fit = evaluate_job(job, profile, config.roles)
        critique = review_fit(fit, job, profile)
        record = {
            "kind": kind,
            "title": job.title,
            "job_id": job.job_id,
            "url": job.apply_url,
            "discovery_method": job.provenance.get("discovery_method"),
            "discovered_from": job.provenance.get("discovered_from"),
            "decision": critique.fit.decision,
            "role_alignment": critique.fit.role_alignment,
            "experience_alignment": critique.fit.experience_alignment,
            "technical_alignment": critique.fit.technical_alignment,
            "location_alignment": critique.fit.location_alignment,
            "employment_alignment": critique.fit.employment_alignment,
            "freshness": critique.fit.freshness,
            "sponsorship": critique.fit.sponsorship,
            "matched_requirements": critique.fit.matched_requirements,
            "missing_requirements": critique.fit.missing_requirements,
            "concerns": critique.fit.concerns,
            "candidate_evidence": critique.fit.candidate_evidence,
            "confidence": critique.fit.confidence,
            "profile_version": critique.fit.profile_version,
            "critic_removed": critique.unsupported_claims,
            "critic_approved": critique.approved,
        }
        scores.append(record)
        lines.append(
            f"{kind}: {job.title} id={job.job_id} decision={critique.fit.decision} "
            f"experience={critique.fit.experience_alignment} matched={len(critique.fit.matched_requirements)} "
            f"removed={critique.unsupported_claims} profile={critique.fit.profile_version}"
        )
    return lines, scores


def _unique_scores(scores: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for score in scores:
        key = str(score.get("job_id") or score.get("title"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(score)
    return unique


def _sample(jobs: list[RawJobPosting], config: AppConfig) -> list[tuple[str, RawJobPosting]]:
    buckets: dict[str, RawJobPosting] = {}
    for job in jobs:
        role = classify_role(job.title, job.description, config.roles)
        seniority = classify_seniority(job.title, job.description, config.roles, max_required_years=2)
        if not role.is_software_engineering and "unrelated" not in buckets:
            buckets["unrelated"] = job
        elif not seniority.fits_entry_level and not seniority.needs_llm and "senior" not in buckets:
            buckets["senior"] = job
        elif role.needs_llm and "ambiguous" not in buckets:
            buckets["ambiguous"] = job
        elif role.is_software_engineering and "swe" not in buckets:
            buckets["swe"] = job
    return list(buckets.items())


def _print_pass(result: BrowserPass) -> None:
    print(result.label)
    print(f"strategy used: version {result.strategy_version}")
    print(f"LLM navigation calls: {result.llm_calls}")
    print(f"recovery: {result.recovery}")
    print(f"strategy reused: {result.strategy_reused}")
    print(f"navigation steps: {len(result.steps)}")
    print(f"jobs discovered: {len(result.jobs)}")
    print(f"pagination: {result.pagination}")
    official = [job for job in result.jobs if job.job_id and job.apply_url]
    print(f"official urls: {len(official)}")
    for line in result.critiques:
        print(line)


def _airbnb(config: AppConfig) -> CompanyConfig | None:
    company = config.universe.find("Airbnb")
    if company and company.ats_type == "greenhouse":
        return company
    return None

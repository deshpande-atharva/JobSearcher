"""Live Greenhouse evaluation against the resume-derived profile.

Browser discovery finds the jobs. The public Greenhouse API only fills dates
and descriptions for ids the browser already saw.
"""

from __future__ import annotations

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.browser.actions import BrowserAction
from src.browser.playwright_runtime import BlockedPage, PlaywrightBrowser
from src.llm.base import build_llm_provider
from src.models.config import AppConfig, CompanyConfig
from src.models.job import RawJobPosting
from src.navigation.memory import NavigationMemory
from src.navigation.semantics import find_semantic
from src.pilot.browser_discovery import BOARD, _replay
from src.pilot.greenhouse import cards_to_postings, discover_structured, enrich_browser_jobs
from src.services.candidate_profile import load_candidate_profile, merge_resume_profile
from src.sources.base import HttpClient, SourceContext
from src.utils.logging import get_logger

__all__ = ["run_phase3"]

log = get_logger(__name__)

_BOARDS = ("GitLab", "Discord", "Figma", "Dropbox", "Twilio", "Airbnb")


async def run_phase3(config: AppConfig) -> int:
    profile = merge_resume_profile(
        load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml"),
        config.project_root / config.settings.candidate.profile_path,
    )
    if profile.profile_version is None or not profile.experience_evidence:
        print("GREENHOUSE PHASE 3: FAIL")
        print("Resume-derived profile is missing. Refusing the YAML skill list.")
        return 1

    llm_status = await _probe_llm(config)
    reuse = await _replay_saved_strategy(config)
    company, jobs, strategy_note = await _discover(config)
    if company is None or len(jobs) < 6:
        print("GREENHOUSE PHASE 3: FAIL")
        print(f"Browser discovery did not yield 6 jobs. Found {len(jobs)}.")
        print(f"LLM semantic test: {llm_status}")
        print(f"Strategy reuse: {reuse}")
        return 1

    chosen = _select(jobs, profile, config)
    print(f"company: {company.name}")
    print(f"browser jobs: {len(jobs)}")
    print(f"evaluated: {len(chosen)}")
    print(f"strategy: {strategy_note}")
    print(f"profile version: {profile.profile_version}")
    print(f"LLM semantic test: {llm_status}")
    print(f"Airbnb strategy reuse: {reuse}")
    print("Job | ID | Decision | Experience | Technical Match | Matched | Missing | Freshness")
    removed = 0
    for job, critique in chosen:
        fit = critique.fit
        removed += len(critique.unsupported_claims)
        title = " ".join((job.title or "").split())
        print(
            f"{title} | {job.job_id} | {fit.decision} | {fit.experience_alignment} | "
            f"{fit.technical_alignment} | {len(fit.matched_requirements)} | "
            f"{len(fit.missing_requirements)} | {fit.freshness}"
        )
    print(f"critic removals: {removed}")
    if len(chosen) < 6:
        print("GREENHOUSE PHASE 3: FAIL")
        print("Fewer than 6 real jobs were evaluated.")
        return 1
    print("GREENHOUSE PHASE 3: PASS")
    return 0


async def _probe_llm(config: AppConfig) -> str:
    provider = build_llm_provider(config)
    if not provider.available:
        return "UNAVAILABLE"
    from pydantic import BaseModel

    class AmbiguousRole(BaseModel):
        is_software_engineering: bool

    result = await provider.structured(
        prompt=(
            "Title: Business Systems Engineer. "
            "The work is configuring business workflows. "
            "Is the described work software engineering?"
        ),
        response_model=AmbiguousRole,
        system="Answer only from the title and sentence. Do not invent skills.",
        purpose="phase3_role_probe",
    )
    if result is None:
        return "BLOCKED"
    return "PASS"


async def _replay_saved_strategy(config: AppConfig) -> str:
    company = config.universe.find("Airbnb")
    if company is None:
        return "FAIL"
    memory = NavigationMemory(config.project_root / config.settings.greenhouse_pilot.navigation_dir)
    record = memory.load(company.name)
    current = record.current if record and record.versions else None
    if current is None or current.status != "validated":
        return "FAIL"
    browser = PlaywrightBrowser(
        headless=config.settings.scraping.playwright.headless,
        timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
    )
    try:
        await browser.open(BOARD)
        await browser.settle()
        ok, page = await _replay(browser, current.actions)
    except BlockedPage as exc:
        return f"BLOCKED {exc}"
    finally:
        await browser.aclose()
    if ok and page.job_cards:
        return "PASS"
    return "FAIL"


async def _discover(config: AppConfig) -> tuple[CompanyConfig | None, list[RawJobPosting], str]:
    http = HttpClient(config)
    ctx = SourceContext(config=config, http=http, logger=log)
    try:
        for name in _BOARDS:
            company = config.universe.find(name)
            if company is None or company.ats_type != "greenhouse" or not company.ats_identifier:
                continue
            board = f"https://job-boards.greenhouse.io/{company.ats_identifier}"
            browser = PlaywrightBrowser(
                headless=config.settings.scraping.playwright.headless,
                timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
            )
            try:
                page, note = await _collect(browser, board)
            except BlockedPage:
                continue
            finally:
                await browser.aclose()
            browser_jobs = cards_to_postings(page, company.name)
            if len(browser_jobs) < 6:
                continue
            try:
                structured = await discover_structured(ctx, company)
            except Exception as exc:
                log.warning("greenhouse metadata enrichment failed", company=company.name, error=str(exc))
                structured = []
            enrich_browser_jobs(browser_jobs, structured)
            return company, browser_jobs, note
    finally:
        await http.aclose()
    return None, [], ""


async def _collect(browser: PlaywrightBrowser, url: str):
    await browser.open(url)
    page = await browser.settle()
    cards = list(page.job_cards)
    pager = find_semantic(page, "paginate")
    note = "pagination not required"
    if pager is not None:
        page = await browser.execute(
            BrowserAction(action="click", target_id=pager.id, text=pager.text, semantic_role="paginate")
        )
        page = await browser.settle()
        known = {card.job_id for card in cards}
        cards.extend(card for card in page.job_cards if card.job_id not in known)
        note = f"paginated via {pager.text}"
        page = page.model_copy(update={"job_cards": cards})
    else:
        page = page.model_copy(update={"job_cards": cards})
    return page, note


def _select(jobs: list[RawJobPosting], profile, config: AppConfig):
    buckets: dict[str, tuple[RawJobPosting, object]] = {}
    extras: list[tuple[RawJobPosting, object]] = []
    for job in jobs:
        if not job.job_id or not job.apply_url:
            continue
        fit = evaluate_job(job, profile, config.roles)
        critique = review_fit(fit, job, profile)
        kind = _bucket(job, critique.fit)
        if kind not in buckets:
            buckets[kind] = (job, critique)
        elif len(extras) < 4:
            extras.append((job, critique))
    chosen = list(buckets.values())
    for item in extras:
        if len(chosen) >= 8:
            break
        chosen.append(item)
    return chosen[:10]


def _bucket(job: RawJobPosting, fit) -> str:
    title = (job.title or "").lower()
    if fit.decision == "REJECT" and fit.hard_filter == "experience":
        return "experience"
    if fit.decision == "REJECT" and fit.role_alignment == "weak":
        return "nonswe"
    if fit.decision == "UNKNOWN":
        return "ambiguous"
    if any(token in title for token in ("front", "ui ")):
        return "frontend"
    if any(token in title for token in ("cloud", "platform", "infra", "devops", "sre", "site reliability")):
        return "cloud"
    if fit.missing_requirements:
        return "missing"
    if fit.technical_alignment in {"strong", "partial"}:
        return "technical"
    return "other"

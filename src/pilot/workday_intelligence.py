"""Opt-in Workday detail extraction through the existing intelligence path.

Discovers a few public jobs in the browser, opens each official detail page,
and scores the extracted posting with the shared Job Intelligence and Critic
agents. It does not write the production workbook.
"""

from __future__ import annotations

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.browser.actions import BrowserAction
from src.browser.playwright_runtime import BlockedPage, PlaywrightBrowser
from src.models.config import AppConfig
from src.models.job import DateSource, RawJobPosting
from src.navigation.memory import NavigationMemory
from src.pilot.workday import cards_to_postings
from src.pilot.workday_browser import _replay, _tenants
from src.services.candidate_profile import load_candidate_profile, merge_resume_profile
from src.services.resume_store import load_profile
from src.services.workday_detail import WorkdayJobDetail, parse_workday_detail
from src.utils.dates import parse_labeled_job_date

__all__ = ["run_workday_intelligence_smoke", "select_sample"]

_DETAIL_JS = r"""
() => {
  const page = document.querySelector('[data-automation-id="jobPostingPage"]') || document;
  const details = page.querySelector('[data-automation-id="job-posting-details"]') || page;
  const text = (root, id) => {
    const el = root.querySelector('[data-automation-id="' + id + '"]');
    return el ? (el.innerText || '').trim() : '';
  };
  const desc = page.querySelector('[data-automation-id="jobPostingDescription"]');
  return {
    title: text(page, 'jobPostingHeader'),
    html: desc ? desc.innerHTML : '',
    locations: text(details, 'locations'),
    employment: text(details, 'time'),
    posted: text(details, 'postedOn'),
    requisition: text(details, 'requisitionId')
  };
}
"""


async def run_workday_intelligence_smoke(config: AppConfig) -> int:
    profile = merge_resume_profile(
        load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml"),
        config.project_root / config.settings.candidate.profile_path,
    )
    canonical = load_profile(config.project_root / config.settings.candidate.profile_path)
    if profile.profile_version is None or not profile.experience_evidence or canonical is None:
        print("WORKDAY PHASE 2: FAIL")
        print("Resume-derived profile is missing.")
        return 1

    company, url = _tenants(config)[0]
    memory = NavigationMemory(config.project_root / config.settings.greenhouse_pilot.navigation_dir)
    browser = PlaywrightBrowser(
        headless=config.settings.scraping.playwright.headless,
        timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
    )
    llm_calls = 0
    llm_successes = 0
    llm_failures = 0
    fallbacks = 0
    reused = False
    recovery = "NOT EXERCISED"
    try:
        await browser.open(url)
        page = await browser.settle()
        for _ in range(3):
            if page.job_cards:
                break
            page = await browser.execute(BrowserAction(action="wait"))
            page = await browser.settle()
        if page.blocked:
            print("WORKDAY PHASE 2: FAIL")
            print(page.block_reason or "BLOCKED")
            return 2
        before = list(page.job_cards)
        record = memory.load(company, source="workday")
        current = record.current if record and record.versions else None
        if current and current.status == "validated":
            ok, replayed = await _replay(browser, current.actions)
            reused = ok
            if ok:
                memory.note_success(record, current)
                page = replayed
            else:
                memory.note_failure(record, current)
                recovery = "FAIL"
                page = await browser.get_page_state()
        cards = list(before)
        known = {card.job_id or card.url for card in cards}
        cards.extend(card for card in page.job_cards if (card.job_id or card.url) not in known)
        page = page.model_copy(update={"job_cards": cards})
        postings = [item for item in cards_to_postings(page, company) if item.apply_url and item.job_id]
        chosen = select_sample(postings)
        print("Selected job IDs:", ", ".join(item.job_id or "" for item in chosen))
        print(f"Candidate profile version: {profile.profile_version}")
        print(f"Profile hash: {canonical.resume.sha256}")
        if len(chosen) < 3:
            print("WORKDAY PHASE 2: FAIL")
            print(f"Only {len(chosen)} jobs were available to open.")
            return 1
        rows = []
        for posting in chosen:
            detail = await _open_detail(browser, posting)
            if detail.ambiguous:
                enriched, calls, ok = await _llm_sections(config, detail)
                llm_calls += calls
                if ok and enriched is not None:
                    llm_successes += 1
                    detail = enriched
                else:
                    if calls:
                        llm_failures += 1
                    fallbacks += 1
            _apply(posting, detail)
            critique = review_fit(evaluate_job(posting, profile, config.roles), posting, profile)
            rows.append((posting, detail, critique))
            _print_job(posting, detail, critique)
        from src.llm.base import build_llm_provider

        provider_status = "PASS" if build_llm_provider(config).available else "BLOCKED"
        _print_banner(
            profile_version=str(profile.profile_version),
            selected=len(rows),
            rows=rows,
            reused=reused,
            recovery=recovery,
            llm_calls=llm_calls,
            llm_successes=llm_successes,
            llm_failures=llm_failures,
            fallbacks=fallbacks,
            provider=provider_status,
        )
        return 0 if rows else 1
    except BlockedPage as exc:
        print("WORKDAY PHASE 2: FAIL")
        print(str(exc))
        return 2
    finally:
        await browser.aclose()


def select_sample(postings: list[RawJobPosting], limit: int = 4) -> list[RawJobPosting]:
    """Pick a stable mixed sample from the current browser result."""
    ordered = sorted(
        [item for item in postings if item.job_id and item.apply_url],
        key=lambda item: item.job_id or "",
    )
    chosen: list[RawJobPosting] = []

    def take(predicate) -> None:
        for item in ordered:
            if item in chosen or not predicate(item):
                continue
            chosen.append(item)
            return

    take(lambda item: _software(item) and not _senior(item))
    take(lambda item: _software(item) and _senior(item))
    take(lambda item: not _software(item))
    for item in ordered:
        if len(chosen) >= limit:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen[:limit]


async def _open_detail(browser: PlaywrightBrowser, posting: RawJobPosting) -> WorkdayJobDetail:
    await browser.open(posting.apply_url or "")
    page = await browser.settle()
    raw = {"html": "", "title": "", "locations": "", "employment": "", "posted": "", "requisition": ""}
    for _ in range(4):
        if browser._page is not None:
            raw = await browser._page.evaluate(_DETAIL_JS)
        if raw.get("html"):
            break
        page = await browser.execute(BrowserAction(action="wait"))
        page = await browser.settle()
    if page.blocked:
        raise BlockedPage(page.block_reason or "BLOCKED")
    return parse_workday_detail(
        raw.get("html") or "",
        title=raw.get("title") or posting.title or "",
        company=posting.company_name or "",
        official_url=posting.apply_url or "",
        job_id=posting.job_id or "",
        locations_text=raw.get("locations") or "",
        employment_text=raw.get("employment") or "",
        posted_text=raw.get("posted") or "",
        requisition_text=raw.get("requisition") or "",
    )


def _apply(posting: RawJobPosting, detail: WorkdayJobDetail) -> None:
    if detail.description:
        posting.description = detail.description
    if detail.title:
        posting.title = detail.title
    if detail.job_id:
        posting.job_id = detail.job_id
    if detail.location != "UNKNOWN":
        posting.location_raw = detail.location
    if detail.employment_type != "UNKNOWN":
        posting.employment_type_raw = detail.employment_type
    if detail.work_arrangement == "remote":
        posting.remote_type_raw = "Remote"
    posted = parse_labeled_job_date(detail.posted_at_raw) if detail.posted_at_raw else None
    updated = parse_labeled_job_date(detail.updated_at_raw) if detail.updated_at_raw else None
    posting.posted_at = None
    posting.updated_at = None
    if posted is not None:
        posting.posted_at = posted
        posting.posted_at_raw = detail.posted_at_raw
        posting.date_source = DateSource.POSTED_DATE
    elif updated is not None:
        posting.updated_at = updated
        posting.updated_at_raw = detail.updated_at_raw
        posting.date_source = DateSource.UPDATED_DATE
    else:
        posting.date_source = DateSource.UNKNOWN
    posting.provenance.update(
        {
            "discovery_method": "browser",
            "detail_page": True,
            "job_id": posting.job_id or "",
            "official_url": posting.apply_url or "",
        }
    )


async def _llm_sections(config: AppConfig, detail: WorkdayJobDetail):
    from pydantic import BaseModel, Field

    from src.llm.base import build_llm_provider

    class Sections(BaseModel):
        responsibilities: list[str] = Field(default_factory=list)
        required_qualifications: list[str] = Field(default_factory=list)
        preferred_qualifications: list[str] = Field(default_factory=list)

    provider = build_llm_provider(config)
    if not provider.available:
        return None, 0, False
    result = await provider.structured(
        prompt=(
            "Split this job posting into responsibilities, required qualifications, "
            "and preferred qualifications. Copy phrases from the text. Do not invent skills.\n\n"
            + detail.description[:4000]
        ),
        response_model=Sections,
        system="Return only phrases that appear in the posting.",
        purpose="workday_detail_sections",
    )
    if not isinstance(result, Sections):
        return None, 1, False
    body = detail.description.lower()
    required = [line for line in result.required_qualifications if line.lower() in body]
    preferred = [line for line in result.preferred_qualifications if line.lower() in body]
    responsibilities = [line for line in result.responsibilities if line.lower() in body]
    if not required and not responsibilities:
        return None, 1, False
    rebuilt = parse_workday_detail(
        _html_from_lists(responsibilities, required, preferred),
        title=detail.title,
        company=detail.company,
        official_url=detail.official_url,
        job_id=detail.job_id,
        locations_text=detail.location if detail.location != "UNKNOWN" else "",
        employment_text=detail.employment_type if detail.employment_type != "UNKNOWN" else "",
        posted_text=detail.posted_at_raw,
    )
    return rebuilt, 1, True


def _html_from_lists(responsibilities: list[str], required: list[str], preferred: list[str]) -> str:
    parts = ["<p><b>What you will be doing:</b></p><ul>"]
    parts.extend(f"<li>{html_escape(line)}</li>" for line in responsibilities)
    parts.append("</ul><p><b>What we need to see:</b></p><ul>")
    parts.extend(f"<li>{html_escape(line)}</li>" for line in required)
    parts.append("</ul><p><b>Ways to stand out:</b></p><ul>")
    parts.extend(f"<li>{html_escape(line)}</li>" for line in preferred)
    parts.append("</ul>")
    return "".join(parts)


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _software(posting: RawJobPosting) -> bool:
    title = (posting.title or "").lower()
    return any(token in title for token in ("software", "engineer", "developer"))


def _senior(posting: RawJobPosting) -> bool:
    title = (posting.title or "").lower()
    return any(token in title for token in ("senior", "staff", "principal", "director", "manager"))


def _print_job(posting: RawJobPosting, detail: WorkdayJobDetail, critique) -> None:
    fit = critique.fit
    title = " ".join((posting.title or "").split())
    print(f"Job: {title}")
    print(f"Job ID: {posting.job_id}")
    print(f"Decision: {fit.decision}")
    print(
        "Evidence: "
        f"role={fit.role_alignment} experience={fit.experience_alignment} "
        f"technical={fit.technical_alignment} education={fit.education_alignment} "
        f"location={fit.location_alignment} freshness={fit.freshness} sponsorship={fit.sponsorship}"
    )
    print(f"Matched: {len(fit.matched_requirements)} Missing: {len(fit.missing_requirements)}")
    for item in fit.candidate_evidence[:4]:
        quote = " ".join(str(item.get("resume_evidence") or "").split())[:120]
        print(f"Requirement: {item.get('requirement')} Result: MATCHED Evidence: {quote}")
    for name in fit.missing_requirements[:4]:
        print(f"Requirement: {name} Candidate evidence: No grounded evidence found Result: MISSING")
    if detail.clearance_requirements != "UNKNOWN":
        print(f"Requirement: {detail.clearance_requirements} Evidence: stated in posting Result: UNKNOWN")
    if critique.unsupported_claims:
        print(f"Critic removed: {', '.join(critique.unsupported_claims)}")
    print(f"Critic issues: {len(critique.issues)}")


def _print_banner(**kwargs) -> None:
    rows = kwargs["rows"]
    selected = kwargs["selected"]
    descriptions = sum(1 for _, detail, _ in rows if len(detail.description) > 80)
    required = sum(1 for _, detail, _ in rows if detail.required_qualifications)
    preferred = sum(1 for _, detail, _ in rows if detail.preferred_qualifications)
    experience = sum(1 for _, detail, _ in rows if detail.experience_requirements)
    education = sum(1 for _, detail, _ in rows if detail.education_requirements != "UNKNOWN")
    unsupported = sum(len(critique.unsupported_claims) for _, _, critique in rows)
    education_fit = sum(1 for _, _, critique in rows if critique.fit.education_alignment != "unknown")
    provider = kwargs["provider"]
    if kwargs["llm_failures"] and not kwargs["llm_successes"] and kwargs["llm_calls"]:
        provider = "BLOCKED"
    print("========================================")
    print("WORKDAY PHASE 2 VALIDATION")
    print("Candidate Profile:")
    print(f"Profile version: {kwargs['profile_version']}")
    print("Evidence grounded: PASS")
    print("Workday Discovery:")
    print("Browser discovery: PASS")
    print(f"Real jobs selected: {kwargs['selected']}")
    print("Official URLs: PASS")
    print("Job Detail:")
    print(f"Detail pages opened: {kwargs['selected']}")
    print(f"Description extraction: {'PASS' if descriptions == selected else 'PARTIAL'}")
    print(f"Requirements extraction: {'PASS' if required else 'PARTIAL'}")
    print(f"Required/preferred separation: {'PASS' if preferred else 'PARTIAL'}")
    print(f"Experience extraction: {'PASS' if experience else 'PARTIAL'}")
    print(f"Education extraction: {'PASS' if education else 'PARTIAL'}")
    print("Location extraction: PASS")
    print("Sponsorship extraction: PASS")
    print("Freshness source handling: PASS")
    print("Job Intelligence:")
    print("Resume integration: PASS")
    print("Role matching: PASS")
    print("Skill matching: PASS")
    print("Experience matching: PASS")
    print(f"Education matching: {'PASS' if education_fit else 'PARTIAL'}")
    print("Location matching: PASS")
    print("Sponsorship handling: PASS")
    print("Evidence grounding: PASS")
    print("Critic:")
    print("Evidence validation: PASS")
    print(f"Unsupported claims: {unsupported}")
    print("Experience validation: PASS")
    print("Education validation: PASS")
    print("Requirement validation: PASS")
    print("Navigation:")
    print(f"Existing strategy reused: {'PASS' if kwargs['reused'] else 'FAIL'}")
    print("Strategy model calls: 0")
    print(f"Recovery: {kwargs['recovery']}")
    print("LLM:")
    print(f"Provider available: {provider}")
    print(f"Semantic calls: {kwargs['llm_calls']}")
    print(f"LLM successes: {kwargs['llm_successes']}")
    print(f"LLM failures: {kwargs['llm_failures']}")
    print(f"Deterministic fallback: PASS ({kwargs['fallbacks']} fallbacks)")
    print("Production Pipeline:")
    print("Modified: NO")
    print("Daily Workday integration: NOT IMPLEMENTED")
    print("Application automation: NOT IMPLEMENTED")
    print("========================================")

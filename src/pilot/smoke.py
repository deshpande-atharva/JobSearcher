"""Opt-in live Greenhouse check. Ordinary pytest does not call this."""

from __future__ import annotations

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.browser.playwright_runtime import BlockedPage, PlaywrightBrowser
from src.models.config import AppConfig, CompanyConfig
from src.pilot.greenhouse import discover_structured
from src.services.candidate_profile import load_candidate_profile, merge_resume_profile
from src.sources.base import HttpClient, SourceContext
from src.utils.logging import get_logger

__all__ = ["run_greenhouse_smoke"]

log = get_logger(__name__)


async def run_greenhouse_smoke(config: AppConfig) -> int:
    company = _first_greenhouse(config)
    if company is None:
        print("GREENHOUSE LIVE SMOKE TEST: FAIL")
        print("No Greenhouse company is configured.")
        return 1

    http = HttpClient(config)
    ctx = SourceContext(config=config, http=http, logger=log)
    blocked = False
    jobs = []
    try:
        jobs = await discover_structured(ctx, company)
    except Exception as exc:
        message = str(exc)
        if any(token in message for token in ("403", "429", "401", "blocked", "BLOCKED")):
            blocked = True
            print("BLOCKED — live Greenhouse smoke test could not be completed.")
            print(message)
        else:
            print("GREENHOUSE LIVE SMOKE TEST: FAIL")
            print(message)
            return 1
    finally:
        await http.aclose()

    if blocked:
        return 2

    official = [
        job
        for job in jobs
        if job.job_id
        and job.apply_url
        and job.apply_url.startswith("https://")
        and job.provenance.get("discovery_method") == "structured"
        and not any(host in job.apply_url.lower() for host in ("linkedin.com", "indeed.com", "glassdoor.com"))
    ]
    print(f"Structured jobs: {len(jobs)}")
    print(f"Official Greenhouse URLs with job IDs: {len(official)}")
    if not official:
        print("GREENHOUSE LIVE SMOKE TEST: FAIL")
        print("The public Greenhouse API returned no job with an id and an official URL.")
        return 1

    profile = merge_resume_profile(
        load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml"),
        config.project_root / config.settings.candidate.profile_path,
    )
    sample = official[0]
    fit = evaluate_job(sample, profile, config.roles)
    critique = review_fit(fit, sample, profile)
    print(f"Sample: {sample.company_name} / {sample.title} / {sample.job_id}")
    print(f"URL: {sample.apply_url}")
    print(f"Intelligence: {critique.fit.decision} evidence={critique.evidence_quality}")

    browser_status = await _probe_browser(official[0].apply_url or "", config)
    print(f"Playwright page open: {browser_status}")
    report = config.project_root / config.settings.greenhouse_pilot.report_dir / "greenhouse-smoke.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                f"company={company.name}",
                f"jobs={len(jobs)}",
                f"official={len(official)}",
                f"decision={critique.fit.decision}",
                f"playwright={browser_status}",
            ]
        ),
        encoding="utf-8",
    )
    if browser_status == "BLOCKED":
        print("BLOCKED — live Greenhouse browser open was refused.")
        print("Structured extraction still returned official job URLs.")
        return 0
    print("GREENHOUSE LIVE SMOKE TEST: PASS")
    return 0


def _first_greenhouse(config: AppConfig) -> CompanyConfig | None:
    for company in config.universe.enabled_companies():
        if company.ats_type == "greenhouse" and company.ats_identifier:
            return company
    return None


async def _probe_browser(url: str, config: AppConfig) -> str:
    browser = PlaywrightBrowser(
        headless=config.settings.scraping.playwright.headless,
        timeout_ms=int(config.settings.scraping.playwright.timeout_seconds * 1000),
    )
    try:
        state = await browser.open(url)
    except BlockedPage as exc:
        return f"BLOCKED {exc}"
    except Exception as exc:
        return f"FAIL {type(exc).__name__}: {exc}"
    finally:
        await browser.aclose()
    if state.blocked:
        return "BLOCKED"
    return "PASS"

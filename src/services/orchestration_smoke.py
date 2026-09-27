"""Isolated Phase 2 smokes. They never write the production workbook or navigation files."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from src.agents.critic_agent import review_fit
from src.agents.extraction_agent import run_extraction
from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
from src.agents.job_intelligence_agent import evaluate_job
from src.llm.base import build_llm_provider
from src.models.config import AppConfig
from src.models.job import RawJobPosting
from src.models.state import PipelineState
from src.navigation.memory import NavigationMemory
from src.pilot.workday_target import discover_target_jobs, workday_board_url
from src.services.discovery_orchestrator import (
    _isolated_workbook,
    _load_saved_profile,
    _safe_message,
    grounding_status,
    orchestrate,
    render_orchestration,
)
from src.services.freshness import is_fresh
from src.services.url_verification import pick_direct_url, verify_url
from src.sources.base import HttpClient

__all__ = ["run_live_downstream_smoke", "run_multi_source_browser_smoke"]


def _stamp(path: Path) -> tuple[int, int, str] | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    return (path.stat().st_mtime_ns, len(data), hashlib.sha256(data).hexdigest())


def _directory_stamps(directory: Path) -> dict[str, tuple[int, int, str] | None]:
    if not directory.exists():
        return {}
    return {item.name: _stamp(item) for item in directory.iterdir() if item.is_file()}


def _copy_navigation(source: Path) -> tempfile.TemporaryDirectory:
    temp = tempfile.TemporaryDirectory(prefix="nav-smoke-")
    destination = Path(temp.name)
    if source.exists():
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, destination / item.name)
    return temp


def _with_browser(config: AppConfig, navigation_dir: Path, *, enable_jobright: bool = False) -> AppConfig:
    config = config.model_copy(update={"dry_run": True, "send_email": False, "fixture_mode": False})
    workday = config.settings.discovery.sources.workday.model_copy(update={"browser_enabled": True})
    updates: dict = {"workday": workday}
    if enable_jobright:
        updates["jobright"] = config.settings.discovery.sources.jobright.model_copy(
            update={"enabled": True, "allow_browser_render": False, "max_navigation_attempts": 0}
        )
    sources = config.settings.discovery.sources.model_copy(update=updates)
    discovery = config.settings.discovery.model_copy(update={"sources": sources})
    pilot = config.settings.greenhouse_pilot.model_copy(update={"navigation_dir": str(navigation_dir)})
    settings = config.settings.model_copy(update={"discovery": discovery, "greenhouse_pilot": pilot})
    return config.model_copy(update={"settings": settings})


def _strategy_line(directory: Path, company: str = "NVIDIA") -> str:
    record = NavigationMemory(directory).load(company, "workday")
    current = record.current if record is not None else None
    if current is None:
        return "missing"
    kinds = ",".join(action.action for action in current.actions)
    return f"version={current.version} status={current.status} actions={kinds}"


def _safety(before_book, after_book, before_nav, after_nav) -> bool:
    return before_book == after_book and before_nav == after_nav


async def run_multi_source_browser_smoke(
    config: AppConfig,
    *,
    sources: tuple[str, ...] = ("greenhouse", "workday"),
) -> int:
    """Greenhouse API plus NVIDIA browser discovery. CXS is not used for Workday."""
    production_flag = config.settings.discovery.sources.workday.browser_enabled
    navigation = config.project_root / config.settings.greenhouse_pilot.navigation_dir
    workbook = config.project_root / config.settings.output.current_workbook
    before_book = _stamp(workbook)
    before_nav = _directory_stamps(navigation)
    print("MULTI-SOURCE BROWSER SMOKE")
    print(f"sources: {', '.join(sources)}")
    print(f"production workday.browser_enabled: {str(production_flag).lower()}")
    print("invocation workday.browser_enabled: true")
    print("workday discovery: browser only")
    print("greenhouse sample: company_limit=1 job_limit=40")
    print("dry-run: workbook will not be written")

    temp = _copy_navigation(navigation)
    http = None
    provider = None
    try:
        smoke = _with_browser(config, Path(temp.name), enable_jobright="jobright" in sources)
        http = HttpClient(smoke)
        provider = build_llm_provider(smoke)
        reports: list = []
        result = await orchestrate(
            smoke,
            sources=sources,
            http=http,
            llm=provider,
            load_saved_profile=True,
            workday_browser_only=True,
            browser_company="NVIDIA",
            greenhouse_company_limit=1,
            greenhouse_job_limit=40,
            browser_reports=reports,
        )
        print(render_orchestration(result))
        browser_jobs = [
            posting
            for posting in result.postings
            if posting.source == "workday" and (posting.provenance or {}).get("discovery_method") == "browser"
        ]
        print(f"workday_browser_jobs: {len(browser_jobs)}")
        if browser_jobs:
            sample = browser_jobs[0]
            print(
                "provenance: "
                f"source={sample.source} "
                f"discovery_method={sample.provenance.get('discovery_method')} "
                f"job_id={sample.provenance.get('source_job_id')}"
            )
        else:
            print("provenance: no browser job")
        calls = sum(getattr(item, "navigation_model_calls", 0) for item in reports)
        reused = [item for item in reports if getattr(item, "strategy_reused", False)]
        version = reused[0].strategy_version if reused else (reports[0].strategy_version if reports else None)
        print(f"strategy_replay: {'PASS' if reused else 'FAIL'}")
        print(f"strategy_version_used: {version}")
        print(f"navigation_model_calls: {calls}")
        print(f"browser_reports: {len(reports)}")
        isolated = _isolated_workbook(result.funnel.state.jobs if result.funnel is not None else [])
        print(f"isolated_xlsx: {isolated}")
        print(f"email: {result.funnel.state.summary.email_status if result.funnel is not None else 'skipped'}")
    except Exception as exc:
        print(f"FAIL: {_safe_message(exc)}")
        return 1
    finally:
        if provider is not None:
            close = getattr(provider, "aclose", None)
            if callable(close):
                await close()
        if http is not None:
            await http.aclose()
        temp.cleanup()

    after_book = _stamp(workbook)
    after_nav = _directory_stamps(navigation)
    unchanged = _safety(before_book, after_book, before_nav, after_nav)
    print(f"production_workbook_unchanged: {before_book == after_book}")
    print(f"navigation_files_unchanged: {before_nav == after_nav}")
    print(f"nvidia_strategy_after: {_strategy_line(navigation)}")
    print(f"production workday.browser_enabled: {str(config.settings.discovery.sources.workday.browser_enabled).lower()}")
    if not unchanged:
        print("FAIL: production workbook or navigation files changed")
        return 1
    return 0


def _select_posting(postings: list[RawJobPosting]) -> RawJobPosting | None:
    usable = [
        posting
        for posting in postings
        if posting.apply_url
        and "myworkdayjobs.com" in posting.apply_url.lower()
        and len(posting.description or "") >= 80
    ]
    united_states = [posting for posting in usable if _looks_us(posting)]
    return (united_states or usable or [None])[0]


def _looks_us(posting: RawJobPosting) -> bool:
    text = f"{posting.location_raw or ''}".lower()
    if any(token in text for token in ("united states", "usa", "u.s.", "remote - us", "remote, us")):
        return True
    states = (
        "al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "ia", "id", "il", "in",
        "ks", "ky", "la", "ma", "md", "me", "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj",
        "nm", "nv", "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "va", "vt", "wa",
        "wi", "wv", "wy",
    )
    return any(f", {state}" in text or text.endswith(f" {state}") for state in states)


async def run_live_downstream_smoke(config: AppConfig, *, source: str = "workday") -> int:
    """Score one real job downstream without treating a stale posting as production-qualified."""
    print("LIVE DOWNSTREAM SMOKE")
    print(f"source: {source}")
    print("dry-run: workbook will not be written")
    if source != "workday":
        print("live downstream: BLOCKED")
        print("reason: this test exercises one public Workday posting")
        return 2

    production_flag = config.settings.discovery.sources.workday.browser_enabled
    navigation = config.project_root / config.settings.greenhouse_pilot.navigation_dir
    workbook = config.project_root / config.settings.output.current_workbook
    before_book = _stamp(workbook)
    before_nav = _directory_stamps(navigation)
    companies = [
        company
        for company in config.target_companies()
        if company.name.lower() == "nvidia" and company.ats_type == "workday"
    ]
    if not companies or not workday_board_url(companies[0]):
        print("live downstream: BLOCKED")
        print("reason: NVIDIA Workday board is not configured")
        return 2

    temp = _copy_navigation(navigation)
    http = None
    provider = None
    try:
        smoke = _with_browser(config, Path(temp.name))
        http = HttpClient(smoke)
        provider = build_llm_provider(smoke)
        discovered = await discover_target_jobs(
            smoke,
            board_url=workday_board_url(companies[0]) or "",
            company_name="NVIDIA",
            max_jobs=40,
            detail_limit=3,
        )
        posting = _select_posting(discovered.detailed or discovered.discovered)
        print(f"production workday.browser_enabled: {str(production_flag).lower()}")
        print(f"strategy_replay: {'PASS' if discovered.strategy_reused else 'FAIL'}")
        print(f"strategy_version_used: {discovered.strategy_version}")
        print(f"navigation_model_calls: {discovered.navigation_model_calls}")
        print(f"detail_pages: {len(discovered.detailed)}")
        if posting is None:
            print("live downstream: BLOCKED")
            print("reason: no public Workday posting with a description and official URL")
            print("production_final_qualified: 0")
            print("live_downstream_evaluated: 0")
        else:
            await _score_one(smoke, posting, http, provider)
    except Exception as exc:
        print(f"FAIL: {_safe_message(exc)}")
        return 1
    finally:
        if provider is not None:
            close = getattr(provider, "aclose", None)
            if callable(close):
                await close()
        if http is not None:
            await http.aclose()
        temp.cleanup()

    after_book = _stamp(workbook)
    after_nav = _directory_stamps(navigation)
    print(f"production_workbook_unchanged: {before_book == after_book}")
    print(f"navigation_files_unchanged: {before_nav == after_nav}")
    print(f"nvidia_strategy_after: {_strategy_line(navigation)}")
    print(f"production workday.browser_enabled: {str(config.settings.discovery.sources.workday.browser_enabled).lower()}")
    if not _safety(before_book, after_book, before_nav, after_nav):
        print("FAIL: production workbook or navigation files changed")
        return 1
    return 0


async def _score_one(config: AppConfig, posting: RawJobPosting, http: HttpClient, provider) -> None:
    fresh, age = is_fresh(
        posting,
        config.freshness_hours,
        use_updated_when_posted_missing=config.settings.run.freshness_use_updated_when_posted_missing,
    )
    age_text = "unknown" if age is None else f"{age:.1f}"
    print(f"selected_job: company={posting.company_name} id={posting.job_id} title={posting.title}")
    print(f"selected_location: {posting.location_raw or '-'}")
    print(f"description_chars: {len(posting.description or '')}")
    print(f"detail_extraction: {'PASS' if len(posting.description or '') >= 80 and posting.title else 'FAIL'}")
    print(f"production_freshness: {'PASS' if fresh else 'REJECT'} age_hours={age_text}")

    state = PipelineState(config=config, raw_postings=[posting], resources={"llm": provider, "http": http})
    await run_extraction(state)
    extracted = state.jobs[0] if state.jobs else None
    print(f"canonical_job: {'PASS' if extracted is not None else 'FAIL'}")

    check = pick_direct_url(posting, config.url_policy)
    checked = await verify_url(check, config, http)
    official = "myworkdayjobs.com" in (checked.url or "").lower()
    identity = bool(posting.job_id) and (
        str(posting.job_id) in (checked.url or "") or checked.job_id == str(posting.job_id)
    )
    if checked.reachable is True and official and identity:
        url_status = "PASS"
    elif checked.reachable is False:
        url_status = "FAIL"
    else:
        url_status = "BLOCKED"
    print(f"url_verification: {url_status}")
    print(f"url_reachable: {checked.reachable}")
    print(f"url_official_domain: {official}")
    print(f"url_identity_consistent: {identity}")
    print(f"url_reason: {checked.reason}")

    if extracted is not None:
        await run_h1b_enrichment(state)
        status = extracted.visa_sponsorship_status.value
        mapped = {"CONFIRMED": "PRESENT", "LIKELY": "PRESENT", "NOT_SUPPORTED": "ABSENT"}.get(status, "UNKNOWN")
        print(f"h1b: {mapped}")
        print(f"h1b_status: {status}")
        print(f"h1b_confidence: {extracted.visa_sponsorship_confidence}")
        print("h1b_filter: false")
    else:
        print("h1b: BLOCKED")

    profile, version, digest, canonical = _load_saved_profile(config)
    validity, coverage = grounding_status(canonical)
    print(f"profile_loads: 1")
    print(f"profile_version: {version}")
    print(f"profile_sha256: {digest or '-'}")
    print(f"profile_validity: {validity}")
    print(f"evidence_grounding: {coverage}")
    if profile is None or extracted is None:
        print("job_intelligence: BLOCKED")
        print("critic: BLOCKED")
    else:
        fit = evaluate_job(posting, profile, config.roles)
        critique = review_fit(fit, posting, profile)
        print(
            "job_intelligence: PASS "
            f"role={fit.role_alignment} experience={fit.experience_alignment} "
            f"education={fit.education_alignment} location={fit.location_alignment} "
            f"freshness={fit.freshness} sponsorship={fit.sponsorship}"
        )
        print(f"matched_requirements: {len(critique.fit.matched_requirements)}")
        print(f"critic: PASS unsupported_claims={len(critique.unsupported_claims)} corrected={critique.corrected_fields}")
        print(f"critic_freshness: {critique.fit.freshness}")

    production_final = 0
    if fresh and extracted is not None:
        from src.pilot.workday_qualify import qualify_workday_postings

        funnel = await qualify_workday_postings(config, [posting], llm=provider, http=http)
        production_final = funnel.final_count
    print(f"production_final_qualified: {production_final}")
    print("live_downstream_evaluated: 1")
    print("live_downstream_note: freshness was not used to skip URL, H1B, intelligence, or critic")
    isolated = _isolated_workbook([extracted] if extracted is not None else [])
    print(f"isolated_xlsx: {isolated}")
    print("email: skipped")

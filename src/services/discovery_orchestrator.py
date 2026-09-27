"""Greenhouse and Workday discovery on the shared qualification path.

Each source runs independently. A failure is recorded and the other source
continues. Browser discovery stays off unless the existing Workday flag is on.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from src.agents.critic_agent import review_fit
from src.agents.discovery_agent import _dedupe_raw
from src.agents.job_intelligence_agent import evaluate_job
from src.llm.base import LLMProvider, NullLLMProvider, build_llm_provider
from src.models.config import AppConfig
from src.models.job import RawJobPosting, RejectionReason
from src.pilot.workday_qualify import QualificationFunnel, qualify_workday_postings
from src.services.candidate_profile import CandidateProfile, load_candidate_profile, merge_resume_profile
from src.services.rejection_codes import rejection_counts
from src.services.resume_store import load_profile
from src.services.xlsx import write_workbooks
from src.sources.base import HttpClient, SourceContext
from src.sources.ashby import AshbySource
from src.sources.greenhouse import GreenhouseSource
from src.sources.jobright import JobrightSource
from src.sources.lever import LeverSource
from src.sources.workday import WorkdaySource
from src.utils.dates import utcnow
from src.utils.logging import get_logger

__all__ = [
    "ORCHESTRATED_SOURCES",
    "OrchestrationResult",
    "SourceFailure",
    "SourceReport",
    "collect_ashby",
    "collect_greenhouse",
    "collect_jobright",
    "collect_lever",
    "collect_workday",
    "orchestrate",
    "run_ashby_smoke",
    "run_jobright_smoke",
    "run_lever_smoke",
    "run_multi_source_smoke",
]

log = get_logger(__name__)

ORCHESTRATED_SOURCES = ("greenhouse", "workday", "lever", "ashby", "jobright")
CANONICAL_PROFILE_SHA256 = "08a662b504efcec14eb4634e5f0d4f59bd99c862c63e4a053ca787857102eead"
Collector = Callable[[], Awaitable[list]]

_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization)\b\s*[=:]\s*\S+"
)


@dataclass
class SourceFailure:
    source: str
    stage: str
    error_type: str
    message: str
    timestamp: str


@dataclass
class SourceRunStats:
    companies_attempted: int = 0
    companies_succeeded: int = 0
    companies_failed: int = 0
    duration_seconds: float = 0.0
    cap_reached: bool = False
    cap_note: str = ""
    timeouts: int = 0


@dataclass
class SourceReport:
    name: str
    enabled: bool
    status: str
    discovered: int = 0
    normalized: int = 0
    deduplicated: int = 0
    rejected: int = 0
    final: int = 0
    browser_enabled: bool | None = None
    discovery: str = "NOT EXERCISED"
    discovery_method: str = ""
    companies_attempted: int = 0
    companies_succeeded: int = 0
    companies_failed: int = 0
    duration_seconds: float = 0.0
    cap_reached: bool = False
    cap_note: str = ""
    timeouts: int = 0
    failures: list[SourceFailure] = field(default_factory=list)


@dataclass
class OrchestrationResult:
    sources: dict[str, SourceReport]
    postings: list[RawJobPosting]
    funnel: QualificationFunnel | None = None
    intelligence_evaluated: int = 0
    critic_reviewed: int = 0
    unsupported_claims: int = 0
    profile_loads: int = 0
    profile_version: int | None = None
    profile_sha256: str = ""
    llm_calls: int = 0
    llm_successes: int = 0
    llm_failures: int = 0
    llm_retries: int = 0
    llm_fallbacks: int = 0
    llm_deterministic_avoided: int = 0
    llm_circuit_skips: int = 0
    llm_recovery_attempts: int = 0
    llm_circuit_state: str = "CLOSED"
    llm_failures_by_category: dict[str, int] = field(default_factory=dict)
    llm_by_purpose: dict[str, int] = field(default_factory=dict)
    fits: list = field(default_factory=list)


class SourceDiscoveryError(RuntimeError):
    """Every company attempt for one source failed."""


def _is_timeout(error_type: str, message: str) -> bool:
    kind = error_type.replace(" ", "").lower()
    if kind in {"timeouterror", "timeout", "asynciotimeouterror"}:
        return True
    return "timed out" in message.lower()


def _safe_message(exc: BaseException) -> str:
    text = _SECRET_RE.sub(r"\1=<redacted>", str(exc))
    return text.replace("\n", " ")[:500]


def _failure(source: str, stage: str, exc: BaseException) -> SourceFailure:
    return SourceFailure(
        source=source,
        stage=stage,
        error_type=type(exc).__name__,
        message=_safe_message(exc),
        timestamp=utcnow().isoformat(),
    )


def ensure_provenance(posting: RawJobPosting) -> RawJobPosting:
    """Fill the shared provenance keys without replacing a source's own method."""
    provenance = dict(posting.provenance or {})
    found = [str(item) for item in provenance.get("discovered_from") or [] if item]
    if posting.source and posting.source not in found:
        found.append(posting.source)
    provenance["discovered_from"] = found
    provenance.setdefault("discovery_method", "api")
    if posting.apply_url and not provenance.get("source_url"):
        provenance["source_url"] = posting.apply_url
    if posting.job_id and not provenance.get("source_job_id"):
        provenance["source_job_id"] = str(posting.job_id)
    if posting.discovered_at is not None and not provenance.get("discovered_at"):
        stamp = posting.discovered_at
        provenance["discovered_at"] = stamp.isoformat() if isinstance(stamp, datetime) else str(stamp)
    return posting.model_copy(update={"provenance": provenance})


def _usable(posting: RawJobPosting) -> bool:
    return bool((posting.company_name or "").strip() and (posting.title or "").strip() and (posting.apply_url or "").strip())


def _count(postings: list[RawJobPosting], source: str) -> int:
    return sum(1 for posting in postings if posting.source == source)


async def _companies(config: AppConfig, source_cls, http: HttpClient):
    ctx = SourceContext(config=config, http=http, logger=log)
    source = source_cls(ctx)
    companies = [company for company in config.target_companies() if source.supports(company)]
    return source, companies


async def _collect_adapter(
    config: AppConfig,
    http: HttpClient,
    source_cls,
    *,
    failures: list[SourceFailure],
    company_limit: int | None = None,
    job_limit: int | None = None,
    stats: SourceRunStats | None = None,
) -> list[RawJobPosting]:
    source, companies = await _companies(config, source_cls, http)
    configured = len(companies)
    if company_limit is not None:
        companies = companies[:company_limit]
    if not source.enabled:
        return []
    jobs: list[RawJobPosting] = []
    company_failures = 0
    company_successes = 0
    timeouts = 0
    capped: list[str] = []
    selection_notes: list[str] = []
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(config.settings.run.max_concurrency)

    async def one(company) -> None:
        nonlocal company_failures, company_successes, timeouts
        async with semaphore:
            result = await source.discover_result(company)
        if result.success:
            company_successes += 1
            if (result.diagnostics or {}).get("source_list_cap_reached"):
                capped.append(company.name)
            note = (result.diagnostics or {}).get("cap_note")
            if note:
                selection_notes.append(str(note))
            jobs.extend(ensure_provenance(posting) for posting in result.jobs)
            return
        company_failures += 1
        error_type = result.status or "SourceError"
        message = _safe_message(RuntimeError(result.error or "discovery failed"))
        if _is_timeout(error_type, message):
            timeouts += 1
        failures.append(
            SourceFailure(
                source=source.name,
                stage="discovery",
                error_type=error_type,
                message=message,
                timestamp=utcnow().isoformat(),
            )
        )

    await asyncio.gather(*(one(company) for company in companies))
    if stats is not None:
        stats.companies_attempted = len(companies)
        stats.companies_succeeded = company_successes
        stats.companies_failed = company_failures
        stats.duration_seconds += time.perf_counter() - started
        stats.timeouts += timeouts
        notes: list[str] = []
        if company_limit is not None and configured > company_limit:
            stats.cap_reached = True
            notes.append(f"company_cap={company_limit} configured={configured}")
        if job_limit is not None and len(jobs) > job_limit:
            stats.cap_reached = True
            notes.append(f"job_cap={job_limit} discovered_before_cap={len(jobs)}")
        if capped:
            stats.cap_reached = True
            notes.append("list_cap=" + ",".join(capped))
        notes.extend(selection_notes)
        if notes:
            stats.cap_note = "; ".join(item for item in (stats.cap_note, *notes) if item)
    if companies and company_failures == len(companies) and not jobs:
        raise SourceDiscoveryError(f"{source.name} failed for every configured company")
    if job_limit is not None:
        return jobs[:job_limit]
    return jobs


async def collect_greenhouse(
    config: AppConfig,
    http: HttpClient,
    *,
    failures: list[SourceFailure] | None = None,
    company_limit: int | None = None,
    job_limit: int | None = None,
    stats: SourceRunStats | None = None,
    **_extras,
) -> list[RawJobPosting]:
    return await _collect_adapter(
        config,
        http,
        GreenhouseSource,
        failures=failures if failures is not None else [],
        company_limit=company_limit,
        job_limit=job_limit,
        stats=stats,
    )


async def collect_workday(
    config: AppConfig,
    http: HttpClient | None,
    *,
    failures: list[SourceFailure] | None = None,
    browser_runner: Callable[[str, str], Awaitable[list[RawJobPosting]]] | None = None,
    browser_only: bool = False,
    browser_company: str | None = None,
    browser_reports: list | None = None,
    stats: SourceRunStats | None = None,
    **_extras,
) -> list[RawJobPosting]:
    """CXS discovery, plus browser discovery only when that flag is on.

    ``browser_only`` skips CXS. The production flag still has to be enabled on
    the config object passed in; this function does not turn the flag on.
    """
    bucket = failures if failures is not None else []
    if browser_only:
        found: list[RawJobPosting] = []
    else:
        if http is None:
            raise SourceDiscoveryError("workday CXS requires an HTTP client")
        found = await _collect_adapter(config, http, WorkdaySource, failures=bucket, stats=stats)
    from src.pilot.workday_target import browser_discovery_enabled, workday_board_url

    if not browser_discovery_enabled(config):
        return found
    if http is None:
        companies = [
            company
            for company in config.target_companies()
            if company.ats_type == "workday" and company.ats_identifier
        ]
    else:
        _source, companies = await _companies(config, WorkdaySource, http)
    if browser_company:
        companies = [company for company in companies if company.name.lower() == browser_company.lower()]
    for company in companies:
        board = workday_board_url(company)
        if not board:
            continue
        try:
            if browser_runner is not None:
                extra = await browser_runner(company.name, board)
            else:
                from src.pilot.workday_target import discover_target_jobs

                discovered = await discover_target_jobs(config, board_url=board, company_name=company.name)
                if browser_reports is not None:
                    browser_reports.append(discovered)
                if stats is not None and discovered.unopened_plausible:
                    stats.cap_reached = True
                    note = (
                        f"detail_page_cap_reached unopened_plausible={discovered.unopened_plausible} "
                        f"detailed={len(discovered.detailed)}"
                    )
                    stats.cap_note = "; ".join(item for item in (stats.cap_note, note) if item)
                extra = list(discovered.detailed)
            found.extend(ensure_provenance(posting) for posting in extra)
        except Exception as exc:
            bucket = failures if failures is not None else []
            bucket.append(_failure("workday", "browser_discovery", exc))
            log.warning("workday browser discovery failed", company=company.name, error=_safe_message(exc))
    return found


async def collect_jobright(
    config: AppConfig,
    http: HttpClient,
    *,
    failures: list[SourceFailure] | None = None,
    stats: SourceRunStats | None = None,
    **_extras,
) -> list[RawJobPosting]:
    """One bounded public Jobright listing. Official URLs only survive later verification."""
    del failures
    if not config.settings.discovery.sources.is_enabled("jobright"):
        return []
    ctx = SourceContext(config=config, http=http, logger=log)
    source = JobrightSource(ctx)
    started = time.perf_counter()
    result = await source.discover_result(None)
    enabled_companies = [company.name for company in config.target_companies() if company.enabled]
    matched = {posting.company_name for posting in result.jobs if posting.company_name}
    if stats is not None:
        stats.companies_attempted = len(enabled_companies)
        stats.companies_succeeded = len(matched) if result.success else 0
        stats.companies_failed = 0 if result.success else 1
        stats.duration_seconds = time.perf_counter() - started
        note = str((result.diagnostics or {}).get("cap_note") or "")
        if (result.diagnostics or {}).get("detail_blocked"):
            note = "; ".join(item for item in (note, "detail_blocked") if item)
        if (result.diagnostics or {}).get("cap_reached") or (result.diagnostics or {}).get("source_list_cap_reached"):
            stats.cap_reached = True
            stats.cap_note = note or "cap_reached"
        elif note:
            stats.cap_note = note
    if not result.success:
        detail = result.error or "jobright discovery failed"
        if result.status == "BLOCKED":
            detail = f"BLOCKED — public automated discovery unavailable: {detail}"
        raise SourceDiscoveryError(detail)
    return [ensure_provenance(posting) for posting in result.jobs]


async def _collect_board(
    config: AppConfig,
    http: HttpClient,
    source_cls,
    *,
    failures: list[SourceFailure] | None,
    stats: SourceRunStats | None,
) -> list[RawJobPosting]:
    board = getattr(config.settings.discovery.sources, source_cls.name)
    return await _collect_adapter(
        config,
        http,
        source_cls,
        failures=failures if failures is not None else [],
        company_limit=board.max_companies,
        job_limit=board.max_jobs,
        stats=stats,
    )


async def collect_lever(
    config: AppConfig,
    http: HttpClient,
    *,
    failures: list[SourceFailure] | None = None,
    stats: SourceRunStats | None = None,
    **_extras,
) -> list[RawJobPosting]:
    return await _collect_board(config, http, LeverSource, failures=failures, stats=stats)


async def collect_ashby(
    config: AppConfig,
    http: HttpClient,
    *,
    failures: list[SourceFailure] | None = None,
    stats: SourceRunStats | None = None,
    **_extras,
) -> list[RawJobPosting]:
    return await _collect_board(config, http, AshbySource, failures=failures, stats=stats)


LIVE_COLLECTORS = {
    "greenhouse": collect_greenhouse,
    "workday": collect_workday,
    "lever": collect_lever,
    "ashby": collect_ashby,
    "jobright": collect_jobright,
}

DEFAULT_SMOKE_SOURCES = ("greenhouse", "workday")


async def orchestrate(
    config: AppConfig,
    *,
    sources: tuple[str, ...] = ORCHESTRATED_SOURCES,
    collectors: dict[str, Collector] | None = None,
    http: HttpClient | None = None,
    llm: LLMProvider | None = None,
    profile: CandidateProfile | None = None,
    load_saved_profile: bool = False,
    h1b=None,
    browser_runner: Callable[[str, str], Awaitable[list[RawJobPosting]]] | None = None,
    workday_browser_only: bool = False,
    browser_company: str | None = None,
    greenhouse_company_limit: int | None = None,
    greenhouse_job_limit: int | None = None,
    browser_reports: list | None = None,
) -> OrchestrationResult:
    """Discover the requested sources, dedupe, then use the shared qualifiers."""
    selected = tuple(name.strip().lower() for name in sources if name.strip())
    reports: dict[str, SourceReport] = {}
    combined: list[RawJobPosting] = []
    owns_http = http is None and collectors is None
    client = http
    if owns_http:
        client = HttpClient(config)

    try:
        for name in selected:
            reports[name] = await _run_source(
                config,
                name,
                collectors=collectors,
                http=client,
                browser_runner=browser_runner,
                workday_browser_only=workday_browser_only,
                browser_company=browser_company,
                greenhouse_company_limit=greenhouse_company_limit,
                greenhouse_job_limit=greenhouse_job_limit,
                browser_reports=browser_reports,
            )
            combined.extend(reports[name].__dict__.get("_jobs", []))
    finally:
        if owns_http and client is not None:
            await client.aclose()

    discovered = list(combined)
    for name, report in reports.items():
        report.discovered = _count(discovered, name)
    deduped = _dedupe_raw(discovered)
    for name, report in reports.items():
        report.deduplicated = _count(deduped, name)
        report.normalized = sum(1 for posting in deduped if posting.source == name and _usable(posting))

    result = OrchestrationResult(sources=reports, postings=deduped)
    provider = llm if llm is not None else NullLLMProvider()
    resources = {"h1b": h1b} if h1b is not None else None
    result.funnel = await qualify_workday_postings(
        config,
        deduped,
        llm=provider,
        http=client if not owns_http else None,
        resources=resources,
    )
    for name, report in reports.items():
        if report.status == "DISABLED":
            continue
        report.final = sum(1 for job in result.funnel.state.jobs if job.source == name)
        report.rejected = sum(1 for item in result.funnel.state.rejected if item.source == name)

    if load_saved_profile and profile is None:
        result.profile_loads += 1
        profile, version, digest, _canonical = _load_saved_profile(config)
        result.profile_version = version
        result.profile_sha256 = digest
    elif profile is not None:
        result.profile_loads += 1
        result.profile_version = profile.profile_version

    if profile is not None and profile.profile_version is not None and result.funnel is not None:
        by_url = {(posting.apply_url or ""): posting for posting in deduped}
        by_id = {(posting.job_id or ""): posting for posting in deduped if posting.job_id}
        for job in result.funnel.state.jobs:
            posting = by_id.get(job.job_id or "") or by_url.get(job.direct_application_url or "")
            if posting is None:
                continue
            fit = evaluate_job(posting, profile, config.roles)
            critique = review_fit(fit, posting, profile)
            result.intelligence_evaluated += 1
            result.critic_reviewed += 1
            result.unsupported_claims += len(critique.unsupported_claims)
            result.fits.append(critique)

    stats = getattr(provider, "stats", None)
    if stats is not None:
        result.llm_calls = stats.calls
        result.llm_successes = getattr(stats, "successes", 0)
        result.llm_failures = stats.failures
        result.llm_retries = getattr(stats, "retries", 0)
        result.llm_fallbacks = getattr(stats, "fallbacks", 0)
        result.llm_deterministic_avoided = getattr(stats, "deterministic_avoided", 0)
        result.llm_circuit_skips = getattr(stats, "skipped_circuit_open", 0)
        result.llm_recovery_attempts = getattr(stats, "recovery_attempts", 0)
        result.llm_circuit_state = getattr(stats, "circuit_state", "CLOSED") or "CLOSED"
        result.llm_failures_by_category = dict(getattr(stats, "failures_by_category", {}) or {})
        result.llm_by_purpose = dict(stats.by_purpose)
    return result


async def _run_source(
    config: AppConfig,
    name: str,
    *,
    collectors: dict[str, Collector] | None,
    http: HttpClient | None,
    browser_runner,
    workday_browser_only: bool = False,
    browser_company: str | None = None,
    greenhouse_company_limit: int | None = None,
    greenhouse_job_limit: int | None = None,
    browser_reports: list | None = None,
) -> SourceReport:
    enabled = config.settings.discovery.sources.is_enabled(name) if name in ORCHESTRATED_SOURCES or hasattr(config.settings.discovery.sources, name) else False
    if name not in ORCHESTRATED_SOURCES:
        report = SourceReport(name=name, enabled=False, status="DISABLED", discovery="NOT IMPLEMENTED")
        report.failures.append(
            SourceFailure(
                source=name,
                stage="discovery",
                error_type="NotImplementedError",
                message=f"{name} is not part of this orchestration phase",
                timestamp=utcnow().isoformat(),
            )
        )
        return report
    source_settings = getattr(config.settings.discovery.sources, name, None)
    browser_flag = getattr(source_settings, "browser_enabled", None)
    browser = None if browser_flag is None else bool(browser_flag)
    report = SourceReport(name=name, enabled=enabled, status="DISABLED", browser_enabled=browser, discovery="DISABLED")
    if not enabled:
        return report

    failures: list[SourceFailure] = []
    jobs: list[RawJobPosting] = []
    stats = SourceRunStats()
    started = time.perf_counter()
    try:
        if collectors is not None and name in collectors:
            raw_items = await collectors[name]()
            for item in raw_items:
                try:
                    if not isinstance(item, RawJobPosting):
                        raise TypeError(f"malformed job from {name}: {type(item).__name__}")
                    jobs.append(ensure_provenance(item))
                except Exception as exc:
                    failures.append(_failure(name, "normalize", exc))
        elif http is None:
            raise SourceDiscoveryError(f"{name} has no HTTP client")
        else:
            collector = LIVE_COLLECTORS.get(name)
            if collector is None:
                raise SourceDiscoveryError(f"{name} has no collector")
            kwargs = {
                "failures": failures,
                "stats": stats,
                "browser_runner": browser_runner,
                "browser_only": workday_browser_only,
                "browser_company": browser_company,
                "browser_reports": browser_reports,
            }
            # Sample caps are a Greenhouse smoke control. Applying them to Workday
            # would drop tenants that are not first in the company list.
            if name == "greenhouse":
                kwargs["company_limit"] = greenhouse_company_limit
                kwargs["job_limit"] = greenhouse_job_limit
            jobs = await collector(config, http, **kwargs)
    except Exception as exc:
        failure = _failure(name, "discovery", exc)
        failures.append(failure)
        jobs = []
        if _is_timeout(failure.error_type, failure.message):
            stats.timeouts += 1
        _apply_stats(report, stats, time.perf_counter() - started)
        report.failures = failures
        report.status = "FAILED"
        report.discovery = "FAIL"
        report.__dict__["_jobs"] = []
        log.warning("source failed; continuing", source=name, error=_safe_message(exc))
        return report

    _apply_stats(report, stats, time.perf_counter() - started)
    report.discovery_method = _methods(jobs)
    report.failures = failures
    report.__dict__["_jobs"] = jobs
    report.status, report.discovery = _source_status(jobs, failures)
    return report


def _apply_stats(report: SourceReport, stats: SourceRunStats, elapsed: float) -> None:
    report.companies_attempted = stats.companies_attempted
    report.companies_succeeded = stats.companies_succeeded
    report.companies_failed = stats.companies_failed
    report.duration_seconds = round(stats.duration_seconds or elapsed, 3)
    report.cap_reached = stats.cap_reached
    report.cap_note = stats.cap_note
    report.timeouts = stats.timeouts


def _methods(jobs: list[RawJobPosting]) -> str:
    found = {
        str((posting.provenance or {}).get("discovery_method") or "")
        for posting in jobs
    }
    return ",".join(sorted(item for item in found if item))


def _source_status(jobs: list[RawJobPosting], failures: list[SourceFailure]) -> tuple[str, str]:
    """Zero jobs without an error is EMPTY. Jobs plus an error is PARTIAL."""
    if jobs and failures:
        return "PARTIAL", "PARTIAL"
    if jobs:
        return "SUCCESS", "PASS"
    if failures:
        return "FAILED", "FAIL"
    return "EMPTY", "EMPTY"


def _load_saved_profile(config: AppConfig):
    path = config.project_root / config.settings.candidate.profile_path
    search = load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml")
    profile = merge_resume_profile(search, path)
    canonical = load_profile(path)
    version = canonical.profile_version if canonical is not None else profile.profile_version
    digest = canonical.resume.sha256 if canonical is not None else ""
    if version is None:
        return None, None, digest, canonical
    return profile, version, digest, canonical


def grounding_status(canonical) -> tuple[str, str]:
    """Separate profile validity from whether skill claims carry resume quotes.

    Validity is the saved version and hash. Coverage is whether each stored
    skill has a quote. This does not rewrite the profile.
    """
    if canonical is None:
        return "FAIL", "FAIL"
    digest = getattr(getattr(canonical, "resume", None), "sha256", "")
    valid = canonical.profile_version == 1 and digest == CANONICAL_PROFILE_SHA256
    quoted = 0
    bare = 0
    for skill in canonical.skills:
        evidence = getattr(skill, "evidence", None) or []
        if any(str(item.get("quote") or "").strip() for item in evidence):
            quoted += 1
        else:
            bare += 1
    if quoted and bare == 0:
        coverage = "PASS"
    elif quoted:
        coverage = "PARTIAL"
    else:
        coverage = "FAIL"
    return ("PASS" if valid else "FAIL"), coverage


def _source_gate_line(result: OrchestrationResult, name: str) -> str:
    """Shared-pipeline counts for one source. Intelligence does not drop jobs."""
    report = result.sources.get(name)
    funnel = result.funnel
    if report is None or funnel is None:
        return ""
    counts: dict[RejectionReason, int] = {}
    for item in funnel.state.rejected:
        if item.source != name:
            continue
        counts[item.reason] = counts.get(item.reason, 0) + 1

    def drop(*reasons: RejectionReason) -> int:
        return sum(counts.get(reason, 0) for reason in reasons)

    after_role = report.deduplicated - drop(RejectionReason.EXTRACTION_FAILED, RejectionReason.ROLE)
    after_exp = after_role - drop(RejectionReason.SENIORITY)
    after_loc = after_exp - drop(RejectionReason.LOCATION)
    after_emp = after_loc - drop(RejectionReason.EMPLOYMENT_TYPE)
    after_fresh = after_emp - drop(RejectionReason.FRESHNESS)
    after_url = after_fresh - drop(RejectionReason.INVALID_URL)
    intelligence = sum(1 for critique in result.fits if critique.fit.job_source == name)
    return (
        "  gates: "
        f"role={after_role} seniority={after_exp} location={after_loc} "
        f"employment={after_emp} freshness={after_fresh} url={after_url} "
        f"h1b={after_url} intelligence={intelligence} critic={intelligence} final={report.final}"
    )


def render_orchestration(result: OrchestrationResult) -> str:
    lines = ["MULTI-SOURCE ORCHESTRATION", "=========================="]
    for name in result.sources:
        report = result.sources[name]
        browser = ""
        if report.browser_enabled is not None:
            browser = f" browser_enabled={str(report.browser_enabled).lower()}"
        lines.append(
            f"{name}: status={report.status}{browser} discovery={report.discovery} "
            f"method={report.discovery_method or '-'} "
            f"companies={report.companies_attempted}/{report.companies_succeeded}/{report.companies_failed} "
            f"discovered={report.discovered} normalized={report.normalized} "
            f"deduplicated={report.deduplicated} rejected={report.rejected} final={report.final} "
            f"duration_seconds={report.duration_seconds} errors={len(report.failures)} "
            f"timeouts={report.timeouts} cap_reached={str(report.cap_reached).lower()}"
        )
        if report.cap_note:
            lines.append(f"  cap: {report.cap_note}")
        for failure in report.failures:
            lines.append(
                f"  failure stage={failure.stage} type={failure.error_type} "
                f"at={failure.timestamp} message={failure.message}"
            )
        gate_line = _source_gate_line(result, name)
        if gate_line:
            lines.append(gate_line)
    funnel = result.funnel
    if funnel is not None:
        lines.append(
            "qualification: "
            f"discovered={sum(item.discovered for item in result.sources.values())} "
            f"normalized={funnel.extracted} "
            f"deduplicated={len(result.postings)} "
            f"role={funnel.role_candidates} "
            f"seniority={funnel.experience_candidates} "
            f"experience={funnel.experience_candidates} "
            f"location={funnel.location_candidates} "
            f"employment={funnel.employment_candidates} "
            f"freshness={funnel.fresh_candidates} "
            f"url_accepted={funnel.url_verified} "
            f"url_verified={sum(1 for job in funnel.state.jobs if job.url_verified)} "
            f"h1b={funnel.url_verified} "
            f"intelligence={result.intelligence_evaluated} "
            f"critic={result.critic_reviewed} "
            f"final={funnel.final_count}"
        )
        lines.append(f"rejections: {rejection_counts(funnel.state.rejected) or {}}")
        lines.append(f"email: {funnel.state.summary.email_status}")
        lines.append("llm_circuit_scope: in-process; a new run starts CLOSED")
    lines.append(f"profile_loads: {result.profile_loads}")
    lines.append(f"profile_version: {result.profile_version}")
    lines.append(f"profile_sha256: {result.profile_sha256 or '-'}")
    lines.append(f"llm_calls: {result.llm_calls}")
    lines.append(f"llm_successes: {result.llm_successes}")
    lines.append(f"llm_failures: {result.llm_failures}")
    lines.append(f"llm_retries: {result.llm_retries}")
    lines.append(f"llm_deterministic_avoided: {result.llm_deterministic_avoided}")
    lines.append(f"llm_circuit_skips: {result.llm_circuit_skips}")
    lines.append(f"llm_fallbacks: {result.llm_fallbacks}")
    lines.append(f"llm_circuit_state: {result.llm_circuit_state}")
    lines.append(f"llm_recovery_attempts: {result.llm_recovery_attempts}")
    lines.append(f"llm_failures_by_category: {result.llm_failures_by_category or {}}")
    lines.append(f"llm_by_purpose: {result.llm_by_purpose or {}}")
    lines.append("llm_by_source: shared agents record purpose, not source")
    return "\n".join(lines)


def _smoke_config(config: AppConfig, *, enable_jobright: bool = False, render_jobright: bool = False) -> AppConfig:
    """Dry-run copy. Workday browser stays at its configured value."""
    config = config.model_copy(update={"dry_run": True, "send_email": False, "fixture_mode": False})
    workday = config.settings.discovery.sources.workday.model_copy(update={"max_jobs": 20})
    lever = config.settings.discovery.sources.lever.model_copy(
        update={"max_jobs": 30, "max_companies": 2, "prioritize_fresh_targets": True}
    )
    ashby = config.settings.discovery.sources.ashby.model_copy(
        update={"max_jobs": 30, "max_companies": 2, "prioritize_fresh_targets": True}
    )
    updates: dict = {"workday": workday, "lever": lever, "ashby": ashby}
    if enable_jobright:
        updates["jobright"] = config.settings.discovery.sources.jobright.model_copy(
            update={
                "enabled": True,
                "allow_browser_render": render_jobright,
                "max_navigation_attempts": 1 if render_jobright else 0,
            }
        )
    sources = config.settings.discovery.sources.model_copy(update=updates)
    discovery = config.settings.discovery.model_copy(update={"sources": sources})
    settings = config.settings.model_copy(update={"discovery": discovery})
    return config.model_copy(update={"settings": settings})


async def run_multi_source_smoke(config: AppConfig, *, sources: tuple[str, ...] = DEFAULT_SMOKE_SOURCES) -> int:
    """Live Greenhouse API and Workday CXS through the shared pipeline. No production workbook write."""
    config = _smoke_config(config, enable_jobright="jobright" in sources)
    production = config.project_root / config.settings.output.current_workbook
    before = production.stat().st_mtime_ns if production.exists() else None
    print("MULTI-SOURCE SMOKE")
    print(f"sources: {', '.join(sources)}")
    print(f"workday.browser_enabled: {config.settings.discovery.sources.workday.browser_enabled}")
    print(f"jobright.enabled: {config.settings.discovery.sources.jobright.enabled}")
    print(f"freshness_hours: {config.freshness_hours}")
    print("dry-run: workbook will not be written")

    http = HttpClient(config)
    provider = build_llm_provider(config)
    try:
        result = await orchestrate(
            config,
            sources=sources,
            http=http,
            llm=provider,
            load_saved_profile=True,
        )
    except Exception as exc:
        print(f"FAIL: {_safe_message(exc)}")
        return 1
    finally:
        close = getattr(provider, "aclose", None)
        if callable(close):
            await close()
        await http.aclose()

    print(render_orchestration(result))
    isolated = _isolated_workbook(result.funnel.state.jobs if result.funnel is not None else [])
    after = production.stat().st_mtime_ns if production.exists() else None
    unchanged = before == after
    print(f"isolated_xlsx: {isolated}")
    print(f"production_workbook_unchanged: {unchanged}")
    print(f"intelligence_evaluated: {result.intelligence_evaluated}")
    print(f"critic_reviewed: {result.critic_reviewed}")
    if not unchanged:
        print("FAIL: production workbook changed")
        return 1
    return 0


def _stage(blocked: bool, exercised: bool, ok: bool) -> str:
    if blocked:
        return "BLOCKED"
    if not exercised:
        return "PARTIAL"
    return "PASS" if ok else "FAIL"


async def run_jobright_smoke(config: AppConfig) -> int:
    """Public Jobright discovery only. Dry-run. Does not write the production workbook."""
    config = _smoke_config(config, enable_jobright=True, render_jobright=True)
    production = config.project_root / config.settings.output.current_workbook
    before = production.stat().st_mtime_ns if production.exists() else None
    http = HttpClient(config)
    provider = build_llm_provider(config)
    try:
        result = await orchestrate(
            config,
            sources=("jobright",),
            http=http,
            llm=provider,
            load_saved_profile=True,
        )
    except Exception as exc:
        print("JOBRIGHT SMOKE")
        print(f"Discovery: BLOCKED — {_safe_message(exc)}")
        return 1
    finally:
        close = getattr(provider, "aclose", None)
        if callable(close):
            await close()
        await http.aclose()

    report = result.sources["jobright"]
    blocked = any("BLOCKED" in failure.message for failure in report.failures)
    postings = [posting for posting in result.postings if posting.source == "jobright"]
    resolved = sum(1 for posting in postings if (posting.provenance or {}).get("official_source"))
    jobs = list(result.funnel.state.jobs) if result.funnel is not None else []
    rejected = list(result.funnel.state.rejected) if result.funnel is not None else []
    verified = sum(1 for job in [*jobs, *rejected] if getattr(job, "url_verified", False))
    final_jobs = [
        job
        for job in jobs
        if job.direct_application_url
        and "jobright.ai" not in job.direct_application_url.lower()
        and "linkedin.com" not in job.direct_application_url.lower()
    ]
    leaked = len(jobs) - len(final_jobs)
    dates_ok = all(
        posting.date_source.value in {"POSTED_DATE", "UPDATED_DATE", "UNKNOWN"}
        and not (posting.date_source.value == "UNKNOWN" and posting.posted_at is not None)
        for posting in postings
    )
    authoritative_dates = sum(1 for posting in postings if posting.date_source.value in {"POSTED_DATE", "UPDATED_DATE"})
    fresh_candidates = result.funnel.fresh_candidates if result.funnel is not None else 0
    discovery = "BLOCKED — public automated discovery unavailable" if blocked else (
        "PASS" if postings and resolved else ("EMPTY" if report.status == "EMPTY" else report.status)
    )
    if blocked:
        discovery = "BLOCKED — public automated discovery unavailable"
    elif report.status == "FAILED":
        discovery = "FAIL"
    elif not postings:
        discovery = "EMPTY"
    elif resolved:
        discovery = "PARTIAL" if report.cap_reached or resolved < len(postings) else "PASS"
    else:
        discovery = "PARTIAL"
    print("JOBRIGHT SMOKE")
    print(f"Discovery: {discovery}")
    print(f"Candidates discovered: {report.discovered}")
    print(f"Normalized: {report.normalized}")
    print(f"Official URLs resolved: {resolved}")
    print(f"Official URLs verified: {verified}")
    print(f"Canonical jobs: {len(postings)}")
    print(f"Cross-source dedup: {'PASS' if report.deduplicated <= report.discovered else 'FAIL'}")
    freshness = "PASS" if dates_ok and authoritative_dates else ("PARTIAL" if dates_ok and postings else _stage(blocked, bool(postings), dates_ok))
    print(f"Freshness: {freshness}")
    print(f"H1B: {_stage(blocked, fresh_candidates > 0, True)}")
    print(f"Job Intelligence: {_stage(blocked, result.intelligence_evaluated > 0, result.intelligence_evaluated > 0)}")
    print(f"Critic: {_stage(blocked, result.critic_reviewed > 0, result.unsupported_claims == 0)}")
    print(f"Final: {len(final_jobs)}")
    print(
        "LLM: "
        f"jobright calls=0 successes=0 failures=0 retries=0 circuit_skips=0 "
        f"deterministic_avoided=0 fallbacks=0 "
        f"shared_calls={result.llm_calls} shared_state={result.llm_circuit_state}"
    )
    print(f"Caps: cap_reached={str(report.cap_reached).lower()} {report.cap_note}")
    print(
        "Source health: "
        f"{report.status} attempted={report.companies_attempted} "
        f"succeeded={report.companies_succeeded} failed={report.companies_failed} "
        f"errors={len(report.failures)} duration={report.duration_seconds:.1f}s"
    )
    isolated = _isolated_workbook(final_jobs)
    after = production.stat().st_mtime_ns if production.exists() else None
    unchanged = before == after
    print(f"isolated_xlsx: {isolated}")
    print(f"production_workbook_unchanged: {unchanged}")
    print(f"jobright_url_in_final: {leaked > 0}")
    if leaked or not unchanged:
        print("FAIL: Jobright URL reached the workbook or the production file changed")
        return 1
    return 0


async def _probe_board_urls(postings: list[RawJobPosting], config: AppConfig, http: HttpClient, host: str) -> tuple[int, int, int]:
    """Classify every posting and probe at most three official URLs."""
    from src.services.url_verification import pick_direct_url, verify_url

    official = 0
    probed = 0
    verified = 0
    for posting in postings:
        url = posting.apply_url or ""
        if host not in url.lower():
            continue
        official += 1
        if probed >= 3:
            continue
        probed += 1
        check = await verify_url(pick_direct_url(posting, config.url_policy), config, http)
        if check.reachable is True:
            verified += 1
    return official, probed, verified


async def run_board_smoke(config: AppConfig, source: str, *, host: str, title: str) -> int:
    """One public ATS board through the shared pipeline. Dry-run. No production workbook write."""
    config = _smoke_config(config)
    production = config.project_root / config.settings.output.current_workbook
    before = production.stat().st_mtime_ns if production.exists() else None
    http = HttpClient(config)
    provider = build_llm_provider(config)
    official = probed = verified = 0
    try:
        result = await orchestrate(
            config,
            sources=(source,),
            http=http,
            llm=provider,
            load_saved_profile=True,
        )
        official, probed, verified = await _probe_board_urls(result.postings, config, http, host)
    except Exception as exc:
        print(title)
        print(f"Discovery: BLOCKED — {_safe_message(exc)}")
        return 1
    finally:
        close = getattr(provider, "aclose", None)
        if callable(close):
            await close()
        await http.aclose()

    report = result.sources[source]
    blocked = any("BLOCKED" in failure.message or "403" in failure.message for failure in report.failures)
    postings = [posting for posting in result.postings if posting.source == source]
    described = [posting for posting in postings if len(posting.description or "") >= 80]
    with_requirements = [
        posting
        for posting in described
        if any(token in (posting.description or "").lower() for token in ("responsibil", "qualification", "requirement", "you'll", "you will"))
    ]
    with_location = [posting for posting in postings if posting.location_raw]
    with_employment = [posting for posting in postings if posting.employment_type_raw]
    with_id = [posting for posting in postings if posting.job_id and (posting.provenance or {}).get("official_job_id")]
    dates_ok = all(
        posting.date_source.value in {"POSTED_DATE", "UPDATED_DATE", "UNKNOWN"}
        and not (posting.date_source.value == "UNKNOWN" and posting.posted_at is not None)
        for posting in postings
    )
    jobs = list(result.funnel.state.jobs) if result.funnel is not None else []
    leaked = [
        job
        for job in jobs
        if any(token in (job.direct_application_url or "").lower() for token in ("jobright.ai", "linkedin.com", "indeed.com"))
    ]
    discovery = "BLOCKED" if blocked else ("FAIL" if report.status == "FAILED" else ("EMPTY" if not postings else ("PARTIAL" if report.cap_reached else "PASS")))
    print(title)
    print(f"Discovery: {discovery}")
    print(f"Normalization: {_stage(blocked, bool(postings), report.normalized == len(postings) and bool(postings))}")
    print(f"IDs: {_stage(blocked, bool(postings), len(with_id) == len(postings) and bool(postings))}")
    print(f"Description extraction: {_stage(blocked, bool(postings), bool(described))}")
    print(f"Requirements: {_stage(blocked, bool(described), bool(with_requirements))}")
    print(f"Location: {_stage(blocked, bool(postings), bool(with_location))}")
    print(f"Employment: {_stage(blocked, bool(postings), bool(with_employment))}")
    print(f"Freshness: {'PASS' if dates_ok and postings else _stage(blocked, bool(postings), dates_ok)}")
    print(f"Official URL: {_stage(blocked, bool(postings), official == len(postings) and official > 0)}")
    print(f"URL verification: {_stage(blocked, probed > 0, verified == probed and probed > 0)} probed={probed} verified={verified}")
    print(f"Dedup: {'PASS' if report.deduplicated <= report.discovered else 'FAIL'}")
    fresh = result.funnel.fresh_candidates if result.funnel is not None else 0
    print(f"H1B: {_stage(blocked, fresh > 0, True)}")
    print(f"Resume: {'PASS' if result.profile_sha256 == CANONICAL_PROFILE_SHA256 and result.profile_version == 1 else 'FAIL'}")
    print(f"Job Intelligence: {_stage(blocked, result.intelligence_evaluated > 0, result.intelligence_evaluated > 0)}")
    print(f"Critic: {_stage(blocked, result.critic_reviewed > 0, result.unsupported_claims == 0)}")
    print(f"Final: {len(jobs) - len(leaked)}")
    print(_source_gate_line(result, source) or "  gates: unavailable")
    survivors = [job for job in jobs if job not in leaked]
    if survivors:
        print("Live fresh qualification: SURVIVOR")
        for job in survivors:
            evidence = (job.visa_sponsorship_evidence or "UNKNOWN")[:180]
            print(
                "  survivor "
                f"title={job.job_title} id={job.job_id} url={job.direct_application_url} "
                f"url_verified={str(job.url_verified).lower()} "
                f"sponsorship_status={job.visa_sponsorship_status.value} "
                f"sponsorship_confidence={job.visa_sponsorship_confidence} "
                f"sponsorship_source={job.visa_sponsorship_source or 'UNKNOWN'} "
                f"sponsorship_evidence={evidence} "
                f"sponsorship_scope={job.sponsorship_scope.value} "
                f"historical_sponsor_status={job.h1b_historical_sponsor} "
                f"historical_sponsor_match_strength={job.h1b_match_strength} "
                f"last_verified={job.h1b_last_verified} "
                f"evidence_age={job.h1b_evidence_age}"
            )
        for critique in result.fits:
            if critique.fit.job_source != source:
                continue
            print(
                "  intelligence "
                f"required={critique.fit.required_requirements} "
                f"preferred={critique.fit.preferred_requirements}"
            )
            print(
                "  critic "
                f"approved={critique.approved} unsupported={critique.unsupported_claims} "
                f"corrected={critique.corrected_fields} job_id={critique.fit.job_id} "
                f"url={critique.fit.official_url} freshness={critique.fit.freshness}"
            )
    else:
        print(f"Live fresh qualification: NONE {report.cap_note}")
        if result.funnel is not None:
            fresh_rejects = [
                item
                for item in result.funnel.state.rejected
                if item.source == source and item.reason is RejectionReason.FRESHNESS
            ]
            ages = [item.age_hours for item in fresh_rejects if item.age_hours is not None]
            unknown_dates = len(fresh_rejects) - len(ages)
            if ages or unknown_dates:
                oldest = f"{max(ages):.1f}" if ages else "n/a"
                newest = f"{min(ages):.1f}" if ages else "n/a"
                print(
                    "  freshness_reject "
                    f"count={len(fresh_rejects)} unknown_dates={unknown_dates} "
                    f"newest_age_hours={newest} oldest_age_hours={oldest}"
                )
    print(
        "LLM: "
        f"calls={result.llm_calls} successes={result.llm_successes} failures={result.llm_failures} "
        f"retries={result.llm_retries} circuit_skips={result.llm_circuit_skips} "
        f"fallbacks={result.llm_fallbacks} deterministic_avoided={result.llm_deterministic_avoided} "
        f"state={result.llm_circuit_state}"
    )
    print(
        "Source health: "
        f"{report.status} attempted={report.companies_attempted} succeeded={report.companies_succeeded} "
        f"failed={report.companies_failed} discovered={report.discovered} normalized={report.normalized} "
        f"deduplicated={report.deduplicated} rejected={report.rejected} final={report.final} "
        f"errors={len(report.failures)} duration={report.duration_seconds:.1f}s"
    )
    print(f"Caps: cap_reached={str(report.cap_reached).lower()} {report.cap_note}")
    isolated = _isolated_workbook([job for job in jobs if job not in leaked])
    after = production.stat().st_mtime_ns if production.exists() else None
    unchanged = before == after
    print(f"isolated_xlsx: {isolated}")
    print(f"production_workbook_unchanged: {unchanged}")
    print(f"aggregator_url_in_final: {bool(leaked)}")
    if leaked or not unchanged:
        print("FAIL: aggregator URL reached the workbook or the production file changed")
        return 1
    return 0


async def run_lever_smoke(config: AppConfig) -> int:
    return await run_board_smoke(config, "lever", host="lever.co", title="LEVER SMOKE")


async def run_ashby_smoke(config: AppConfig) -> int:
    return await run_board_smoke(config, "ashby", host="ashbyhq.com", title="ASHBY SMOKE")


def _isolated_workbook(jobs) -> str:
    with tempfile.TemporaryDirectory(prefix="multi-source-") as directory:
        root = Path(directory)
        current, _archive = write_workbooks(
            jobs,
            current_path=root / "jobs.xlsx",
            archive_dir=root / "archive",
            existing_rows=[],
            write_archive=True,
        )
        existed = current.exists()
    return "written-and-removed" if existed else "failed"

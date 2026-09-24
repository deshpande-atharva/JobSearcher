"""Per-company discovery and coverage reports built from real pipeline data."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.models.job import DateSource, RejectionReason
from src.services.freshness import is_fresh
from src.utils.normalization import normalize_company_name

if TYPE_CHECKING:
    from src.models.state import PipelineState

__all__ = [
    "CompanyDiscoveryRow",
    "CoverageMetrics",
    "FreshnessSourceStats",
    "attach_discovery_reports",
    "render_source_summary",
]

_ATS_SOURCES = frozenset(
    {"greenhouse", "lever", "ashby", "smartrecruiters", "workday", "icims"}
)


class CompanyDiscoveryRow(BaseModel):
    company: str
    ats: str = "Unknown"
    identifier: str = "-"
    method: str = "none"
    source: str = "-"
    raw: int = 0
    swe: int = 0
    exp: int = 0
    fresh: int = 0
    final: int = 0
    ats_result: str = "-"
    career_result: str = "-"
    fallback_used: bool = False
    failure: str | None = None
    structured_status: str = "-"
    structured_http: int | None = None
    structured_jobs: int = 0
    fallback_status: str = "-"
    fallback_jobs: int = 0
    list_cap_reached: bool = False


class CoverageMetrics(BaseModel):
    companies_configured: int = 0
    companies_attempted: int = 0
    companies_successfully_discovered: int = 0
    companies_with_verified_ats: int = 0
    companies_with_automatic_ats: int = 0
    companies_with_manual_ats: int = 0
    companies_with_no_ats: int = 0
    companies_with_career_fallback: int = 0
    companies_with_complete_failure: int = 0
    ats_greenhouse: int = 0
    ats_lever: int = 0
    ats_ashby: int = 0
    ats_workday: int = 0
    ats_smartrecruiters: int = 0
    ats_icims: int = 0
    ats_unknown: int = 0


class FreshnessSourceStats(BaseModel):
    source: str
    discovered: int = 0
    timestamp_known: int = 0
    within_window: int = 0
    timestamp_unknown: int = 0


def attach_discovery_reports(state: "PipelineState") -> None:
    """Fill summary report fields from actual pipeline objects."""
    rows = _company_rows(state)
    state.summary.company_discovery_rows = rows
    state.summary.coverage = _coverage(state, rows)
    if not state.summary.freshness_by_source:
        state.summary.freshness_by_source = _freshness_from_raw(state)
    deduped: dict[str, int] = {}
    for posting in state.raw_postings:
        source = posting.source or "unknown"
        deduped[source] = deduped.get(source, 0) + 1
    state.summary.jobs_by_source_deduped = deduped
    state.summary.workday_funnel = _workday_funnel(state)
    state.summary.workday_samples = _workday_samples(state)
    state.summary.accepted_job_audit = _accepted_audit(state)


def _workday_funnel(state: "PipelineState") -> dict[str, int]:
    """Stage counts for Workday jobs only. One rejection reason per job."""
    health = state.summary.source_health.get("workday")
    if health is None:
        return {}
    rejected = [item for item in state.rejected if (item.source or "") == "workday"]
    accepted = [job for job in state.jobs if (job.source or "") == "workday"]
    counts: dict[RejectionReason, int] = defaultdict(int)
    for item in rejected:
        counts[item.reason] += 1
    extracted = len(rejected) + len(accepted)
    role = extracted - counts[RejectionReason.ROLE] - counts[RejectionReason.EXTRACTION_FAILED]
    experience = role - counts[RejectionReason.SENIORITY]
    us = experience - counts[RejectionReason.LOCATION]
    employment = us - counts[RejectionReason.EMPLOYMENT_TYPE]
    fresh = employment - counts[RejectionReason.FRESHNESS]
    url = fresh - counts[RejectionReason.INVALID_URL]
    return {
        "discovered": health.jobs_discovered,
        "role": max(role, 0),
        "experience": max(experience, 0),
        "us": max(us, 0),
        "employment": max(employment, 0),
        "fresh": max(fresh, 0),
        "url": max(url, 0),
        "final": len(accepted),
    }


_SAMPLE_TITLE_HINTS = (
    "senior software",
    "staff software",
    "principal software",
    "software engineer ii",
    "software engineer iii",
    "site reliability",
    "platform engineer",
    "infrastructure engineer",
    "software development engineer",
)


def _workday_samples(state: "PipelineState") -> list[dict[str, str]]:
    """A few Workday rejections so a zero final count can be checked by hand."""
    rejected = [item for item in state.rejected if (item.source or "") == "workday"]
    if not rejected:
        return []
    raw = [posting for posting in state.raw_postings if (posting.source or "") == "workday"]
    chosen: list = []
    seen: set[tuple[str, str]] = set()

    def consider(item) -> None:
        key = (item.company, item.title)
        if key in seen or len(chosen) >= 8:
            return
        title = (item.title or "").lower()
        if any(hint in title for hint in _SAMPLE_TITLE_HINTS) or item.reason in {
            RejectionReason.FRESHNESS,
            RejectionReason.LOCATION,
            RejectionReason.INVALID_URL,
            RejectionReason.ROLE,
        }:
            seen.add(key)
            chosen.append(item)

    for hint in _SAMPLE_TITLE_HINTS:
        for item in rejected:
            if hint in (item.title or "").lower():
                consider(item)
                break
    for reason in (
        RejectionReason.SENIORITY,
        RejectionReason.ROLE,
        RejectionReason.LOCATION,
        RejectionReason.FRESHNESS,
        RejectionReason.INVALID_URL,
    ):
        for item in rejected:
            if item.reason is reason:
                consider(item)
                break

    samples: list[dict[str, str]] = []
    for item in chosen:
        match = next(
            (
                posting
                for posting in raw
                if posting.company_name == item.company
                and (posting.apply_url == item.url or posting.title == item.title)
            ),
            None,
        )
        posted = None
        job_id = None
        if match is not None:
            posted = match.posted_at_raw or (match.posted_at.isoformat() if match.posted_at else None)
            job_id = match.job_id
        samples.append(
            {
                "company": item.company,
                "title": item.title,
                "job_id": job_id or "",
                "posted": posted or "",
                "date_source": item.date_source.value if item.date_source else "",
                "reason": item.reason.value if hasattr(item.reason, "value") else str(item.reason),
                "detail": item.detail or "",
            }
        )
    return samples


def _accepted_audit(state: "PipelineState") -> list[dict[str, str]]:
    from src.utils.dates import format_datetime

    rows: list[dict[str, str]] = []
    for job in state.jobs:
        rows.append(
            {
                "company": job.company,
                "title": job.job_title,
                "job_id": job.job_id or "",
                "posted": format_datetime(job.posted_at) if job.posted_at else "",
                "date_source": job.date_source.value if job.date_source else "",
                "location": job.location or "",
                "employment": job.employment_type.value if job.employment_type else "",
                "source": job.source or "",
                "url": job.direct_application_url or "",
                "sponsorship": job.sponsorship_display,
            }
        )
    return rows


def _company_rows(state: "PipelineState") -> list[CompanyDiscoveryRow]:
    attempted = list(state.config.target_companies())
    rejected_by_company: dict[str, list] = defaultdict(list)
    for item in state.rejected:
        rejected_by_company[normalize_company_name(item.company)].append(item)
    accepted_by_company: dict[str, list] = defaultdict(list)
    for job in state.jobs:
        accepted_by_company[normalize_company_name(job.company)].append(job)

    rows: list[CompanyDiscoveryRow] = []
    for company in attempted:
        key = normalize_company_name(company.name)
        outcomes = [o for o in state.company_outcomes if normalize_company_name(o.company) == key]
        ats_outcomes = [o for o in outcomes if o.source in _ATS_SOURCES]
        career_outcomes = [o for o in outcomes if o.source == "company_career"]
        ats_outcome = ats_outcomes[0] if ats_outcomes else None
        career_outcome = career_outcomes[0] if career_outcomes else None

        raw = sum(o.jobs_found for o in outcomes)
        rejected = rejected_by_company.get(key, [])
        accepted = accepted_by_company.get(key, [])
        swe, exp, fresh = _funnel_counts(rejected, accepted)

        ats_label = "Unknown"
        identifier = "-"
        method = "none"
        if ats_outcome and ats_outcome.ats_type:
            ats_label = ats_outcome.ats_type.title() if ats_outcome.ats_type != "icims" else "iCIMS"
            if ats_outcome.ats_type == "smartrecruiters":
                ats_label = "SmartRecruiters"
            identifier = ats_outcome.ats_identifier or "-"
            method = ats_outcome.detection_method or "none"
        elif company.has_structured_discovery:
            ats_label = company.ats_type or "Unknown"
            identifier = company.ats_identifier or "-"
            method = company.ats_discovery_mode

        primary = "-"
        if ats_outcome and ats_outcome.succeeded and ats_outcome.jobs_found:
            primary = "ATS"
        elif career_outcome and career_outcome.succeeded and career_outcome.jobs_found:
            primary = "Career"
        elif ats_outcome and ats_outcome.succeeded:
            primary = "ATS"
        elif career_outcome and career_outcome.succeeded:
            primary = "Career"
        elif outcomes:
            primary = outcomes[0].source

        fallback = bool(
            (career_outcome and (getattr(career_outcome, "fallback_used", False) or career_outcome.jobs_found))
            and ats_outcome is not None
        )
        if career_outcome and ats_outcome and (not ats_outcome.succeeded or ats_outcome.jobs_found == 0):
            fallback = True

        failure = None
        if outcomes and not any(o.succeeded for o in outcomes):
            failure = "NO_ACCESSIBLE_DISCOVERY_SOURCE"
            errors = [o.error for o in outcomes if o.error]
            if errors:
                failure = errors[-1]
        elif ats_outcome and not ats_outcome.succeeded and ats_outcome.error:
            failure = ats_outcome.error

        structured_status, structured_http, structured_jobs = _structured_fields(ats_outcome)
        fallback_status, fallback_jobs = _fallback_fields(career_outcome, ats_outcome)
        rows.append(
            CompanyDiscoveryRow(
                company=company.name,
                ats=ats_label if isinstance(ats_label, str) else "Unknown",
                identifier=identifier if len(identifier) <= 40 else identifier[:37] + "...",
                method=method,
                source=primary,
                raw=raw,
                swe=swe,
                exp=exp,
                fresh=fresh,
                final=len(accepted),
                ats_result=_result_label(ats_outcome),
                career_result=_result_label(career_outcome),
                fallback_used=fallback,
                failure=failure,
                structured_status=structured_status,
                structured_http=structured_http,
                structured_jobs=structured_jobs,
                fallback_status=fallback_status,
                fallback_jobs=fallback_jobs,
                list_cap_reached=bool((getattr(ats_outcome, "diagnostics", None) or {}).get("source_list_cap_reached")),
            )
        )
    return rows


def _funnel_counts(rejected: list, accepted: list) -> tuple[int, int, int]:
    """SWE / 0-2yr / Fresh remaining counts from extracted jobs for one company."""
    extracted = len(rejected) + len(accepted)
    role = sum(1 for r in rejected if r.reason is RejectionReason.ROLE)
    extraction = sum(1 for r in rejected if r.reason is RejectionReason.EXTRACTION_FAILED)
    seniority = sum(1 for r in rejected if r.reason is RejectionReason.SENIORITY)
    location = sum(1 for r in rejected if r.reason is RejectionReason.LOCATION)
    employment = sum(1 for r in rejected if r.reason is RejectionReason.EMPLOYMENT_TYPE)
    freshness = sum(1 for r in rejected if r.reason is RejectionReason.FRESHNESS)
    swe = max(extracted - extraction - role, 0)
    exp = max(swe - seniority, 0)
    fresh = max(exp - location - employment - freshness, 0)
    return swe, exp, fresh


def _result_label(outcome) -> str:
    if outcome is None:
        return "-"
    if not outcome.succeeded:
        return f"failed ({outcome.error or 'error'})"
    if outcome.jobs_found == 0:
        return "returned 0"
    return f"success ({outcome.jobs_found})"


def _structured_fields(outcome) -> tuple[str, int | None, int]:
    if outcome is None:
        return "-", None, 0
    diagnostics = getattr(outcome, "diagnostics", None) or {}
    raw_status = getattr(outcome, "status", None) or ""
    if raw_status:
        status = raw_status
    elif outcome.succeeded:
        status = "OK" if outcome.jobs_found else "EMPTY"
    else:
        status = "ERROR"
    if diagnostics.get("fallback_used") and diagnostics.get("failure_reason") and outcome.succeeded:
        status = "ERROR"
    http = getattr(outcome, "http_status", None)
    if http is None:
        http = diagnostics.get("http_status")
    jobs = diagnostics.get("cxs_jobs")
    if jobs is None:
        jobs = 0 if diagnostics.get("fallback_used") else outcome.jobs_found
    return str(status), http if isinstance(http, int) else None, int(jobs or 0)


def classify_if_needed(status: str, outcome) -> str:
    if status in {"ERROR", "BLOCKED", "UNSUPPORTED"}:
        return status
    from src.sources.base import classify_source_status

    return classify_source_status(getattr(outcome, "error", None), http_status=getattr(outcome, "http_status", None))


def _fallback_fields(career_outcome, ats_outcome) -> tuple[str, int]:
    if career_outcome is not None:
        status = getattr(career_outcome, "status", None)
        if not status:
            status = "OK" if career_outcome.succeeded and career_outcome.jobs_found else (
                "EMPTY" if career_outcome.succeeded else "ERROR"
            )
        return str(status), career_outcome.jobs_found
    if ats_outcome is not None and getattr(ats_outcome, "fallback_used", False):
        status = getattr(ats_outcome, "fallback_status", None) or (
            "OK" if ats_outcome.succeeded and ats_outcome.jobs_found else "EMPTY"
        )
        jobs = (getattr(ats_outcome, "diagnostics", None) or {}).get("fallback_jobs")
        if jobs is None:
            jobs = ats_outcome.jobs_found if ats_outcome.succeeded else 0
        return str(status), int(jobs or 0)
    return "not needed", 0


def _coverage(state: "PipelineState", rows: list[CompanyDiscoveryRow]) -> CoverageMetrics:
    configured = len(state.config.universe.companies)
    attempted = list(state.config.target_companies())
    metrics = CoverageMetrics(
        companies_configured=configured,
        companies_attempted=len(attempted),
        companies_successfully_discovered=sum(1 for row in rows if row.raw > 0 or row.ats_result.startswith("success") or row.ats_result == "returned 0" or row.career_result.startswith("success") or row.career_result == "returned 0"),
        companies_with_complete_failure=sum(1 for row in rows if row.failure == "NO_ACCESSIBLE_DISCOVERY_SOURCE" or (row.ats_result.startswith("failed") and row.career_result in {"-", *()} and not row.raw)),
        companies_with_career_fallback=sum(1 for row in rows if row.fallback_used),
    )
    # Refine complete-failure: every attempted source failed.
    failed = 0
    for row in rows:
        ats_failed = row.ats_result.startswith("failed") or row.ats_result == "-"
        career_failed = row.career_result.startswith("failed") or row.career_result == "-"
        if ats_failed and career_failed and row.raw == 0:
            failed += 1
    metrics.companies_with_complete_failure = failed

    for company in attempted:
        if company.has_structured_discovery:
            metrics.companies_with_verified_ats += 1
            if company.ats_discovery_mode == "manual":
                metrics.companies_with_manual_ats += 1
            else:
                metrics.companies_with_automatic_ats += 1
            _count_ats(metrics, company.ats_type)
        else:
            row = next((r for r in rows if r.company == company.name), None)
            if row and row.ats != "Unknown":
                metrics.companies_with_automatic_ats += 1
                metrics.companies_with_verified_ats += 1
                _count_ats(metrics, row.ats.lower())
            else:
                metrics.companies_with_no_ats += 1
                metrics.ats_unknown += 1
    return metrics


def _count_ats(metrics: CoverageMetrics, ats_type: str | None) -> None:
    mapping = {
        "greenhouse": "ats_greenhouse",
        "lever": "ats_lever",
        "ashby": "ats_ashby",
        "workday": "ats_workday",
        "smartrecruiters": "ats_smartrecruiters",
        "icims": "ats_icims",
    }
    key = mapping.get((ats_type or "").lower())
    if key:
        setattr(metrics, key, getattr(metrics, key) + 1)
    else:
        metrics.ats_unknown += 1


def _freshness_from_raw(state: "PipelineState") -> dict[str, FreshnessSourceStats]:
    hours = state.config.freshness_hours
    buckets: dict[str, FreshnessSourceStats] = {}
    for posting in state.raw_postings:
        source = posting.source or "unknown"
        stats = buckets.setdefault(source, FreshnessSourceStats(source=source))
        stats.discovered += 1
        unknown = posting.date_source is DateSource.UNKNOWN or (
            posting.posted_at is None and posting.updated_at is None
        )
        if unknown:
            stats.timestamp_unknown += 1
            continue
        stats.timestamp_known += 1
        if is_fresh(posting, hours)[0]:
            stats.within_window += 1
    return buckets


def render_company_discovery_report(rows: list[CompanyDiscoveryRow]) -> str:
    lines = [
        "COMPANY DISCOVERY REPORT",
        "========================",
        f"{'Company':<24} {'ATS':<14} {'Source':<8} {'Raw':>5} {'SWE':>5} {'0-2yr':>7} {'Fresh':>7} {'Final':>7}",
        "-" * 84,
    ]
    if not rows:
        lines.append("(no companies attempted)")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"{row.company[:24]:<24} {row.ats[:14]:<14} {row.source[:8]:<8} "
            f"{row.raw:5d} {row.swe:5d} {row.exp:7d} {row.fresh:7d} {row.final:7d}"
        )
    return "\n".join(lines)


def render_coverage(metrics: CoverageMetrics) -> str:
    lines = [
        "DISCOVERY COVERAGE",
        "==================",
        f"Companies configured              : {metrics.companies_configured}",
        f"Companies attempted               : {metrics.companies_attempted}",
        f"Companies successfully discovered : {metrics.companies_successfully_discovered}",
        f"Companies with verified ATS       : {metrics.companies_with_verified_ats}",
        f"Companies with automatically detected ATS : {metrics.companies_with_automatic_ats}",
        f"Companies with manual ATS         : {metrics.companies_with_manual_ats}",
        f"Companies with no ATS             : {metrics.companies_with_no_ats}",
        f"Companies with career-page fallback : {metrics.companies_with_career_fallback}",
        f"Companies with complete discovery failure : {metrics.companies_with_complete_failure}",
        "ATS coverage:",
        f"  Greenhouse        : {metrics.ats_greenhouse}",
        f"  Lever             : {metrics.ats_lever}",
        f"  Ashby             : {metrics.ats_ashby}",
        f"  Workday           : {metrics.ats_workday}",
        f"  SmartRecruiters   : {metrics.ats_smartrecruiters}",
        f"  iCIMS             : {metrics.ats_icims}",
        f"  Unknown           : {metrics.ats_unknown}",
    ]
    return "\n".join(lines)


def render_freshness_by_source(buckets: dict[str, FreshnessSourceStats], hours: float) -> str:
    lines = [
        "FRESHNESS BY SOURCE",
        "===================",
        f"(window = {hours:g} hours; unknown timestamps are not treated as fresh)",
    ]
    if not buckets:
        lines.append("(no raw postings)")
        return "\n".join(lines)
    for name in sorted(buckets):
        stats = buckets[name]
        lines.append(name)
        lines.append(f"  discovered: {stats.discovered}")
        lines.append(f"  timestamp known: {stats.timestamp_known}")
        lines.append(f"  within {hours:g}h: {stats.within_window}")
        lines.append(f"  timestamp unknown: {stats.timestamp_unknown}")
    return "\n".join(lines)


def render_company_detail(row: CompanyDiscoveryRow) -> str:
    structured = row.structured_status if row.structured_status != "-" else row.ats_result
    fallback = row.fallback_status
    lines = [
        f"Company: {row.company}",
        f"ATS: {row.ats}",
        "Structured source:",
        f"    status: {structured}",
    ]
    if row.structured_http is not None:
        lines.append(f"    HTTP: {row.structured_http}")
    lines.append(f"    jobs: {row.structured_jobs}")
    lines.append("Fallback:")
    if fallback in {"-", "not needed"}:
        lines.append("    not needed")
    else:
        lines.append(f"    status: {fallback}")
        lines.append(f"    jobs: {row.fallback_jobs}")
    lines.append("Final discovered:")
    lines.append(f"    {row.raw}")
    if row.list_cap_reached:
        lines.append(
            f"    source/list cap may have been reached; inventory may be incomplete (discovered {row.raw})"
        )
    if row.failure:
        lines.append(f"Failure reason: {row.failure}")
    return "\n".join(lines)


def render_source_summary(summary) -> str:
    """Aggregate source metrics from recorded health + deduped job counts."""
    lines = ["SOURCE SUMMARY", "=============="]
    health = summary.source_health or {}
    deduped = getattr(summary, "jobs_by_source_deduped", {}) or {}
    if not health:
        lines.append("(no sources attempted)")
        return "\n".join(lines)

    def _label(name: str) -> str:
        return {
            "greenhouse": "Greenhouse",
            "lever": "Lever",
            "ashby": "Ashby",
            "workday": "Workday",
            "jobright": "Jobright",
            "smartrecruiters": "SmartRecruiters",
            "icims": "iCIMS",
            "company_career": "Career-page fallback",
            "h1bgrader": "H1BGrader",
        }.get(name, name)

    for name in sorted(health):
        h = health[name]
        lines.append(_label(name))
        lines.append(f"  companies: {h.attempted}")
        raw_jobs = h.jobs_discovered
        lines.append(f"  jobs: {raw_jobs}")
        if name in deduped:
            lines.append(f"  deduped jobs: {deduped[name]}")
        if name == "workday":
            lines.append(f"  structured jobs: {raw_jobs}")
            lines.append(f"  HTTP 400: {h.http_400}")
            fallback = health.get("company_career")
            if fallback:
                lines.append(f"  fallback jobs: {fallback.jobs_discovered}")
        if name == "jobright":
            lines.append(f"  parsed jobs: {raw_jobs}")
        lines.append(f"  status: {_source_health_label(h)}")
        if h.errors:
            lines.append(f"  last error: {h.errors[-1]}")
    return "\n".join(lines)


def _source_health_label(health) -> str:
    if health.blocked and health.blocked == health.failed and health.successful == 0:
        return "blocked"
    if health.unsupported and health.successful == 0 and health.error == 0:
        return "unsupported"
    if health.failed == 0 and health.jobs_discovered > 0:
        return "healthy"
    if health.failed == 0 and health.empty == health.attempted:
        return "empty"
    if health.failed and health.jobs_discovered > 0:
        return "degraded"
    if health.failed:
        return "error"
    return "empty"

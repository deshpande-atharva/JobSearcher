"""Shared pipeline state and run reporting.

:class:`PipelineState` is the typed state object LangGraph threads through every
node. Agents receive it, return a dict of changed keys, and never reach outside
it for mutable data.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.job import (
    DateSource,
    Job,
    RawJobPosting,
    RejectionReason,
    VisaSponsorshipStatus,
)
from src.utils.dates import format_datetime, utcnow

if TYPE_CHECKING:  # avoid a circular import at runtime
    from src.models.config import AppConfig
    from src.services.discovery_report import CompanyDiscoveryRow, CoverageMetrics, FreshnessSourceStats

__all__ = [
    "CompanyOutcome",
    "PipelineState",
    "RejectedJob",
    "RunSummary",
    "SourceHealth",
]


class CompanyOutcome(BaseModel):
    """Per-company discovery result.

    One company failing is expected and never fatal; the failure is recorded
    here and surfaced in the run summary. ``succeeded=True`` with
    ``jobs_found=0`` means the source was reachable and returned nothing.
    """

    company: str
    source: str
    succeeded: bool
    jobs_found: int = 0
    error: str | None = None
    duration_seconds: float | None = None
    ats_type: str | None = None
    ats_identifier: str | None = None
    detection_method: str | None = None
    fallback_used: bool = False
    status: str = ""
    http_status: int | None = None
    fallback_status: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SourceHealth(BaseModel):
    """Aggregated per-source discovery health."""

    source: str
    attempted: int = 0
    successful: int = 0
    failed: int = 0
    empty: int = 0
    error: int = 0
    blocked: int = 0
    unsupported: int = 0
    jobs_discovered: int = 0
    http_400: int = 0
    errors: list[str] = Field(default_factory=list)


class RejectedJob(BaseModel):
    """A posting that left the pipeline, kept for the run report.

    Sponsorship is never a rejection reason -- see :class:`RejectionReason`.
    """

    company: str
    title: str
    reason: RejectionReason
    detail: str | None = None
    url: str | None = None
    source: str | None = None
    date_source: DateSource | None = None
    age_hours: float | None = None
    experience_category: str | None = None


class RunSummary(BaseModel):
    """Counters and outcomes for one pipeline run."""

    model_config = ConfigDict(validate_assignment=True)

    run_started_at: datetime = Field(default_factory=utcnow)
    run_finished_at: datetime | None = None
    dry_run: bool = False
    fixture_mode: bool = False

    # --- discovery ----------------------------------------------------------
    companies_attempted: int = 0
    companies_succeeded: int = 0
    companies_failed: int = 0
    failed_companies: list[str] = Field(default_factory=list)
    sources_attempted: list[str] = Field(default_factory=list)
    failed_sources: list[str] = Field(default_factory=list)

    jobs_discovered: int = 0
    jobs_after_cross_source_dedup: int = 0
    jobs_processed: int = 0
    jobs_truncated: int = 0
    jobs_extracted: int = 0
    jobs_accepted: int = 0
    unknown_timestamps: int = 0
    freshness_hours_used: float = 24.0
    source_health: dict[str, SourceHealth] = Field(default_factory=dict)
    empty_sources: list[str] = Field(default_factory=list)
    company_discovery_rows: list[Any] = Field(default_factory=list)
    coverage: Any | None = None
    freshness_by_source: dict[str, Any] = Field(default_factory=dict)
    freshness_breakdown: dict[str, int] = Field(default_factory=dict)
    nearest_freshness_misses: list[dict[str, Any]] = Field(default_factory=list)
    experience_filter_report: dict[str, int] = Field(default_factory=dict)
    jobs_by_source_deduped: dict[str, int] = Field(default_factory=dict)
    workday_funnel: dict[str, int] = Field(default_factory=dict)
    workday_samples: list[dict[str, Any]] = Field(default_factory=list)
    accepted_job_audit: list[dict[str, Any]] = Field(default_factory=list)

    # --- eligibility rejections (never sponsorship) -------------------------
    rejected_by_role: int = 0
    rejected_by_seniority: int = 0
    rejected_by_location: int = 0
    rejected_by_employment_type: int = 0
    rejected_by_freshness: int = 0
    rejected_by_invalid_url: int = 0
    rejected_by_extraction: int = 0
    rejected_by_quality_control: int = 0

    duplicates_removed: int = 0
    new_jobs: int = 0
    historical_jobs_skipped: int = 0

    # --- H-1B evidence distribution (informational) -------------------------
    h1b_confirmed: int = 0
    h1b_likely: int = 0
    h1b_unknown: int = 0
    h1b_not_supported: int = 0
    h1b_lookup_failures: int = 0

    # --- LLM ----------------------------------------------------------------
    llm_calls: int = 0
    llm_failures: int = 0
    llm_enabled: bool = False

    # --- output -------------------------------------------------------------
    workbook_path: str | None = None
    archive_path: str | None = None
    email_status: str = "not attempted"
    warnings: list[str] = Field(default_factory=list)

    def note(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def count_rejection(self, reason: RejectionReason, amount: int = 1) -> None:
        """Increment the counter matching ``reason``.

        Mapping is explicit so a new rejection reason cannot silently vanish
        from the report.
        """
        mapping = {
            RejectionReason.ROLE: "rejected_by_role",
            RejectionReason.SENIORITY: "rejected_by_seniority",
            RejectionReason.LOCATION: "rejected_by_location",
            RejectionReason.EMPLOYMENT_TYPE: "rejected_by_employment_type",
            RejectionReason.FRESHNESS: "rejected_by_freshness",
            RejectionReason.INVALID_URL: "rejected_by_invalid_url",
            RejectionReason.EXTRACTION_FAILED: "rejected_by_extraction",
            RejectionReason.QUALITY_CONTROL: "rejected_by_quality_control",
            RejectionReason.DUPLICATE: "duplicates_removed",
            RejectionReason.ALREADY_SEEN: "historical_jobs_skipped",
        }
        attribute = mapping[reason]
        setattr(self, attribute, getattr(self, attribute) + amount)

    def count_sponsorship(self, status: VisaSponsorshipStatus) -> None:
        """Tally the sponsorship distribution. Purely a report, not a filter."""
        mapping = {
            VisaSponsorshipStatus.CONFIRMED: "h1b_confirmed",
            VisaSponsorshipStatus.LIKELY: "h1b_likely",
            VisaSponsorshipStatus.UNKNOWN: "h1b_unknown",
            VisaSponsorshipStatus.NOT_SUPPORTED: "h1b_not_supported",
        }
        attribute = mapping[status]
        setattr(self, attribute, getattr(self, attribute) + 1)

    @property
    def duration_seconds(self) -> float | None:
        if self.run_finished_at is None:
            return None
        return (self.run_finished_at - self.run_started_at).total_seconds()

    @property
    def total_rejected(self) -> int:
        return (
            self.rejected_by_role
            + self.rejected_by_seniority
            + self.rejected_by_location
            + self.rejected_by_employment_type
            + self.rejected_by_freshness
            + self.rejected_by_invalid_url
            + self.rejected_by_extraction
            + self.rejected_by_quality_control
        )

    def as_lines(self) -> list[str]:
        """Render the summary for the console and the notification email."""
        duration = self.duration_seconds
        lines = [
            f"Run timestamp        : {format_datetime(self.run_started_at)}",
            f"Duration             : {duration:.1f}s" if duration is not None else "Duration             : n/a",
            f"Mode                 : {'dry-run' if self.dry_run else 'live'}"
            + (" / fixture-mode" if self.fixture_mode else ""),
            "",
            "-- Discovery --------------------------------------------------",
            f"Companies attempted  : {self.companies_attempted}",
            f"Companies successful : {self.companies_succeeded}",
            f"Companies failed     : {self.companies_failed}",
            f"Jobs discovered      : {self.jobs_discovered}",
            f"Deduplicated         : {self.jobs_after_cross_source_dedup or self.jobs_discovered}",
            f"Processed            : {self.jobs_processed or self.jobs_after_cross_source_dedup or self.jobs_discovered}",
            f"Truncated            : {self.jobs_truncated}",
            f"Jobs extracted       : {self.jobs_extracted}",
            f"Unknown timestamps   : {self.unknown_timestamps}",
            "",
            "-- Eligibility ------------------------------------------------",
            f"Jobs accepted        : {self.jobs_accepted}",
            f"Rejected by role     : {self.rejected_by_role}",
            f"Rejected by seniority: {self.rejected_by_seniority}",
            f"Rejected by location : {self.rejected_by_location}",
            f"Rejected by emp type : {self.rejected_by_employment_type}",
            f"Rejected by freshness: {self.rejected_by_freshness}",
            f"Rejected invalid URL : {self.rejected_by_invalid_url}",
            f"Rejected by QC       : {self.rejected_by_quality_control}",
            f"Duplicates removed   : {self.duplicates_removed}",
            f"New jobs             : {self.new_jobs}",
            f"Historical skipped   : {self.historical_jobs_skipped}",
            "",
            "-- H-1B evidence (informational, never a filter) --------------",
            f"Confirmed            : {self.h1b_confirmed}",
            f"Likely               : {self.h1b_likely}",
            f"Unknown              : {self.h1b_unknown}",
            f"Not Supported        : {self.h1b_not_supported}",
            f"Lookup failures      : {self.h1b_lookup_failures}",
            "",
            "-- Output -----------------------------------------------------",
            f"Workbook             : {self.workbook_path or 'not written'}",
            f"Archive              : {self.archive_path or 'not written'}",
            f"Email                : {self.email_status}",
            f"LLM calls / failures : {self.llm_calls} / {self.llm_failures}"
            + ("" if self.llm_enabled else "  (LLM disabled)"),
        ]
        lines += ["", self.discovery_health_report(), "", self.source_summary_report(), "", self.filter_funnel()]
        coverage = self.coverage_report()
        if coverage:
            lines += ["", coverage]
        freshness = self.freshness_source_report()
        if freshness:
            lines += ["", freshness]
        breakdown = self.freshness_candidate_report()
        if breakdown:
            lines += ["", breakdown]
        misses = self.nearest_misses_report()
        if misses:
            lines += ["", misses]
        experience = self.experience_report_text()
        if experience:
            lines += ["", experience]
        why = self.why_zero_report()
        if why:
            lines += ["", why]
        company_report = self.company_discovery_report()
        if company_report:
            lines += ["", company_report]
        workday = self.workday_funnel_report()
        if workday:
            lines += ["", workday]
        accepted = self.accepted_jobs_report()
        if accepted:
            lines += ["", accepted]
        if self.failed_companies:
            lines += ["", "Failed companies:"]
            lines += [f"  - {name}" for name in self.failed_companies[:40]]
        if self.failed_sources:
            lines += ["", "Failed sources:"]
            lines += [f"  - {name}" for name in self.failed_sources]
        if self.warnings:
            lines += ["", "Warnings:"]
            lines += [f"  - {message}" for message in self.warnings[:40]]
        return lines

    def render(self) -> str:
        return "\n".join(self.as_lines())

    def record_source(
        self,
        source: str,
        *,
        success: bool,
        jobs: int,
        error: str | None = None,
        company: str | None = None,
        status: str | None = None,
        http_status: int | None = None,
    ) -> None:
        health = self.source_health.setdefault(source, SourceHealth(source=source))
        health.attempted += 1
        resolved: Literal["OK", "EMPTY", "ERROR", "BLOCKED", "UNSUPPORTED"]
        if status in {"OK", "EMPTY", "ERROR", "BLOCKED", "UNSUPPORTED"}:
            resolved = status  # type: ignore[assignment]
        elif success:
            resolved = "OK" if jobs else "EMPTY"
        else:
            from src.sources.base import classify_source_status

            resolved = classify_source_status(error, http_status=http_status)
        if http_status == 400:
            health.http_400 += 1
        if resolved == "OK":
            health.successful += 1
            health.jobs_discovered += jobs
        elif resolved == "EMPTY":
            health.successful += 1
            health.empty += 1
            label = f"{source}/{company}" if company and company != "*" else source
            if label not in self.empty_sources:
                self.empty_sources.append(label)
        elif resolved == "BLOCKED":
            health.failed += 1
            health.blocked += 1
            if error:
                health.errors.append(error[:240])
        elif resolved == "UNSUPPORTED":
            health.failed += 1
            health.unsupported += 1
            if error:
                health.errors.append(error[:240])
        else:
            health.failed += 1
            health.error += 1
            if error:
                health.errors.append(error[:240])

    def remaining_after(self, *reasons: str) -> int:
        start = self.jobs_extracted
        mapping = {
            "role": self.rejected_by_role,
            "seniority": self.rejected_by_seniority,
            "location": self.rejected_by_location,
            "employment": self.rejected_by_employment_type,
            "freshness": self.rejected_by_freshness,
            "url": self.rejected_by_invalid_url,
        }
        return max(start - sum(mapping[r] for r in reasons), 0)

    def discovery_health_report(self) -> str:
        lines = [
            "DISCOVERY HEALTH",
            "================",
        ]
        if not self.source_health:
            lines.append("(no sources attempted)")
            return "\n".join(lines)
        for name in sorted(self.source_health):
            h = self.source_health[name]
            lines.append(name)
            lines.append(f"  attempted: {h.attempted}")
            lines.append(f"  successful: {h.successful}")
            lines.append(f"  failed: {h.failed}")
            lines.append(f"  empty (success, 0 jobs): {h.empty}")
            lines.append(f"  error: {h.error}")
            lines.append(f"  blocked: {h.blocked}")
            lines.append(f"  unsupported: {h.unsupported}")
            lines.append(f"  jobs discovered: {h.jobs_discovered}")
            if h.http_400:
                lines.append(f"  HTTP 400: {h.http_400}")
            if h.errors:
                lines.append(f"  last error: {h.errors[-1]}")
        return "\n".join(lines)

    def source_summary_report(self) -> str:
        from src.services.discovery_report import render_source_summary

        return render_source_summary(self)

    def filter_funnel(self) -> str:
        after_role = self.remaining_after("role")
        after_seniority = self.remaining_after("role", "seniority")
        after_location = self.remaining_after("role", "seniority", "location")
        after_emp = self.remaining_after("role", "seniority", "location", "employment")
        after_fresh = self.remaining_after("role", "seniority", "location", "employment", "freshness")
        after_url = self.remaining_after(
            "role", "seniority", "location", "employment", "freshness", "url"
        )
        raw = self.jobs_discovered
        deduped = self.jobs_after_cross_source_dedup or raw
        processed = self.jobs_processed or (deduped - self.jobs_truncated)
        lines = [
            "FILTER FUNNEL",
            "=============",
            f"Discovered: {raw}",
            f"Deduplicated: {deduped}",
            f"Processed: {processed}",
            f"Truncated: {self.jobs_truncated}",
            f"{raw} discovered",
            f"- {max(raw - deduped, 0)} cross-source duplicates",
            f"- {self.jobs_truncated} truncated by max_jobs_per_run",
            f"- {self.rejected_by_role} non-target roles",
            f"- {self.rejected_by_seniority} seniority mismatch",
            f"- {self.rejected_by_location} non-US / international",
            f"- {self.rejected_by_employment_type} employment mismatch",
            f"- {self.rejected_by_freshness} older than freshness window "
            f"({self.unknown_timestamps} unknown timestamps)",
            f"- {self.rejected_by_invalid_url} invalid/missing direct URL",
            f"- {self.duplicates_removed} in-run duplicates",
            f"- {self.historical_jobs_skipped} already in tracker",
            f"= {self.jobs_accepted} accepted",
            "",
            f"AFTER CROSS-SOURCE DEDUP : {deduped}",
            f"AFTER ROLE FILTER        : {after_role}",
            f"AFTER EXPERIENCE FILTER  : {after_seniority}",
            f"AFTER LOCATION FILTER    : {after_location}",
            f"AFTER EMPLOYMENT FILTER  : {after_emp}",
            f"AFTER FRESHNESS          : {after_fresh}",
            f"AFTER URL VALIDATION     : {after_url}",
            f"UNKNOWN TIMESTAMP JOBS   : {self.unknown_timestamps}",
            f"FINAL ACCEPTED           : {self.jobs_accepted}",
        ]
        return "\n".join(lines)

    def freshness_candidate_report(self) -> str:
        data = self.freshness_breakdown
        if not data:
            return ""
        hours = self.freshness_hours_used
        return "\n".join(
            [
                "FRESHNESS BREAKDOWN",
                "===================",
                f"Candidates before freshness: {data.get('candidates', 0)}",
                "Posted date:",
                f"  within {hours:g}h: {data.get('posted_within', 0)}",
                f"  older than {hours:g}h: {data.get('posted_older', 0)}",
                "Updated date:",
                f"  within {hours:g}h: {data.get('updated_within', 0)}",
                f"  older than {hours:g}h: {data.get('updated_older', 0)}",
                "Unknown:",
                f"  {data.get('unknown', 0)}",
            ]
        )

    def nearest_misses_report(self) -> str:
        if not self.nearest_freshness_misses:
            return ""
        lines = [
            "NEAREST FRESHNESS MISSES",
            "========================",
            "(diagnostic only — not written to the tracker)",
        ]
        for item in self.nearest_freshness_misses:
            lines.append(item.get("company") or "-")
            lines.append(f"  {item.get('title') or '-'}")
            lines.append(f"  age: {item.get('age_hours')}h")
            lines.append(f"  date source: {item.get('date_source')}")
        return "\n".join(lines)

    def experience_report_text(self) -> str:
        data = self.experience_filter_report
        if not data:
            return ""
        return "\n".join(
            [
                "EXPERIENCE FILTER REPORT",
                "========================",
                f"Role-qualified: {data.get('role_qualified', 0)}",
                f"Accepted: {data.get('accepted', 0)}",
                "Rejected:",
                f"  explicit 3+ years: {data.get('explicit_3', 0)}",
                f"  explicit 4+ years: {data.get('explicit_4', 0)}",
                f"  explicit 5+ years: {data.get('explicit_5_plus', 0)}",
                f"  senior/staff/etc.: {data.get('seniority_title', 0)}",
                f"  ambiguous: {data.get('ambiguous', 0)}",
                f"  no experience requirement: {data.get('no_requirement', 0)}",
            ]
        )

    def why_zero_report(self) -> str:
        if self.jobs_accepted > 0:
            return ""
        after_role = self.remaining_after("role")
        after_seniority = self.remaining_after("role", "seniority")
        after_location = self.remaining_after("role", "seniority", "location")
        after_emp = self.remaining_after("role", "seniority", "location", "employment")
        after_fresh = self.remaining_after(
            "role", "seniority", "location", "employment", "freshness"
        )
        after_url = self.remaining_after(
            "role", "seniority", "location", "employment", "freshness", "url"
        )
        sources_ok = any(h.successful for h in self.source_health.values())
        bottleneck = self.primary_bottleneck()
        return "\n".join(
            [
                "WHY ZERO JOBS?",
                "==============",
                "Discovery:",
                f"  {self.jobs_discovered} raw jobs found",
                f"  Sources {'operational' if sources_ok else 'failed or empty'}",
                "Role:",
                f"  {after_role} SWE-compatible",
                "Experience:",
                f"  {after_seniority} within configured experience policy",
                "Location:",
                f"  {after_location} U.S.",
                "Employment:",
                f"  {after_emp} allowed types",
                "Freshness:",
                f"  {after_fresh} within {self.freshness_hours_used:g}h",
                f"  {self.rejected_by_freshness - self.unknown_timestamps} older than "
                f"{self.freshness_hours_used:g}h",
                f"  {self.unknown_timestamps} unknown timestamp",
                "URL:",
                f"  {after_url} would have valid direct URLs",
                f"Primary bottleneck: {bottleneck}",
            ]
        )

    def primary_bottleneck(self) -> str:
        """First sequential stage that reduced remaining jobs to zero."""
        if self.jobs_discovered == 0:
            if self.failed_sources or any(h.failed for h in self.source_health.values()):
                return "DISCOVERY (source failure)"
            return "DISCOVERY (zero jobs returned)"
        after_role = self.remaining_after("role")
        after_seniority = self.remaining_after("role", "seniority")
        after_location = self.remaining_after("role", "seniority", "location")
        after_emp = self.remaining_after("role", "seniority", "location", "employment")
        after_fresh = self.remaining_after(
            "role", "seniority", "location", "employment", "freshness"
        )
        after_url = self.remaining_after(
            "role", "seniority", "location", "employment", "freshness", "url"
        )
        stages = [
            ("ROLE", after_role),
            ("EXPERIENCE", after_seniority),
            ("LOCATION", after_location),
            ("EMPLOYMENT", after_emp),
            ("FRESHNESS", after_fresh),
            ("URL", after_url),
        ]
        previous = self.jobs_extracted
        for name, remaining in stages:
            if previous > 0 and remaining == 0:
                return name
            previous = remaining
        if self.historical_jobs_skipped and self.jobs_accepted == 0:
            return "HISTORICAL DEDUP"
        return "NONE"

    def coverage_report(self) -> str:
        if self.coverage is None:
            return ""
        from src.services.discovery_report import render_coverage

        return render_coverage(self.coverage)

    def freshness_source_report(self) -> str:
        if not self.freshness_by_source:
            return ""
        from src.services.discovery_report import render_freshness_by_source

        return render_freshness_by_source(self.freshness_by_source, self.freshness_hours_used)

    def workday_funnel_report(self) -> str:
        data = self.workday_funnel
        if not data:
            return ""
        lines = [
            "WORKDAY FUNNEL",
            "==============",
            f"Discovered: {data.get('discovered', 0)}",
            f"Role: {data.get('role', 0)}",
            f"Experience: {data.get('experience', 0)}",
            f"US: {data.get('us', 0)}",
            f"Employment: {data.get('employment', 0)}",
            f"Fresh: {data.get('fresh', 0)}",
            f"URL: {data.get('url', 0)}",
            f"Final: {data.get('final', 0)}",
        ]
        if self.workday_samples:
            lines += ["", "WORKDAY SAMPLE", "=============="]
            for item in self.workday_samples:
                lines.append(item.get("company") or "-")
                lines.append(f"  title: {item.get('title')}")
                lines.append(f"  job ID: {item.get('job_id') or '-'}")
                lines.append(f"  posted: {item.get('posted') or '-'}")
                lines.append(f"  date source: {item.get('date_source') or '-'}")
                lines.append(f"  rejection: {item.get('reason') or '-'}")
                if item.get("detail"):
                    lines.append(f"  detail: {item.get('detail')}")
        return "\n".join(lines)

    def accepted_jobs_report(self) -> str:
        if not self.accepted_job_audit:
            return ""
        lines = ["ACCEPTED JOBS", "=============="]
        for item in self.accepted_job_audit:
            lines.append(f"{item.get('company')} | {item.get('title')}")
            lines.append(f"  job ID: {item.get('job_id')}")
            lines.append(f"  posted: {item.get('posted')} ({item.get('date_source')})")
            lines.append(f"  location: {item.get('location')}")
            lines.append(f"  employment: {item.get('employment')}")
            lines.append(f"  source: {item.get('source')}")
            lines.append(f"  url: {item.get('url')}")
            lines.append(f"  sponsorship: {item.get('sponsorship')}")
        return "\n".join(lines)

    def company_discovery_report(self) -> str:
        if not self.company_discovery_rows:
            return ""
        from src.services.discovery_report import render_company_discovery_report

        return render_company_discovery_report(self.company_discovery_rows)


class PipelineState(BaseModel):
    """Typed state threaded through the LangGraph pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AppConfig
    raw_postings: list[RawJobPosting] = Field(default_factory=list)
    jobs: list[Job] = Field(default_factory=list)
    rejected: list[RejectedJob] = Field(default_factory=list)
    company_outcomes: list[CompanyOutcome] = Field(default_factory=list)
    summary: RunSummary = Field(default_factory=RunSummary)
    # Populated by the history service, consumed by the dedup and output agents.
    known_keys: set[str] = Field(default_factory=set)
    preserved_tracking: dict[str, dict[str, str]] = Field(default_factory=dict)
    # Shared singletons injected by the runner (http client, llm, services).
    resources: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def reject(
        self,
        job: Job,
        reason: RejectionReason,
        detail: str | None = None,
    ) -> None:
        """Record a rejection and update the matching summary counter."""
        self.rejected.append(
            RejectedJob(
                company=job.company,
                title=job.job_title,
                reason=reason,
                detail=detail,
                url=job.direct_application_url,
                source=job.source,
                date_source=job.date_source,
            )
        )
        self.summary.count_rejection(reason)


# Resolve the forward reference to AppConfig now that both modules exist.
def _rebuild() -> None:
    from src.models.config import AppConfig  # noqa: F401

    PipelineState.model_rebuild()


_rebuild()

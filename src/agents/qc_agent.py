"""Final quality checks. Sponsorship status never causes a drop."""

from __future__ import annotations

from src.models.job import (
    ACCEPTED_EMPLOYMENT_TYPES,
    DecisionSource,
    Job,
    JobDecision,
    RejectionReason,
    SponsorshipScope,
    VisaSponsorshipStatus,
)
from src.models.state import PipelineState
from src.utils.logging import get_logger
from src.utils.urls import UrlKind, classify_url, is_http_url

log = get_logger(__name__)

_GUARANTEE_PHRASES = (
    "will sponsor your h-1b",
    "will sponsor your h1b",
    "guarantees sponsorship",
    "this job will sponsor",
)


async def run_quality_control(state: PipelineState) -> None:
    policy = state.config.url_policy
    kept: list[Job] = []
    seen: set[str] = set()

    for job in state.jobs:
        warnings: list[str] = []

        if not job.company or not job.job_title:
            state.reject(job, RejectionReason.QUALITY_CONTROL, "missing company or title")
            continue
        if not is_http_url(job.direct_application_url):
            state.reject(job, RejectionReason.QUALITY_CONTROL, "direct application URL is not valid http(s)")
            continue

        verdict = classify_url(job.direct_application_url, policy)
        if verdict.kind is UrlKind.AGGREGATOR:
            state.reject(job, RejectionReason.QUALITY_CONTROL, "final URL is an aggregator")
            continue
        if not verdict.acceptable_as_final:
            state.reject(job, RejectionReason.QUALITY_CONTROL, verdict.reason or "final URL is not acceptable")
            continue

        if job.employment_type not in ACCEPTED_EMPLOYMENT_TYPES:
            state.reject(job, RejectionReason.QUALITY_CONTROL, "employment type slipped past the filter")
            continue

        if job.dedup_key in seen:
            state.reject(job, RejectionReason.QUALITY_CONTROL, "accidental duplicate after earlier dedup")
            continue
        seen.add(job.dedup_key)

        if job.posted_at and job.updated_at and job.updated_at < job.posted_at:
            warnings.append("updated date is earlier than posted date")

        evidence = (job.visa_sponsorship_evidence or "").lower()
        if any(phrase in evidence for phrase in _GUARANTEE_PHRASES):
            job.visa_sponsorship_evidence = (
                "Current posting indicates sponsorship is available."
                if job.visa_sponsorship_status is VisaSponsorshipStatus.CONFIRMED
                else "Historical H-1B/LCA evidence exists for the employer; this is not a guarantee."
            )
            warnings.append("rewrote unsupported sponsorship guarantee")

        if (
            job.visa_sponsorship_status is VisaSponsorshipStatus.CONFIRMED
            and job.sponsorship_scope is SponsorshipScope.HISTORICAL_COMPANY
        ):
            job.visa_sponsorship_status = VisaSponsorshipStatus.LIKELY
            job.sponsorship_scope = SponsorshipScope.HISTORICAL_COMPANY
            warnings.append("company-level historical evidence cannot be CONFIRMED")

        if (
            job.visa_sponsorship_status is VisaSponsorshipStatus.CONFIRMED
            and job.sponsorship_scope in (
                SponsorshipScope.HISTORICAL_ROLE,
                SponsorshipScope.HISTORICAL_COMPANY,
            )
        ):
            job.visa_sponsorship_status = VisaSponsorshipStatus.LIKELY
            warnings.append("historical evidence cannot be represented as CONFIRMED")

        job.qc_warnings = warnings
        job.record(JobDecision(agent="qc", passed=True, decided_by=DecisionSource.DETERMINISTIC))
        kept.append(job)

    # NOT_SUPPORTED and UNKNOWN must still be here.
    state.jobs = kept
    state.summary.jobs_accepted = len(kept)
    state.summary.new_jobs = sum(1 for job in kept if job.is_new)
    log.info("quality control complete", accepted=len(kept), rejected=state.summary.rejected_by_quality_control)

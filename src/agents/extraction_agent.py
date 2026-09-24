"""Turn raw discovery payloads into normalized :class:`Job` objects."""

from __future__ import annotations

from src.llm.base import LLMProvider, NullLLMProvider
from src.llm.prompts.extraction import build_extraction_prompt
from src.llm.schemas import ExtractionResult
from src.models.job import (
    DateSource,
    DecisionSource,
    EmploymentType,
    Job,
    JobDecision,
    RawJobPosting,
    RejectionReason,
    RemoteType,
)
from src.models.state import PipelineState, RejectedJob
from src.utils.dates import parse_datetime
from src.utils.logging import get_logger
from src.utils.normalization import (
    clean_text,
    detect_employment_type,
    detect_remote_type,
    normalize_location,
)
from src.utils.urls import extract_job_id

log = get_logger(__name__)


async def run_extraction(state: PipelineState) -> None:
    llm: LLMProvider = state.resources.get("llm") or NullLLMProvider()
    jobs: list[Job] = []

    for raw in state.raw_postings:
        posting = raw
        if _needs_llm_extraction(posting) and llm.can_call():
            posting = await _enrich_with_llm(posting, llm, state)

        job = _to_job(posting)
        if job is None:
            state.rejected.append(
                RejectedJob(
                    company=posting.company_name or "unknown",
                    title=posting.title or "unknown",
                    reason=RejectionReason.EXTRACTION_FAILED,
                    detail="missing company, title or application URL",
                    url=posting.apply_url,
                )
            )
            state.summary.count_rejection(RejectionReason.EXTRACTION_FAILED)
            continue
        jobs.append(job)

    state.jobs = jobs
    state.summary.jobs_extracted = len(jobs)
    log.info("extraction complete", extracted=len(jobs), rejected=state.summary.rejected_by_extraction)


def _needs_llm_extraction(raw: RawJobPosting) -> bool:
    return not (raw.company_name and raw.title) and bool(raw.description or raw.apply_url)


async def _enrich_with_llm(
    raw: RawJobPosting, llm: LLMProvider, state: PipelineState
) -> RawJobPosting:
    system, user = build_extraction_prompt(
        text=raw.description or raw.title,
        known_company=raw.company_name,
        source_url=raw.apply_url,
    )
    result = await llm.structured(
        prompt=user,
        response_model=ExtractionResult,
        system=system,
        purpose="extraction",
    )
    state.summary.llm_calls = llm.stats.calls
    state.summary.llm_failures = llm.stats.failures
    if result is None:
        return raw

    updates = raw.model_copy()
    if result.company and not updates.company_name:
        updates.company_name = result.company
    if result.job_title and not updates.title:
        updates.title = result.job_title
    if result.location and not updates.location_raw:
        updates.location_raw = result.location
    if result.job_id and not updates.job_id:
        updates.job_id = result.job_id
    if result.posted_date and updates.posted_at is None:
        parsed = parse_datetime(result.posted_date)
        if parsed is not None:
            updates.posted_at = parsed
            updates.date_source = DateSource.POSTED_DATE
            updates.posted_at_raw = result.posted_date
    if result.employment_type and result.employment_type != "Unknown":
        updates.employment_type_raw = updates.employment_type_raw or result.employment_type
    if result.remote_type and result.remote_type != "Unknown":
        updates.remote_type_raw = updates.remote_type_raw or result.remote_type
    return updates


def _to_job(raw: RawJobPosting) -> Job | None:
    company = clean_text(raw.company_name)
    title = clean_text(raw.title)
    url = (raw.apply_url or "").strip()
    if not company or not title or not url:
        return None

    loc = normalize_location(raw.location_raw, description=raw.description)
    remote = detect_remote_type(raw.remote_type_raw, raw.location_raw, raw.title, raw.description)
    if remote is RemoteType.UNKNOWN:
        remote = loc.remote_type
    employment = detect_employment_type(
        raw.employment_type_raw, title=raw.title, description=raw.description
    )
    job_id = raw.job_id or extract_job_id(url)

    posted_at = raw.posted_at
    updated_at = raw.updated_at
    date_source = raw.date_source
    if posted_at is not None and date_source is DateSource.UNKNOWN:
        date_source = DateSource.POSTED_DATE
    elif posted_at is None and updated_at is not None and date_source is DateSource.UNKNOWN:
        date_source = DateSource.UPDATED_DATE

    job = Job(
        company=company,
        job_title=title,
        normalized_role=title,
        location=loc.display or (raw.location_raw or "Unknown"),
        location_city=loc.city,
        location_state=loc.state,
        remote_type=remote,
        employment_type=employment,
        posted_at=posted_at,
        updated_at=updated_at,
        date_source=date_source,
        found_at=raw.discovered_at,
        job_id=str(job_id) if job_id else None,
        source=raw.source,
        direct_application_url=url,
        discovery_url=url,
        description=raw.description,
    )
    job.record(
        JobDecision(
            agent="extraction",
            passed=True,
            decided_by=DecisionSource.DETERMINISTIC,
        )
    )
    return job

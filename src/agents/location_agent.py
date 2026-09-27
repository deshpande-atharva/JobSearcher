"""Keep U.S. locations and accepted employment types."""

from __future__ import annotations

import time

from src.models.job import (
    ACCEPTED_EMPLOYMENT_TYPES,
    DecisionSource,
    EmploymentType,
    JobDecision,
    RejectionReason,
    RemoteType,
)
from src.models.state import PipelineState
from src.utils.logging import get_logger
from src.utils.normalization import detect_employment_type, normalize_location

log = get_logger(__name__)


async def run_location_employment(state: PipelineState) -> None:
    require_us = state.config.settings.filters.require_us_location
    allowed = set(state.config.settings.filters.employment_types) or set(ACCEPTED_EMPLOYMENT_TYPES)
    kept = []
    location_seconds = 0.0
    employment_seconds = 0.0

    for job in state.jobs:
        started = time.perf_counter()
        loc = normalize_location(job.location, description=job.description)
        job.location_city = loc.city
        job.location_state = loc.state
        if job.remote_type is RemoteType.UNKNOWN:
            job.remote_type = loc.remote_type
        if loc.display and loc.display != "Unknown":
            job.location = loc.display

        location_rejected = False
        if loc.is_international_only:
            state.reject(job, RejectionReason.LOCATION, f"international-only location: {job.location}")
            location_rejected = True
        else:
            us_ok = loc.is_us
            if require_us and not us_ok:
                state.reject(job, RejectionReason.LOCATION, f"could not confirm a U.S. location: {job.location}")
                location_rejected = True
        location_seconds += time.perf_counter() - started
        if location_rejected:
            continue

        started = time.perf_counter()
        if job.employment_type is EmploymentType.UNKNOWN:
            job.employment_type = detect_employment_type(
                None, title=job.job_title, description=job.description
            )
        if job.employment_type is EmploymentType.UNKNOWN:
            # Most software-engineering requisitions omit the type and are full-time.
            job.employment_type = EmploymentType.FULL_TIME
        if job.employment_type not in allowed:
            state.reject(
                job,
                RejectionReason.EMPLOYMENT_TYPE,
                f"employment type {job.employment_type.value} is outside the target profile",
            )
            employment_seconds += time.perf_counter() - started
            continue

        job.record(JobDecision(agent="location", passed=True, decided_by=DecisionSource.DETERMINISTIC))
        kept.append(job)
        employment_seconds += time.perf_counter() - started

    state.summary.stage_seconds["location_gate"] = round(location_seconds, 3)
    state.summary.stage_seconds["employment_gate"] = round(employment_seconds, 3)
    state.jobs = kept
    log.info(
        "location/employment complete",
        kept=len(kept),
        loc=state.summary.rejected_by_location,
        emp=state.summary.rejected_by_employment_type,
    )

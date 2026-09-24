"""Keep jobs whose posted (or, if configured, updated) timestamp is within the window."""

from __future__ import annotations

from src.models.job import DateSource, DecisionSource, JobDecision, RejectionReason
from src.models.state import PipelineState
from src.services.freshness import date_channel, is_fresh
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_freshness(state: PipelineState) -> None:
    hours = state.config.freshness_hours
    allow_updated = state.config.settings.run.freshness_use_updated_when_posted_missing
    breakdown = {
        "candidates": len(state.jobs),
        "posted_within": 0,
        "posted_older": 0,
        "updated_within": 0,
        "updated_older": 0,
        "unknown": 0,
    }
    misses: list[tuple[float, object]] = []
    kept = []
    for job in state.jobs:
        channel = date_channel(job)
        fresh, age = is_fresh(
            job,
            hours,
            use_updated_when_posted_missing=allow_updated,
        )
        if channel is DateSource.POSTED_DATE:
            if fresh:
                breakdown["posted_within"] += 1
            else:
                breakdown["posted_older"] += 1
        elif channel is DateSource.UPDATED_DATE:
            if fresh:
                breakdown["updated_within"] += 1
            else:
                breakdown["updated_older"] += 1
        else:
            breakdown["unknown"] += 1

        if not fresh:
            if age is None:
                state.summary.unknown_timestamps += 1
                detail = "no reliable posted/updated timestamp"
            else:
                detail = f"age {age:.1f}h exceeds {hours}h window"
                misses.append((age, job))
            state.reject(job, RejectionReason.FRESHNESS, detail)
            if state.rejected:
                state.rejected[-1].age_hours = age
                state.rejected[-1].date_source = channel
            continue
        job.record(
            JobDecision(
                agent="freshness",
                passed=True,
                detail=f"age {age:.1f}h via {channel.value}" if age is not None else "fresh",
                decided_by=DecisionSource.DETERMINISTIC,
            )
        )
        kept.append(job)
    state.jobs = kept
    state.summary.freshness_breakdown = breakdown
    misses.sort(key=lambda item: item[0])
    state.summary.nearest_freshness_misses = [
        {
            "company": job.company,
            "title": job.job_title,
            "age_hours": round(age, 1),
            "date_source": date_channel(job).value,
        }
        for age, job in misses[:10]
    ]
    log.info("freshness complete", kept=len(kept), rejected=state.summary.rejected_by_freshness)

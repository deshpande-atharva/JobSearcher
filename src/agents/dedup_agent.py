"""Remove in-run duplicates and jobs already present in historical trackers."""

from __future__ import annotations

from src.models.job import RejectionReason
from src.models.state import PipelineState
from src.services.deduplication import deduplicate
from src.services.xlsx import apply_tracking
from src.utils.logging import get_logger

log = get_logger(__name__)


def _stamp_sighting(job, state: PipelineState) -> None:
    """Record when this run saw the job. Does not change posted_at or updated_at."""
    sightings = state.resources.get("sightings") or {}
    now = state.summary.run_started_at
    previous = sightings.get(job.dedup_key)
    if previous is None:
        job.first_seen_at = now
        job.last_seen_at = now
        return
    first, last = previous
    job.first_seen_at = first
    job.last_seen_at = last if last >= now else now


async def run_dedup(state: PipelineState) -> None:
    """Drop in-run duplicates. Keep previously seen jobs in today's snapshot.

    A job that still qualifies is written again with ``is_new=False`` so
    Applied/Status can be restored by identity. It is not removed from the
    current workbook just because it was discovered on an earlier day.
    """
    original = list(state.jobs)
    unique, duplicates, historical = deduplicate(original, state.known_keys)
    duplicate_ids = {id(job) for job in duplicates}
    for job in duplicates:
        state.reject(job, RejectionReason.DUPLICATE, f"duplicate of {job.dedup_key}")

    kept = []
    for job in original:
        if id(job) in duplicate_ids:
            continue
        apply_tracking(job, state.preserved_tracking)
        _stamp_sighting(job, state)
        kept.append(job)

    state.jobs = kept
    log.info(
        "dedup complete",
        unique=len(unique),
        duplicates=len(duplicates),
        previously_seen=len(historical),
        kept=len(kept),
    )

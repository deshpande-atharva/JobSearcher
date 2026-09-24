"""Remove in-run duplicates and jobs already present in historical trackers."""

from __future__ import annotations

from src.models.job import RejectionReason
from src.models.state import PipelineState
from src.services.deduplication import deduplicate
from src.services.xlsx import apply_tracking
from src.utils.logging import get_logger

log = get_logger(__name__)


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
        kept.append(job)

    state.jobs = kept
    log.info(
        "dedup complete",
        unique=len(unique),
        duplicates=len(duplicates),
        previously_seen=len(historical),
        kept=len(kept),
    )

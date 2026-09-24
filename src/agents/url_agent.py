"""Accept only a legitimate direct application URL as the final link."""

from __future__ import annotations

from src.models.job import DecisionSource, JobDecision, RejectionReason
from src.models.state import PipelineState
from src.services.url_verification import pick_direct_url, verify_url
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_url_verification(state: PipelineState) -> None:
    policy = state.config.url_policy
    http = state.resources.get("http")
    kept = []

    for job in state.jobs:
        # Rebuild a lightweight posting-shaped object from the job so we can
        # reuse the same candidate-selection logic.
        from src.models.job import RawJobPosting

        raw = RawJobPosting(
            source=job.source,
            company_name=job.company,
            title=job.job_title,
            job_id=job.job_id,
            apply_url=job.direct_application_url,
            alternate_urls=[u for u in (job.discovery_url,) if u and u != job.direct_application_url],
        )
        check = pick_direct_url(raw, policy)
        check = await verify_url(check, state.config, http)
        if not check.accepted:
            state.reject(job, RejectionReason.INVALID_URL, check.reason)
            continue
        job.direct_application_url = check.url
        if check.job_id and not job.job_id:
            job.job_id = check.job_id
        job.record(
            JobDecision(
                agent="url",
                passed=True,
                detail=check.reason,
                decided_by=DecisionSource.DETERMINISTIC,
            )
        )
        kept.append(job)

    state.jobs = kept
    log.info("url verification complete", kept=len(kept), rejected=state.summary.rejected_by_invalid_url)

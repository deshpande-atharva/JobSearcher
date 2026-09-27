"""Score surviving jobs with the shared intelligence and critic agents.

This node does not accept or reject a job. Deterministic gates have already run.
"""

from __future__ import annotations

import time

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.models.job import RawJobPosting
from src.models.state import PipelineState
from src.services.candidate_profile import load_candidate_profile, merge_resume_profile
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_job_intelligence(state: PipelineState) -> None:
    started = time.perf_counter()
    profile_path = state.config.project_root / state.config.settings.candidate.profile_path
    search = load_candidate_profile(state.config.project_root / "config" / "candidate_profile.yaml")
    profile = merge_resume_profile(search, profile_path)
    if profile.profile_version is None:
        state.summary.intelligence_evaluated = 0
        state.summary.critic_reviewed = 0
        state.summary.stage_seconds["intelligence"] = round(time.perf_counter() - started, 3)
        state.summary.stage_seconds["job_intelligence_gate"] = 0.0
        state.summary.stage_seconds["critic_gate"] = 0.0
        log.info("job intelligence skipped", reason="no saved resume profile")
        return

    evaluated = 0
    removed = 0
    intelligence_seconds = 0.0
    critic_seconds = 0.0
    for job in list(state.jobs):
        identity = (job.job_id, job.direct_application_url, job.posted_at, job.date_source, job.source)
        try:
            posting = _posting_from_job(job)
            mark = time.perf_counter()
            fit = evaluate_job(posting, profile, state.config.roles)
            intelligence_seconds += time.perf_counter() - mark
            mark = time.perf_counter()
            critique = review_fit(fit, posting, profile)
            critic_seconds += time.perf_counter() - mark
        except Exception as exc:
            log.warning("job intelligence skipped one job", error_type=type(exc).__name__)
            continue
        if (job.job_id, job.direct_application_url, job.posted_at, job.date_source, job.source) != identity:
            job.job_id, job.direct_application_url, job.posted_at, job.date_source, job.source = identity
        evaluated += 1
        removed += len(critique.unsupported_claims)
    state.summary.intelligence_evaluated = evaluated
    state.summary.critic_reviewed = evaluated
    state.summary.unsupported_claims_removed = removed
    state.summary.stage_seconds["intelligence"] = round(time.perf_counter() - started, 3)
    state.summary.stage_seconds["job_intelligence_gate"] = round(intelligence_seconds, 3)
    state.summary.stage_seconds["critic_gate"] = round(critic_seconds, 3)
    log.info("job intelligence complete", evaluated=evaluated, unsupported_removed=removed)


def _posting_from_job(job) -> RawJobPosting:
    return RawJobPosting(
        source=job.source,
        company_name=job.company,
        title=job.job_title,
        description=job.description,
        location_raw=job.location,
        job_id=job.job_id,
        apply_url=job.direct_application_url,
        posted_at=job.posted_at,
        updated_at=job.updated_at,
        date_source=job.date_source,
        employment_type_raw=job.employment_type.value if job.employment_type else None,
        provenance=dict(job.provenance or {}),
    )

"""Keep software-engineering-related roles; escalate ambiguity to Gemini."""

from __future__ import annotations

from src.llm.base import LLMProvider, NullLLMProvider
from src.llm.prompts.role_classification import build_role_prompt
from src.llm.schemas import RoleClassificationResult
from src.models.job import DecisionSource, Job, JobDecision, RejectionReason
from src.models.state import PipelineState
from src.services.roles import classify_role
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_role_classification(state: PipelineState) -> None:
    llm: LLMProvider = state.resources.get("llm") or NullLLMProvider()
    kept: list[Job] = []

    for job in state.jobs:
        verdict = classify_role(job.job_title, job.description, state.config.roles)
        if verdict.needs_llm and llm.can_call():
            verdict_llm = await _ask_llm(job, llm, state)
            if verdict_llm is not None:
                if verdict_llm.is_software_engineering:
                    job.role_family = verdict_llm.role_family
                    job.normalized_role = state.config.roles.label_for(verdict_llm.role_family)
                    job.record(
                        JobDecision(
                            agent="role",
                            passed=True,
                            detail=verdict_llm.reasoning,
                            decided_by=DecisionSource.LLM,
                        )
                    )
                    kept.append(job)
                    continue
                state.reject(job, RejectionReason.ROLE, verdict_llm.reasoning or "LLM: not software engineering")
                continue

        if verdict.is_software_engineering:
            job.role_family = verdict.family
            job.normalized_role = verdict.label
            job.record(
                JobDecision(
                    agent="role",
                    passed=True,
                    detail=verdict.detail,
                    decided_by=verdict.decided_by,
                )
            )
            kept.append(job)
            continue

        state.reject(job, RejectionReason.ROLE, verdict.detail)

    state.jobs = kept
    log.info("role classification complete", kept=len(kept), rejected=state.summary.rejected_by_role)


async def _ask_llm(job: Job, llm: LLMProvider, state: PipelineState) -> RoleClassificationResult | None:
    system, user = build_role_prompt(
        title=job.job_title,
        company=job.company,
        description=job.description,
        candidate_families=[job.role_family] if job.role_family else None,
    )
    result = await llm.structured(
        prompt=user,
        response_model=RoleClassificationResult,
        system=system,
        purpose="role",
    )
    state.summary.llm_calls = llm.stats.calls
    state.summary.llm_failures = llm.stats.failures
    return result

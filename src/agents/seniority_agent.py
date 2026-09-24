"""Keep new-grad / 0-2 year roles. Preferred experience never rejects."""

from __future__ import annotations

from src.llm.base import LLMProvider, NullLLMProvider
from src.llm.prompts.seniority import build_seniority_prompt
from src.llm.schemas import SeniorityResult
from src.models.job import DecisionSource, Job, JobDecision, RejectionReason
from src.models.state import PipelineState
from src.services.seniority import classify_seniority
from src.utils.logging import get_logger
from src.utils.normalization import split_required_and_preferred

log = get_logger(__name__)


async def run_seniority(state: PipelineState) -> None:
    llm: LLMProvider = state.resources.get("llm") or NullLLMProvider()
    filters = state.config.settings.filters
    kept: list[Job] = []

    for job in state.jobs:
        verdict = classify_seniority(
            job.job_title,
            job.description,
            state.config.roles,
            max_required_years=filters.max_required_years,
            ignore_preferred=filters.ignore_preferred_experience,
        )
        job.required_years_min = verdict.min_years
        job.required_years_max = verdict.max_years

        if verdict.needs_llm and llm.can_call():
            required, preferred = split_required_and_preferred(job.description)
            llm_result = await _ask_llm(job, required, preferred, llm, state)
            if llm_result is not None:
                job.required_years_min = llm_result.required_years_min
                job.required_years_max = llm_result.required_years_max
                if llm_result.fits_entry_level:
                    job.record(
                        JobDecision(
                            agent="seniority",
                            passed=True,
                            detail=llm_result.reasoning,
                            decided_by=DecisionSource.LLM,
                        )
                    )
                    kept.append(job)
                    continue
                state.reject(job, RejectionReason.SENIORITY, llm_result.reasoning or "LLM: above entry level")
                if state.rejected:
                    state.rejected[-1].experience_category = "ambiguous"
                continue

        if verdict.fits_entry_level:
            job.record(
                JobDecision(
                    agent="seniority",
                    passed=True,
                    detail=verdict.detail,
                    decided_by=verdict.decided_by,
                )
            )
            kept.append(job)
            continue

        if verdict.needs_llm:
            # LLM unavailable: do not guess. Conservative reject of unknown seniority.
            state.reject(job, RejectionReason.SENIORITY, verdict.detail + " (LLM unavailable)")
            if state.rejected:
                state.rejected[-1].experience_category = _experience_category(verdict)
            continue
        state.reject(job, RejectionReason.SENIORITY, verdict.detail)
        if state.rejected:
            state.rejected[-1].experience_category = _experience_category(verdict)

    state.jobs = kept
    state.summary.experience_filter_report = _experience_report(state, len(kept))
    log.info("seniority complete", kept=len(kept), rejected=state.summary.rejected_by_seniority)


def _experience_category(verdict) -> str:
    years = verdict.min_years
    detail = (verdict.detail or "").lower()
    if "title seniority" in detail:
        return "seniority_title"
    if years is not None:
        if years >= 5:
            return "explicit_5_plus"
        if years >= 4:
            return "explicit_4"
        if years >= 3:
            return "explicit_3"
        return "explicit_other"
    if verdict.needs_llm or "no clear required-years" in detail:
        return "no_requirement" if "no clear" in detail else "ambiguous"
    return "ambiguous"


def _experience_report(state: PipelineState, accepted: int) -> dict[str, int]:
    rejected = [r for r in state.rejected if r.reason is RejectionReason.SENIORITY]
    counts = {
        "role_qualified": accepted + len(rejected),
        "accepted": accepted,
        "explicit_3": 0,
        "explicit_4": 0,
        "explicit_5_plus": 0,
        "seniority_title": 0,
        "ambiguous": 0,
        "no_requirement": 0,
    }
    for item in rejected:
        key = item.experience_category or "ambiguous"
        if key in counts:
            counts[key] += 1
        else:
            counts["ambiguous"] += 1
    return counts


async def _ask_llm(
    job: Job,
    required: str,
    preferred: str,
    llm: LLMProvider,
    state: PipelineState,
) -> SeniorityResult | None:
    system, user = build_seniority_prompt(
        title=job.job_title,
        description=job.description,
        required_section=required,
        preferred_section=preferred,
    )
    result = await llm.structured(
        prompt=user,
        response_model=SeniorityResult,
        system=system,
        purpose="seniority",
    )
    state.summary.llm_calls = llm.stats.calls
    state.summary.llm_failures = llm.stats.failures
    return result

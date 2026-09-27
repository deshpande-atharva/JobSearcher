"""Keep software-engineering roles. Escalate only genuine ambiguity to Gemini."""

from __future__ import annotations

from src.llm.base import LLMProvider, NullLLMProvider
from src.llm.prompts.role_classification import build_role_prompt
from src.llm.schemas import RoleClassificationResult
from src.models.job import DecisionSource, Job, JobDecision, RejectionReason
from src.models.state import PipelineState
from src.services.freshness import is_fresh
from src.services.roles import REPORT_FAMILIES, classify_role, semantic_family
from src.utils.logging import get_logger

log = get_logger(__name__)

# A low-confidence answer is not an accept or a reject.
_DECISION_CONFIDENCE = 0.6
_FAMILY_KEYS = ("required", "attempted", "accepted", "rejected", "uncertain", "unavailable")
_IDENTITY = (
    "job_id",
    "source",
    "direct_application_url",
    "posted_at",
    "date_source",
    "location",
    "employment_type",
    "visa_sponsorship_status",
    "visa_sponsorship_confidence",
    "visa_sponsorship_source",
    "visa_sponsorship_evidence",
)


async def run_role_classification(state: PipelineState) -> None:
    llm: LLMProvider = state.resources.get("llm") or NullLLMProvider()
    state.summary.llm_state_before = str(getattr(llm, "circuit_state", "CLOSED") or "CLOSED")
    _prepare_families(state)
    kept: list[Job] = []

    for job in state.jobs:
        verdict = classify_role(job.job_title, job.description, state.config.roles)
        if verdict.classification_state == "DETERMINISTIC_ACCEPT":
            _note_avoided(llm)
            state.summary.deterministic_accepts += 1
            _keep_deterministic(job, verdict)
            kept.append(job)
            continue
        if verdict.classification_state == "DETERMINISTIC_REJECT":
            _note_avoided(llm)
            state.summary.deterministic_rejects += 1
            state.reject(job, RejectionReason.ROLE, verdict.detail)
            continue

        family = semantic_family(verdict)
        state.summary.semantic_review_required += 1
        _count_fresh(state, job, "fresh_semantic_review_required")
        _bump(state, family, "required")
        if not llm.can_call():
            _note_unavailable(llm)
            if str(getattr(llm, "circuit_state", "")) == "OPEN":
                state.summary.semantic_reviews_blocked_by_circuit += 1
            _hold(state, job, "SEMANTIC_REVIEW_UNAVAILABLE", family, "semantic review unavailable")
            continue

        state.summary.semantic_review_attempted += 1
        _count_fresh(state, job, "fresh_semantic_review_attempted")
        _bump(state, family, "attempted")
        identity = _snapshot(job)
        result = await _ask_llm(job, llm, state)
        _restore(job, identity)
        outcome = _llm_outcome(result)
        if outcome == "ACCEPT" and result is not None:
            state.summary.semantic_review_accepted += 1
            _count_fresh(state, job, "fresh_semantic_review_accepted")
            _bump(state, family, "accepted")
            job.role_family = result.role_family
            job.normalized_role = state.config.roles.label_for(result.role_family)
            job.record(
                JobDecision(
                    agent="role",
                    passed=True,
                    detail=result.reasoning,
                    decided_by=DecisionSource.LLM,
                )
            )
            kept.append(job)
            continue
        if outcome == "REJECT" and result is not None:
            state.summary.semantic_review_rejected += 1
            _count_fresh(state, job, "fresh_semantic_review_rejected")
            _bump(state, family, "rejected")
            state.reject(job, RejectionReason.ROLE, result.reasoning or "LLM: not software engineering")
            continue
        if outcome == "UNCERTAIN":
            state.summary.semantic_review_uncertain += 1
            _count_fresh(state, job, "fresh_semantic_review_uncertain")
            _bump(state, family, "uncertain")
            _hold(
                state,
                job,
                "SEMANTIC_REVIEW_UNCERTAIN",
                family,
                "semantic review uncertain",
            )
            continue
        _hold(state, job, "SEMANTIC_REVIEW_UNAVAILABLE", family, "semantic review unavailable")

    state.jobs = kept
    log.info(
        "role classification complete",
        kept=len(kept),
        rejected=state.summary.rejected_by_role,
        semantic_unavailable=state.summary.semantic_review_unavailable,
        semantic_uncertain=state.summary.semantic_review_uncertain,
    )


def _llm_outcome(result: RoleClassificationResult | None) -> str:
    if result is None:
        return "UNAVAILABLE"
    if result.confidence < _DECISION_CONFIDENCE:
        return "UNCERTAIN"
    if result.is_software_engineering:
        return "ACCEPT"
    return "REJECT"


def _count_fresh(state: PipelineState, job: Job, field: str) -> None:
    """Observational only. This does not move or replace the freshness gate."""
    fresh, _age = is_fresh(
        job,
        state.config.freshness_hours,
        use_updated_when_posted_missing=state.config.settings.run.freshness_use_updated_when_posted_missing,
    )
    if fresh:
        setattr(state.summary, field, getattr(state.summary, field) + 1)


def _prepare_families(state: PipelineState) -> None:
    for name in REPORT_FAMILIES:
        state.summary.semantic_by_family.setdefault(name, {key: 0 for key in _FAMILY_KEYS})


def _bump(state: PipelineState, family: str, key: str) -> None:
    bucket = state.summary.semantic_by_family.setdefault(family, {item: 0 for item in _FAMILY_KEYS})
    bucket[key] = int(bucket.get(key, 0)) + 1


def _hold(state: PipelineState, job: Job, code: str, family: str, detail: str) -> None:
    if code == "SEMANTIC_REVIEW_UNAVAILABLE":
        state.summary.semantic_review_unavailable += 1
        _count_fresh(state, job, "fresh_semantic_review_unavailable")
        _bump(state, family, "unavailable")
    state.reject(
        job,
        RejectionReason.ROLE,
        detail,
        report_code=code,
        semantic_family=family,
    )


def _keep_deterministic(job: Job, verdict) -> None:
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


def _snapshot(job: Job) -> dict:
    return {name: getattr(job, name) for name in _IDENTITY}


def _restore(job: Job, identity: dict) -> None:
    for name, value in identity.items():
        setattr(job, name, value)


def _note_avoided(llm: LLMProvider) -> None:
    note = getattr(llm, "note_deterministic_avoided", None)
    if callable(note):
        note()


def _note_unavailable(llm: LLMProvider) -> None:
    note = getattr(llm, "note_unavailable", None)
    if callable(note):
        note()


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

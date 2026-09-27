"""Classify employer-timestamp age, then apply the configured eligibility window.

A tier is not a role decision. Jobs older than 24 hours stay eligible through
the recent window. Aging jobs stay eligible only when role and seniority
already passed deterministically. Stale, old, and unknown timestamps are left
out of the daily tracker. Discovery time is never the posting date.
"""

from __future__ import annotations

from src.models.job import DateSource, DecisionSource, Job, JobDecision, RejectionReason
from src.models.state import PipelineState
from src.services.freshness import FreshnessTier, date_channel, freshness_verdict, is_fresh, tier_is_eligible
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_freshness(state: PipelineState) -> None:
    hours = state.config.freshness_hours
    policy = state.config.settings.freshness
    tiers = policy.tiers
    allow_updated = state.config.settings.run.freshness_use_updated_when_posted_missing
    breakdown = {
        "candidates": len(state.jobs),
        "posted_within": 0,
        "posted_older": 0,
        "updated_within": 0,
        "updated_older": 0,
        "unknown": 0,
    }
    tier_counts = {tier.value: 0 for tier in FreshnessTier}
    misses: list[tuple[float, object]] = []
    kept = []
    for job in state.jobs:
        channel = date_channel(job)
        now = state.resources.get("freshness_now")
        observational, _age = is_fresh(
            job,
            hours,
            now=now,
            use_updated_when_posted_missing=allow_updated,
        )
        verdict = freshness_verdict(
            job,
            now=now,
            use_updated_when_posted_missing=allow_updated,
            very_fresh_hours=tiers.very_fresh_hours,
            fresh_hours=tiers.fresh_hours,
            recent_hours=tiers.recent_hours,
            aging_hours=tiers.aging_hours,
            stale_hours=tiers.stale_hours,
        )
        job.freshness_tier = verdict.tier.value
        tier_counts[verdict.tier.value] += 1
        age = verdict.age_hours
        if channel is DateSource.POSTED_DATE:
            breakdown["posted_within" if observational else "posted_older"] += 1
        elif channel is DateSource.UPDATED_DATE:
            breakdown["updated_within" if observational else "updated_older"] += 1
        else:
            breakdown["unknown"] += 1

        eligible = tier_is_eligible(
            verdict.tier,
            strongly_qualified=_strongly_qualified(job),
            eligible_through=policy.eligible_through,
            aging_if_strongly_qualified=policy.aging_if_strongly_qualified,
            enabled=policy.enabled,
        )
        if not eligible:
            if age is None or verdict.tier is FreshnessTier.UNKNOWN:
                state.summary.unknown_timestamps += 1
                detail = "freshness tier UNKNOWN; no employer posted or updated timestamp"
            else:
                detail = (
                    f"freshness tier {verdict.tier.value} "
                    f"age {age:.1f}h via {verdict.date_source.value}"
                )
                misses.append((age, job))
            state.reject(job, RejectionReason.FRESHNESS, detail)
            if state.rejected:
                state.rejected[-1].age_hours = age
                state.rejected[-1].date_source = channel
            continue
        age_text = f"{age:.1f}h" if age is not None else "disabled"
        job.record(
            JobDecision(
                agent="freshness",
                passed=True,
                detail=f"tier {verdict.tier.value} age {age_text} via {channel.value}",
                decided_by=DecisionSource.DETERMINISTIC,
            )
        )
        kept.append(job)
    state.jobs = kept
    state.summary.freshness_breakdown = breakdown
    state.summary.freshness_tiers = tier_counts
    misses.sort(key=lambda item: item[0])
    state.summary.nearest_freshness_misses = [
        {
            "company": job.company,
            "title": job.job_title,
            "age_hours": round(age, 1),
            "date_source": date_channel(job).value,
            "freshness_tier": getattr(job, "freshness_tier", ""),
        }
        for age, job in misses[:10]
    ]
    log.info("freshness complete", kept=len(kept), rejected=state.summary.rejected_by_freshness)


def _strongly_qualified(job: Job) -> bool:
    """Deterministic role and seniority passes. An LLM accept is not enough."""
    passed = {decision.agent: decision for decision in job.decisions if decision.passed}
    role = passed.get("role")
    seniority = passed.get("seniority")
    location = passed.get("location")
    if role is None or seniority is None or location is None:
        return False
    return (
        role.decided_by == DecisionSource.DETERMINISTIC
        and seniority.decided_by == DecisionSource.DETERMINISTIC
    )

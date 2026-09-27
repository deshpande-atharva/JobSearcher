"""Run Workday postings through the existing qualification agents.

This is the same order as the daily graph. It does not define Workday-specific
role, experience, location, employment, freshness, URL, H-1B, or dedup rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.dedup_agent import run_dedup
from src.agents.extraction_agent import run_extraction
from src.agents.freshness_agent import run_freshness
from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
from src.agents.location_agent import run_location_employment
from src.agents.output_agent import run_output
from src.agents.qc_agent import run_quality_control
from src.agents.role_agent import run_role_classification
from src.agents.seniority_agent import run_seniority
from src.agents.url_agent import run_url_verification
from src.llm.base import LLMProvider, NullLLMProvider
from src.models.config import AppConfig
from src.models.job import RawJobPosting
from src.models.state import PipelineState

__all__ = ["QualificationFunnel", "qualify_workday_postings"]


@dataclass
class QualificationFunnel:
    state: PipelineState
    extracted: int
    role_candidates: int
    experience_candidates: int
    location_candidates: int
    employment_candidates: int
    fresh_candidates: int
    url_verified: int
    deduped_count: int
    final_count: int


async def qualify_workday_postings(
    config: AppConfig,
    postings: list[RawJobPosting],
    *,
    llm: LLMProvider | None = None,
    http=None,
    resources: dict | None = None,
) -> QualificationFunnel:
    state = PipelineState(config=config)
    state.raw_postings = list(postings)
    state.resources["llm"] = llm or NullLLMProvider()
    if http is not None:
        state.resources["http"] = http
    for key, value in (resources or {}).items():
        if value is not None:
            state.resources[key] = value
    await run_extraction(state)
    extracted = len(state.jobs)
    await run_role_classification(state)
    role_candidates = len(state.jobs)
    await run_seniority(state)
    experience_candidates = len(state.jobs)
    before_location = len(state.jobs)
    location_rejects = state.summary.rejected_by_location
    await run_location_employment(state)
    location_candidates = before_location - (state.summary.rejected_by_location - location_rejects)
    employment_candidates = len(state.jobs)
    await run_freshness(state)
    fresh_candidates = len(state.jobs)
    await run_url_verification(state)
    url_verified = len(state.jobs)
    await run_h1b_enrichment(state)
    await run_dedup(state)
    deduped_count = len(state.jobs)
    await run_quality_control(state)
    await run_output(state)
    return QualificationFunnel(
        state=state,
        extracted=extracted,
        role_candidates=role_candidates,
        experience_candidates=experience_candidates,
        location_candidates=location_candidates,
        employment_candidates=employment_candidates,
        fresh_candidates=fresh_candidates,
        url_verified=url_verified,
        deduped_count=deduped_count,
        final_count=len(state.jobs),
    )

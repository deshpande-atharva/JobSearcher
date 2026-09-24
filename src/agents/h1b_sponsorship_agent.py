"""Enrich qualified jobs with H-1B sponsorship evidence.

This agent never calls ``state.reject`` for a sponsorship reason. Every job it
receives is still in ``state.jobs`` when it returns, including NOT_SUPPORTED
and UNKNOWN.
"""

from __future__ import annotations

from src.llm.base import LLMProvider, NullLLMProvider
from src.llm.prompts.sponsorship import build_sponsorship_language_prompt
from src.llm.schemas import SponsorshipLanguageResult
from src.models.job import (
    DecisionSource,
    Job,
    LanguagePolarity,
    SponsorshipLanguageFinding,
    VisaSponsorshipStatus,
)
from src.models.state import PipelineState
from src.services.h1b import analyze_sponsorship, scan_sponsorship_language
from src.sources.h1bgrader import H1BGraderClient
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_h1b_enrichment(state: PipelineState) -> None:
    settings = state.config.settings.visa
    incoming = list(state.jobs)

    if not settings.enabled:
        for job in incoming:
            job.apply_sponsorship(analyze_sponsorship(
                job,
                scan_sponsorship_language(None),
                None,
                config=state.config,
            ))
        _tally(state)
        return

    llm: LLMProvider = state.resources.get("llm") or NullLLMProvider()
    client = state.resources.get("h1b") or H1BGraderClient(
        state.config, http=state.resources.get("http")
    )

    for job in incoming:
        language = scan_sponsorship_language(job.description if settings.use_current_posting_language else None)
        if (
            language.polarity is LanguagePolarity.AMBIGUOUS
            and settings.use_gemini_for_ambiguous_language
            and llm.can_call()
        ):
            language = await _refine_language(job, language, llm, state)

        skip_lookup = (
            settings.skip_lookup_when_job_explicit
            and language.confidence >= settings.explicit_language_confidence
            and language.polarity in (LanguagePolarity.POSITIVE, LanguagePolarity.NEGATIVE)
        )

        lookup = None
        if not skip_lookup and settings.provider != "none":
            company_cfg = state.config.universe.find(job.company)
            aliases = company_cfg.all_names[1:] if company_cfg else ()
            try:
                lookup = await client.lookup(job.company, aliases)
            except Exception as exc:
                log.info("h1b lookup failed; marking UNKNOWN", company=job.company, error=str(exc))
                from src.models.job import H1BLookupResult
                from src.utils.dates import utcnow

                lookup = H1BLookupResult(
                    company=job.company, found=False, error=str(exc), retrieved_at=utcnow()
                )
            if lookup.lookup_failed:
                state.summary.h1b_lookup_failures += 1

        aliases = ()
        company_cfg = state.config.universe.find(job.company)
        if company_cfg:
            aliases = company_cfg.all_names[1:]

        evidence = analyze_sponsorship(
            job, language, lookup, config=state.config, aliases=aliases
        )
        job.apply_sponsorship(evidence)

    # Invariant: every incoming job is still present.
    state.jobs = incoming
    _tally(state)
    log.info(
        "h1b enrichment complete",
        jobs=len(state.jobs),
        confirmed=state.summary.h1b_confirmed,
        likely=state.summary.h1b_likely,
        unknown=state.summary.h1b_unknown,
        not_supported=state.summary.h1b_not_supported,
    )


async def _refine_language(
    job: Job,
    current: SponsorshipLanguageFinding,
    llm: LLMProvider,
    state: PipelineState,
) -> SponsorshipLanguageFinding:
    snippets = [current.quote] if current.quote else None
    system, user = build_sponsorship_language_prompt(
        company=job.company,
        title=job.job_title,
        description=job.description,
        candidate_snippets=snippets,
    )
    result = await llm.structured(
        prompt=user,
        response_model=SponsorshipLanguageResult,
        system=system,
        purpose="sponsorship",
    )
    state.summary.llm_calls = llm.stats.calls
    state.summary.llm_failures = llm.stats.failures
    if result is None:
        return current
    try:
        polarity = LanguagePolarity(result.polarity)
    except ValueError:
        return current
    # The model is not allowed to invent CONFIRMED; we only adopt polarity.
    return SponsorshipLanguageFinding(
        polarity=polarity,
        confidence=min(result.confidence, 0.85),
        quote=result.quote or current.quote,
        decided_by=DecisionSource.LLM,
    )


def _tally(state: PipelineState) -> None:
    state.summary.h1b_confirmed = 0
    state.summary.h1b_likely = 0
    state.summary.h1b_unknown = 0
    state.summary.h1b_not_supported = 0
    for job in state.jobs:
        state.summary.count_sponsorship(job.visa_sponsorship_status)

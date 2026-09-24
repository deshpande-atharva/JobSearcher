"""H-1B evidence tests.

Sponsorship is enrichment. These tests lock the invariant that otherwise-
qualified jobs remain visible for every sponsorship outcome.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.config import VisaSettings
from src.models.job import (
    DecisionSource,
    H1BLookupResult,
    H1BMatchStrength,
    H1BRecord,
    LanguagePolarity,
    RejectionReason,
    SponsorshipLanguageFinding,
    SponsorshipScope,
    VisaSponsorshipStatus,
)
from src.models.state import PipelineState
from src.services.h1b import analyze_sponsorship, scan_sponsorship_language
from tests.conftest import make_job


def _analyze(tmp_config, job, language, lookup=None):
    return analyze_sponsorship(job, language, lookup, config=tmp_config)


def _lookup(*records: H1BRecord, error: str | None = None) -> H1BLookupResult:
    return H1BLookupResult(
        company="Acme Robotics",
        found=bool(records) and error is None,
        records=list(records),
        error=error,
    )


def test_rejection_reason_has_no_sponsorship_member() -> None:
    names = {item.name.lower() for item in RejectionReason} | {item.value.lower() for item in RejectionReason}
    banned = {"h1b", "sponsorship", "visa", "not_supported", "unknown_sponsorship"}
    assert names.isdisjoint(banned)


def test_require_h1b_sponsorship_is_rejected_by_config() -> None:
    with pytest.raises(ValidationError):
        VisaSettings(require_h1b_sponsorship=True)  # type: ignore[call-arg]


def test_explicit_sponsorship_statement(tmp_config) -> None:
    language = scan_sponsorship_language("Visa sponsorship available for qualified candidates.")
    assert language.polarity is LanguagePolarity.POSITIVE
    job = make_job()
    evidence = _analyze(tmp_config, job, language)
    assert evidence.status is VisaSponsorshipStatus.CONFIRMED
    assert evidence.scope is SponsorshipScope.JOB_SPECIFIC
    assert "available" in (evidence.evidence or "").lower()


def test_explicit_h1b_sponsorship_statement(tmp_config) -> None:
    language = scan_sponsorship_language("H-1B sponsorship available.")
    assert language.is_explicit_positive
    evidence = _analyze(tmp_config, make_job(), language)
    assert evidence.status is VisaSponsorshipStatus.CONFIRMED


def test_explicit_no_sponsorship(tmp_config) -> None:
    language = scan_sponsorship_language("We do not provide visa sponsorship.")
    assert language.polarity is LanguagePolarity.NEGATIVE
    evidence = _analyze(tmp_config, make_job(), language)
    assert evidence.status is VisaSponsorshipStatus.NOT_SUPPORTED
    assert evidence.scope is SponsorshipScope.JOB_SPECIFIC


def test_must_not_require_sponsorship_now_or_future(tmp_config) -> None:
    language = scan_sponsorship_language(
        "Candidates must not require sponsorship now or in the future."
    )
    assert language.polarity is LanguagePolarity.NEGATIVE
    evidence = _analyze(tmp_config, make_job(), language)
    assert evidence.status is VisaSponsorshipStatus.NOT_SUPPORTED


def test_no_sponsorship_information(tmp_config) -> None:
    language = scan_sponsorship_language("We build backend services in Python.")
    assert language.polarity is LanguagePolarity.ABSENT
    evidence = _analyze(tmp_config, make_job(), language, _lookup())
    assert evidence.status is VisaSponsorshipStatus.UNKNOWN


def test_historical_same_role_same_location(tmp_config) -> None:
    language = scan_sponsorship_language("No mention of visas.")
    lookup = _lookup(
        H1BRecord(
            employer="Acme Robotics Inc",
            job_title="Software Development Engineer",
            city="Seattle",
            state="WA",
            fiscal_year=2025,
            approvals=40,
        )
    )
    evidence = _analyze(tmp_config, make_job(), language, lookup)
    assert evidence.status is VisaSponsorshipStatus.LIKELY
    assert evidence.match_strength is H1BMatchStrength.STRONG
    assert evidence.scope is SponsorshipScope.HISTORICAL_ROLE
    assert evidence.historical_sponsor is True


def test_historical_unrelated_role_is_not_confirmed(tmp_config) -> None:
    language = scan_sponsorship_language("No mention of visas.")
    lookup = _lookup(
        H1BRecord(
            employer="Acme Robotics Inc",
            job_title="Marketing Manager",
            city="Chicago",
            state="IL",
            fiscal_year=2025,
            approvals=2,
        )
    )
    evidence = _analyze(tmp_config, make_job(), language, lookup)
    assert evidence.status is not VisaSponsorshipStatus.CONFIRMED
    assert evidence.scope is not SponsorshipScope.JOB_SPECIFIC


def test_different_location_is_weaker_than_same_location(tmp_config) -> None:
    language = scan_sponsorship_language("")
    same = _analyze(
        tmp_config,
        make_job(),
        language,
        _lookup(
            H1BRecord(
                employer="Acme Robotics Inc",
                job_title="Software Engineer",
                city="Seattle",
                state="WA",
                fiscal_year=2025,
            )
        ),
    )
    different = _analyze(
        tmp_config,
        make_job(),
        language,
        _lookup(
            H1BRecord(
                employer="Acme Robotics Inc",
                job_title="Software Engineer",
                city="Miami",
                state="FL",
                fiscal_year=2025,
            )
        ),
    )
    assert same.match_strength is H1BMatchStrength.STRONG
    assert different.match_strength is H1BMatchStrength.MODERATE


def test_old_vs_recent_sponsorship(tmp_config) -> None:
    language = scan_sponsorship_language("")
    old = _analyze(
        tmp_config,
        make_job(),
        language,
        _lookup(
            H1BRecord(
                employer="Acme Robotics Inc",
                job_title="Software Engineer",
                city="Seattle",
                state="WA",
                fiscal_year=2014,
            )
        ),
    )
    recent = _analyze(
        tmp_config,
        make_job(),
        language,
        _lookup(
            H1BRecord(
                employer="Acme Robotics Inc",
                job_title="Software Engineer",
                city="Seattle",
                state="WA",
                fiscal_year=2025,
            )
        ),
    )
    assert recent.status is VisaSponsorshipStatus.LIKELY
    assert old.status is not VisaSponsorshipStatus.CONFIRMED
    assert old.match_strength in {H1BMatchStrength.WEAK, H1BMatchStrength.MODERATE}


def test_current_no_sponsorship_overrides_historical(tmp_config) -> None:
    language = scan_sponsorship_language("We do not provide visa sponsorship.")
    lookup = _lookup(
        H1BRecord(
            employer="Acme Robotics Inc",
            job_title="Software Engineer",
            city="Seattle",
            state="WA",
            fiscal_year=2025,
            approvals=40,
        )
    )
    evidence = _analyze(tmp_config, make_job(), language, lookup)
    assert evidence.status is VisaSponsorshipStatus.NOT_SUPPORTED
    assert evidence.scope is SponsorshipScope.JOB_SPECIFIC
    assert evidence.historical_sponsor is True


def test_company_level_history_is_not_job_specific(tmp_config) -> None:
    language = scan_sponsorship_language("")
    evidence = _analyze(
        tmp_config,
        make_job(),
        language,
        _lookup(H1BRecord(employer="Acme Robotics Inc", fiscal_year=2025, approvals=10)),
    )
    assert evidence.scope is not SponsorshipScope.JOB_SPECIFIC
    assert evidence.status is not VisaSponsorshipStatus.CONFIRMED


def test_lookup_failure_is_unknown(tmp_config) -> None:
    language = scan_sponsorship_language("")
    evidence = _analyze(tmp_config, make_job(), language, _lookup(error="timeout"))
    assert evidence.status is VisaSponsorshipStatus.UNKNOWN
    assert evidence.lookup_failed is True


def test_ambiguous_language_stays_unknown(tmp_config) -> None:
    language = scan_sponsorship_language(
        "Employment eligibility will be evaluated based on business needs."
    )
    assert language.polarity is LanguagePolarity.AMBIGUOUS
    evidence = _analyze(tmp_config, make_job(), language)
    assert evidence.status is VisaSponsorshipStatus.UNKNOWN


def test_gemini_cannot_emit_confirmed_from_history() -> None:
    """The LLM response model has no CONFIRMED option by design."""
    from src.llm.schemas import SponsorshipLanguageResult

    fields = SponsorshipLanguageResult.model_fields["polarity"].annotation
    assert "CONFIRMED" not in str(fields)


@pytest.mark.asyncio
async def test_not_supported_and_unknown_jobs_remain(tmp_config) -> None:
    from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
    from src.models.job import H1BLookupResult
    from src.utils.dates import utcnow

    supported = make_job(job_id="a", description="Visa sponsorship available. New graduates welcome.")
    denied = make_job(
        job_id="b",
        location="Boston, MA",
        description="We do not provide visa sponsorship. Entry-level software engineer.",
    )
    unknown = make_job(job_id="c", description="Write production Python services. New graduates welcome.")

    class _Fake:
        async def lookup(self, company: str, aliases=()):
            return H1BLookupResult(company=company, found=False, retrieved_at=utcnow())

    state = PipelineState(config=tmp_config, jobs=[supported, denied, unknown])
    state.resources["h1b"] = _Fake()
    state.resources["llm"] = None
    await run_h1b_enrichment(state)

    assert len(state.jobs) == 3
    statuses = {job.job_id: job.visa_sponsorship_status for job in state.jobs}
    assert statuses["a"] is VisaSponsorshipStatus.CONFIRMED
    assert statuses["b"] is VisaSponsorshipStatus.NOT_SUPPORTED
    assert statuses["c"] is VisaSponsorshipStatus.UNKNOWN
    assert state.summary.rejected_by_role == 0
    assert "h1b" not in {r.reason.value for r in state.rejected}


def test_historical_evidence_never_becomes_job_specific_guarantee(tmp_config) -> None:
    language = scan_sponsorship_language("Build APIs.")
    evidence = _analyze(
        tmp_config,
        make_job(),
        language,
        _lookup(
            H1BRecord(
                employer="Acme Robotics Inc",
                job_title="Software Engineer",
                city="Seattle",
                state="WA",
                fiscal_year=2025,
            )
        ),
    )
    assert evidence.status is VisaSponsorshipStatus.LIKELY
    assert evidence.scope is SponsorshipScope.HISTORICAL_ROLE
    assert "guarantee" not in (evidence.evidence or "").lower()
    assert "will sponsor" not in (evidence.evidence or "").lower()


def test_display_values() -> None:
    assert VisaSponsorshipStatus.CONFIRMED.display == "Confirmed"
    assert VisaSponsorshipStatus.LIKELY.display == "Likely"
    assert VisaSponsorshipStatus.UNKNOWN.display == "Unknown"
    assert VisaSponsorshipStatus.NOT_SUPPORTED.display == "Not Supported"


def test_apply_sponsorship_does_not_drop_job() -> None:
    job = make_job()
    from src.models.job import SponsorshipEvidence

    job.apply_sponsorship(
        SponsorshipEvidence(status=VisaSponsorshipStatus.NOT_SUPPORTED, evidence="x")
    )
    assert job.visa_sponsorship_status is VisaSponsorshipStatus.NOT_SUPPORTED
    assert job.job_title == "Software Engineer"

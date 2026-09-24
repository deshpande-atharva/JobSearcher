"""Core job domain models.

The two important types are :class:`RawJobPosting` (whatever a discovery source
could scrape, deliberately permissive) and :class:`Job` (a fully qualified,
validated posting that is safe to write to the tracker).

Sponsorship fields live flat on :class:`Job` so the XLSX writer, the QC agent
and the tests all read the same names, and :class:`SponsorshipEvidence` is the
value object the H-1B service produces and applies onto a job.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.dates import utcnow

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RemoteType(StrEnum):
    """Normalized work arrangement."""

    REMOTE = "Remote"
    HYBRID = "Hybrid"
    ONSITE = "On-site"
    UNKNOWN = "Unknown"


class EmploymentType(StrEnum):
    """Employment types accepted by the target profile."""

    FULL_TIME = "Full-time"
    CONTRACT = "Contract"
    INTERNSHIP = "Internship"
    CO_OP = "Co-op"
    # Present so a posting can be classified honestly and then rejected by the
    # employment-type filter, rather than being silently coerced to Full-time.
    PART_TIME = "Part-time"
    TEMPORARY = "Temporary"
    VOLUNTEER = "Volunteer"
    UNKNOWN = "Unknown"


ACCEPTED_EMPLOYMENT_TYPES: frozenset[EmploymentType] = frozenset(
    {
        EmploymentType.FULL_TIME,
        EmploymentType.CONTRACT,
        EmploymentType.INTERNSHIP,
        EmploymentType.CO_OP,
    }
)


class DateSource(StrEnum):
    """Where a job's timestamp came from.

    This exists so an "updated" timestamp is never silently presented as the
    original posting date.
    """

    POSTED_DATE = "POSTED_DATE"
    UPDATED_DATE = "UPDATED_DATE"
    DISCOVERED_DATE = "DISCOVERED_DATE"
    UNKNOWN = "UNKNOWN"


class AppliedFlag(StrEnum):
    """Values offered by the Excel ``Applied`` dropdown."""

    NOT_APPLIED = "\u2610 Not Applied"
    APPLIED = "\u2611 Applied"


class ApplicationStatus(StrEnum):
    """Values offered by the Excel ``Status`` dropdown."""

    NOT_STARTED = "Not Started"
    APPLIED = "Applied"
    OA = "OA"
    RECRUITER_SCREEN = "Recruiter Screen"
    INTERVIEW = "Interview"
    REJECTED = "Rejected"
    OFFER = "Offer"
    WITHDRAWN = "Withdrawn"


class VisaSponsorshipStatus(StrEnum):
    """H-1B sponsorship *evidence* classification.

    This is enrichment. No value of this enum -- including ``NOT_SUPPORTED`` --
    may remove a job from the pipeline output.
    """

    CONFIRMED = "CONFIRMED"
    LIKELY = "LIKELY"
    UNKNOWN = "UNKNOWN"
    NOT_SUPPORTED = "NOT_SUPPORTED"

    @property
    def display(self) -> str:
        """Human-facing label used in the spreadsheet."""
        return _SPONSORSHIP_DISPLAY[self]


_SPONSORSHIP_DISPLAY: dict[VisaSponsorshipStatus, str] = {
    VisaSponsorshipStatus.CONFIRMED: "Confirmed",
    VisaSponsorshipStatus.LIKELY: "Likely",
    VisaSponsorshipStatus.UNKNOWN: "Unknown",
    VisaSponsorshipStatus.NOT_SUPPORTED: "Not Supported",
}

SPONSORSHIP_DISPLAY_TO_STATUS: dict[str, VisaSponsorshipStatus] = {
    v.lower(): k for k, v in _SPONSORSHIP_DISPLAY.items()
}


class SponsorshipScope(StrEnum):
    """What the sponsorship signal actually covers.

    Keeps company-level history from being presented as a job-level promise.
    """

    JOB_SPECIFIC = "JOB_SPECIFIC"
    COMPANY_POLICY = "COMPANY_POLICY"
    HISTORICAL_ROLE = "HISTORICAL_ROLE"
    HISTORICAL_COMPANY = "HISTORICAL_COMPANY"
    UNKNOWN = "UNKNOWN"


class H1BMatchStrength(StrEnum):
    """How well historical H-1B records line up with the current posting."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class LanguagePolarity(StrEnum):
    """Direction of sponsorship language found in a posting."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    AMBIGUOUS = "AMBIGUOUS"
    ABSENT = "ABSENT"


class DecisionSource(StrEnum):
    """Whether a classification came from rules, the LLM, or a fallback."""

    DETERMINISTIC = "DETERMINISTIC"
    LLM = "LLM"
    DEFAULT = "DEFAULT"


class RejectionReason(StrEnum):
    """Every reason a job may leave the pipeline.

    There is intentionally no sponsorship-related member. H-1B evidence is
    enrichment and can never reject a job; ``tests/test_h1b_sponsorship.py``
    asserts this enum stays free of sponsorship reasons so the invariant cannot
    be broken by a later refactor.
    """

    EXTRACTION_FAILED = "extraction_failed"
    ROLE = "role"
    SENIORITY = "seniority"
    LOCATION = "location"
    EMPLOYMENT_TYPE = "employment_type"
    FRESHNESS = "freshness"
    INVALID_URL = "invalid_url"
    DUPLICATE = "duplicate"
    ALREADY_SEEN = "already_seen"
    QUALITY_CONTROL = "quality_control"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class SponsorshipLanguageFinding(BaseModel):
    """Result of reading sponsorship language out of the posting text."""

    polarity: LanguagePolarity = LanguagePolarity.ABSENT
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    quote: str | None = None
    decided_by: DecisionSource = DecisionSource.DETERMINISTIC

    @property
    def is_explicit_negative(self) -> bool:
        return self.polarity is LanguagePolarity.NEGATIVE

    @property
    def is_explicit_positive(self) -> bool:
        return self.polarity is LanguagePolarity.POSITIVE


class H1BRecord(BaseModel):
    """One historical H-1B/LCA record as reported by the evidence provider.

    Sourced from public H-1B/LCA disclosure data (via H1BGrader). Historical
    only: it describes what an employer did previously, never what a particular
    current posting will do.
    """

    employer: str
    job_title: str | None = None
    city: str | None = None
    state: str | None = None
    fiscal_year: int | None = None
    approvals: int | None = None
    denials: int | None = None
    # When the record's activity ended, used for recency scoring.
    record_date: datetime | None = None
    provider: str = "h1bgrader"

    @property
    def location_label(self) -> str:
        parts = [p for p in (self.city, self.state) if p]
        return ", ".join(parts)


class H1BLookupResult(BaseModel):
    """Outcome of a historical sponsorship lookup for one company."""

    company: str
    found: bool = False
    records: list[H1BRecord] = Field(default_factory=list)
    provider: str = "h1bgrader"
    error: str | None = None
    retrieved_at: datetime | None = None
    from_cache: bool = False

    @property
    def lookup_failed(self) -> bool:
        return self.error is not None


class SponsorshipEvidence(BaseModel):
    """Everything known about sponsorship for one job.

    Produced by :mod:`src.services.h1b` and copied onto a :class:`Job`. It is
    purely descriptive -- there is no ``eligible`` or ``should_include`` field,
    by design.
    """

    status: VisaSponsorshipStatus = VisaSponsorshipStatus.UNKNOWN
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    evidence: str | None = None
    scope: SponsorshipScope = SponsorshipScope.UNKNOWN
    historical_sponsor: bool | None = None
    match_strength: H1BMatchStrength | None = None
    last_verified: datetime | None = None
    # Age in days of the strongest supporting historical record.
    evidence_age_days: int | None = None
    lookup_failed: bool = False
    decided_by: DecisionSource = DecisionSource.DETERMINISTIC

    @classmethod
    def unknown(cls, reason: str = "No reliable sponsorship information found.") -> SponsorshipEvidence:
        return cls(
            status=VisaSponsorshipStatus.UNKNOWN,
            confidence=0.15,
            evidence=reason,
            scope=SponsorshipScope.UNKNOWN,
        )


class JobDecision(BaseModel):
    """Audit trail entry for one agent's verdict on one job."""

    agent: str
    passed: bool
    reason: RejectionReason | None = None
    detail: str | None = None
    decided_by: DecisionSource = DecisionSource.DETERMINISTIC


# ---------------------------------------------------------------------------
# Raw discovery payload
# ---------------------------------------------------------------------------


class RawJobPosting(BaseModel):
    """A posting as discovered, before normalization or validation.

    Deliberately permissive: a discovery source should hand over whatever it
    genuinely saw and let the extraction agent normalize it. Missing data stays
    ``None`` -- sources must never invent titles, dates, IDs or URLs.
    """

    model_config = ConfigDict(extra="ignore")

    source: str
    company_name: str | None = None
    title: str | None = None
    location_raw: str | None = None
    description: str | None = None
    employment_type_raw: str | None = None
    remote_type_raw: str | None = None
    job_id: str | None = None
    # The best application URL this source could offer. May be an aggregator
    # link; the URL agent is responsible for resolving/rejecting it.
    apply_url: str | None = None
    # Secondary candidates the URL agent may fall back to.
    alternate_urls: list[str] = Field(default_factory=list)
    posted_at_raw: str | None = None
    updated_at_raw: str | None = None
    posted_at: datetime | None = None
    updated_at: datetime | None = None
    date_source: DateSource = DateSource.UNKNOWN
    discovered_at: datetime = Field(default_factory=utcnow)
    # Free-form provenance for debugging (page number, endpoint, fixture name).
    provenance: dict[str, Any] = Field(default_factory=dict)

    def label(self) -> str:
        return f"{self.company_name or 'unknown company'} / {self.title or 'unknown title'}"


# ---------------------------------------------------------------------------
# Final job
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


class Job(BaseModel):
    """A validated job that belongs in the tracker."""

    model_config = ConfigDict(validate_assignment=True)

    # --- identity -----------------------------------------------------------
    company: str
    company_key: str = ""
    job_title: str
    normalized_role: str
    role_family: str = "OTHER_SOFTWARE"

    # --- placement ----------------------------------------------------------
    location: str
    location_city: str | None = None
    location_state: str | None = None
    remote_type: RemoteType = RemoteType.UNKNOWN
    employment_type: EmploymentType = EmploymentType.UNKNOWN

    # --- timing -------------------------------------------------------------
    posted_at: datetime | None = None
    updated_at: datetime | None = None
    date_source: DateSource = DateSource.UNKNOWN
    found_at: datetime = Field(default_factory=utcnow)

    # --- provenance ---------------------------------------------------------
    job_id: str | None = None
    source: str
    direct_application_url: str
    discovery_url: str | None = None

    # --- user-owned tracking columns ---------------------------------------
    # Never overwritten once a value exists in the workbook.
    applied: AppliedFlag = AppliedFlag.NOT_APPLIED
    status: ApplicationStatus = ApplicationStatus.NOT_STARTED

    # --- H-1B enrichment (informational only) ------------------------------
    visa_sponsorship_status: VisaSponsorshipStatus = VisaSponsorshipStatus.UNKNOWN
    visa_sponsorship_confidence: float | None = None
    visa_sponsorship_source: str | None = None
    visa_sponsorship_evidence: str | None = None
    sponsorship_scope: SponsorshipScope = SponsorshipScope.UNKNOWN
    h1b_historical_sponsor: bool | None = None
    h1b_match_strength: H1BMatchStrength | None = None
    h1b_last_verified: datetime | None = None
    # Age in days of the strongest supporting historical record.
    h1b_evidence_age: int | None = None

    # --- internals (never written to the workbook) -------------------------
    description: str | None = Field(default=None, exclude=True)
    required_years_min: float | None = Field(default=None, exclude=True)
    required_years_max: float | None = Field(default=None, exclude=True)
    decisions: list[JobDecision] = Field(default_factory=list, exclude=True)
    qc_warnings: list[str] = Field(default_factory=list, exclude=True)
    is_new: bool = Field(default=True, exclude=True)

    @field_validator("company", "job_title", "location", mode="before")
    @classmethod
    def _collapse_whitespace(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _WHITESPACE_RE.sub(" ", value).strip()
        return value

    def model_post_init(self, __context: Any) -> None:  # noqa: D105
        if not self.company_key:
            from src.utils.normalization import normalize_company_name

            object.__setattr__(self, "company_key", normalize_company_name(self.company))

    # --- sponsorship --------------------------------------------------------

    def apply_sponsorship(self, evidence: SponsorshipEvidence) -> None:
        """Copy sponsorship evidence onto the flat job fields.

        Only ever writes the informational columns. Nothing here can change
        whether the job is included.
        """
        self.visa_sponsorship_status = evidence.status
        self.visa_sponsorship_confidence = evidence.confidence
        self.visa_sponsorship_source = evidence.source
        self.visa_sponsorship_evidence = evidence.evidence
        self.sponsorship_scope = evidence.scope
        self.h1b_historical_sponsor = evidence.historical_sponsor
        self.h1b_match_strength = evidence.match_strength
        self.h1b_last_verified = evidence.last_verified
        self.h1b_evidence_age = evidence.evidence_age_days

    @property
    def sponsorship_display(self) -> str:
        return self.visa_sponsorship_status.display

    # --- deduplication ------------------------------------------------------

    @property
    def dedup_key(self) -> str:
        """Stable identity for this posting.

        Primary key is company + job id. Two postings that share a title but
        carry different job IDs are different jobs and both survive. When no job
        ID exists we fall back to company + normalized title + location +
        canonical URL.
        """
        from src.utils.normalization import normalize_location_key, normalize_title
        from src.utils.urls import canonicalize_url

        if self.job_id:
            return f"{self.company_key}|id:{self.job_id.strip().lower()}"
        parts = [
            self.company_key,
            normalize_title(self.job_title),
            normalize_location_key(self.location),
            canonicalize_url(self.direct_application_url),
        ]
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
        return f"{self.company_key}|fp:{digest}"

    def record(self, decision: JobDecision) -> None:
        self.decisions.append(decision)

    def label(self) -> str:
        return f"{self.company} / {self.job_title} ({self.location})"

"""H-1B sponsorship *evidence* -- never a filter.

This module answers: "What evidence exists regarding H-1B sponsorship for this
job?" It does not answer: "Should this job be removed?"

Evidence priority:

1. Current job-specific sponsorship statement
2. Current official company sponsorship policy
3. Recent H1B evidence for same/similar role + location
4. Recent company-level H1B evidence
5. Older historical H1B evidence
6. No evidence

Historical H1BGrader data alone can never become CONFIRMED. CONFIRMED is
reserved for an explicit current job-level or official company-policy statement
that sponsorship is available.

NOT_SUPPORTED and UNKNOWN never remove a job. There is no code path here that
returns an eligibility flag.
"""

from __future__ import annotations

import re
from datetime import datetime

from src.models.config import AppConfig, RolesConfig, VisaSettings
from src.models.job import (
    DecisionSource,
    H1BLookupResult,
    H1BMatchStrength,
    H1BRecord,
    Job,
    LanguagePolarity,
    SponsorshipEvidence,
    SponsorshipLanguageFinding,
    SponsorshipScope,
    VisaSponsorshipStatus,
)
from src.utils.dates import days_between, utcnow
from src.utils.normalization import normalize_company_name, normalize_location, normalize_title

__all__ = [
    "EVIDENCE_TEMPLATES",
    "analyze_sponsorship",
    "companies_match",
    "historical_match_strength",
    "scan_sponsorship_language",
    "score_record",
]

EVIDENCE_TEMPLATES = {
    "confirmed": "Current job posting explicitly states sponsorship is available.",
    "not_supported": "Current job posting states that visa sponsorship is unavailable.",
    "likely_role": (
        "Recent historical H-1B/LCA records exist for the employer and a similar "
        "software-engineering role in the same location."
    ),
    "likely_company": "Recent historical H-1B/LCA records exist for the employer.",
    "unknown": "No reliable sponsorship information found.",
    "unknown_lookup": "Historical sponsorship lookup was unavailable; no reliable current statement found.",
    "ambiguous": "Sponsorship language in the posting is ambiguous or conditional.",
    "historical_old": "Older historical H-1B/LCA records exist for the employer; they are not current job-level evidence.",
    "unrelated_role": (
        "Historical H-1B/LCA records exist for the employer but not for a similar "
        "role or location; this is not job-specific sponsorship evidence."
    ),
}


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_POSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:visa|h-?1b|immigration|employment visa)\s+sponsorship\s+(?:is\s+)?available\b", re.I),
    re.compile(r"\bwe\s+(?:do\s+)?sponsor\s+(?:h-?1b|visas?|qualified candidates|work visas?)\b", re.I),
    re.compile(r"\b(?:will|we will)\s+sponsor\b", re.I),
    re.compile(r"\bsponsorship\s+(?:is\s+)?(?:available|provided|offered)\b", re.I),
    re.compile(r"\bwe\s+provide\s+(?:visa|immigration|h-?1b)\s+sponsorship\b", re.I),
    re.compile(r"\bemployment visa sponsorship available\b", re.I),
    re.compile(r"\bimmigration sponsorship available\b", re.I),
    re.compile(r"\bh-?1b sponsorship available\b", re.I),
    re.compile(r"\bvisa support (?:is\s+)?(?:available|provided)\b", re.I),
)

_NEGATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:will|do|does)\s+not\s+sponsor\b", re.I),
    re.compile(r"\bno visa sponsorship\b", re.I),
    re.compile(r"\bunable to sponsor\b", re.I),
    re.compile(r"\bcannot sponsor\b", re.I),
    re.compile(r"\bnot able to sponsor\b", re.I),
    re.compile(r"\bsponsorship is not available\b", re.I),
    re.compile(
        r"\b(?:do|does|will)\s+not\s+(?:provide|offer|give)\s+(?:visa\s+)?sponsorship\b",
        re.I,
    ),
    re.compile(r"\bmust be authorized to work\b.*\bwithout (?:visa\s+)?sponsorship\b", re.I),
    re.compile(r"\bwithout the need for (?:visa\s+)?sponsorship\b", re.I),
    re.compile(
        r"\bmust not require sponsorship(?:\s+now(?:\s+or\s+in the future)?)?",
        re.I,
    ),
    re.compile(r"\bwill not consider candidates who require sponsorship\b", re.I),
    re.compile(r"\bno sponsorships?\s+(?:is|are|will be)\s+available\b", re.I),
    re.compile(r"\bnot sponsor(?:ing)?\s+(?:h-?1b|work visas?|visas?)\b", re.I),
)

_AMBIGUOUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcase[- ]by[- ]case\b", re.I),
    re.compile(r"\bbased on business needs\b", re.I),
    re.compile(r"\bmay (?:be )?(?:sponsor|available)\b", re.I),
    re.compile(r"\bsponsorship (?:may be|might be|will be) (?:considered|evaluated|available)\b", re.I),
    re.compile(r"\bemployment eligibility will be evaluated\b", re.I),
    re.compile(r"\bsubject to applicable immigration\b", re.I),
)

_TOPIC_RE = re.compile(
    r"\b(?:visa|h-?1b|sponsorship|immigrat|work authorization|authorised to work|"
    r"authorized to work|employment eligibility)\b",
    re.I,
)


def scan_sponsorship_language(text: str | None) -> SponsorshipLanguageFinding:
    """Deterministic read of sponsorship wording in a posting."""
    if not text or not text.strip():
        return SponsorshipLanguageFinding(polarity=LanguagePolarity.ABSENT, confidence=0.2)

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]
    topic = [s for s in sentences if _TOPIC_RE.search(s)]

    for sentence in topic or sentences:
        for pattern in _NEGATIVE_PATTERNS:
            if pattern.search(sentence):
                return SponsorshipLanguageFinding(
                    polarity=LanguagePolarity.NEGATIVE,
                    confidence=0.92,
                    quote=sentence[:400],
                    decided_by=DecisionSource.DETERMINISTIC,
                )
        for pattern in _POSITIVE_PATTERNS:
            if pattern.search(sentence):
                return SponsorshipLanguageFinding(
                    polarity=LanguagePolarity.POSITIVE,
                    confidence=0.9,
                    quote=sentence[:400],
                    decided_by=DecisionSource.DETERMINISTIC,
                )

    for sentence in topic:
        for pattern in _AMBIGUOUS_PATTERNS:
            if pattern.search(sentence):
                return SponsorshipLanguageFinding(
                    polarity=LanguagePolarity.AMBIGUOUS,
                    confidence=0.55,
                    quote=sentence[:400],
                    decided_by=DecisionSource.DETERMINISTIC,
                )

    if topic:
        # Mentions work authorization or visas without a clear yes/no.
        return SponsorshipLanguageFinding(
            polarity=LanguagePolarity.AMBIGUOUS,
            confidence=0.45,
            quote=topic[0][:400],
            decided_by=DecisionSource.DETERMINISTIC,
        )

    return SponsorshipLanguageFinding(polarity=LanguagePolarity.ABSENT, confidence=0.2)


def companies_match(left: str | None, right: str | None, aliases: tuple[str, ...] = ()) -> bool:
    """True when two employer strings refer to the same organization."""
    keys = {normalize_company_name(name) for name in (left, *aliases) if name}
    keys.discard("")
    other = normalize_company_name(right)
    if not other or not keys:
        return False
    if other in keys:
        return True
    for key in keys:
        if len(key) >= 4 and (key in other or other in key):
            return True
    return False


def _title_related(job_title: str | None, record_title: str | None, roles: RolesConfig, family: str) -> bool:
    if not record_title:
        return False
    left = normalize_title(job_title)
    right = normalize_title(record_title)
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    group = roles.group_for(family)
    if not group:
        return False
    families = roles.h1b_role_groups.get(group, ())
    for member in families:
        spec = roles.role_families.get(member)
        if spec is None:
            continue
        if any(keyword.lower() in right for keyword in spec.keywords if keyword):
            return True
    return False


def _location_related(job: Job, record: H1BRecord) -> bool:
    if not record.city and not record.state:
        return False
    job_loc = normalize_location(job.location)
    rec_state = (record.state or "").strip().upper()
    rec_city = (record.city or "").strip().lower()
    state_match = bool(job_loc.state and rec_state and job_loc.state == rec_state)
    city_match = bool(job_loc.city and rec_city and job_loc.city.lower() == rec_city)
    if city_match and state_match:
        return True
    if city_match:
        return True
    if state_match and not rec_city:
        return True
    if state_match and rec_city:
        return True  # same state, different city -- still a location signal
    return False


def score_record(
    job: Job,
    record: H1BRecord,
    roles: RolesConfig,
    settings: VisaSettings,
    *,
    now: datetime | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[int, bool, bool, bool]:
    """Return ``(points, role_match, location_match, recent)``."""
    if not companies_match(job.company, record.employer, aliases):
        return 0, False, False, False

    reference = now or utcnow()
    year = record.fiscal_year
    recent = True
    if year is not None:
        recent = year >= reference.year - settings.historical_lookback_years
    elif record.record_date is not None:
        age_days = days_between(record.record_date, reference)
        recent = age_days is not None and age_days <= settings.historical_lookback_years * 365

    role_match = False
    if settings.use_job_title_matching:
        role_match = _title_related(job.job_title, record.job_title, roles, job.role_family)

    location_match = False
    if settings.use_location_matching:
        location_match = _location_related(job, record)

    points = 10
    if role_match:
        points += 20
    if location_match:
        points += 15
    if recent:
        points += 15
    else:
        points += 3
    if record.approvals:
        points += min(record.approvals, 20)
    return points, role_match, location_match, recent


def historical_match_strength(
    job: Job,
    lookup: H1BLookupResult | None,
    roles: RolesConfig,
    settings: VisaSettings,
    *,
    now: datetime | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[H1BMatchStrength, H1BRecord | None, bool]:
    """Score historical records. Lookup failure is UNKNOWN, not NONE."""
    if lookup is None:
        return H1BMatchStrength.UNKNOWN, None, False
    if lookup.lookup_failed:
        return H1BMatchStrength.UNKNOWN, None, False
    if not lookup.found or not lookup.records:
        return H1BMatchStrength.NONE, None, False

    best: tuple[int, H1BRecord, bool, bool, bool] | None = None
    for record in lookup.records:
        points, role_match, location_match, recent = score_record(
            job, record, roles, settings, now=now, aliases=aliases
        )
        if points <= 0:
            continue
        if best is None or points > best[0]:
            best = (points, record, role_match, location_match, recent)

    if best is None:
        return H1BMatchStrength.NONE, None, False

    _, record, role_match, location_match, recent = best
    if role_match and location_match and recent:
        return H1BMatchStrength.STRONG, record, True
    if role_match and recent:
        return H1BMatchStrength.MODERATE, record, True
    if role_match and location_match:
        return H1BMatchStrength.MODERATE, record, True
    if recent:
        return H1BMatchStrength.WEAK, record, True
    return H1BMatchStrength.WEAK, record, True


def analyze_sponsorship(
    job: Job,
    language: SponsorshipLanguageFinding,
    lookup: H1BLookupResult | None,
    *,
    config: AppConfig,
    aliases: tuple[str, ...] = (),
    now: datetime | None = None,
) -> SponsorshipEvidence:
    """Combine current language and historical evidence into a label.

    This function has no ``eligible`` / ``include`` output. Callers must attach
    the result to the job and keep the job.
    """
    settings = config.settings.visa
    roles = config.roles
    reference = now or utcnow()
    strength, best_record, historical_sponsor = historical_match_strength(
        job, lookup, roles, settings, now=reference, aliases=aliases
    )
    evidence_age = days_between(best_record.record_date, reference) if best_record else None
    if evidence_age is None and best_record and best_record.fiscal_year:
        evidence_age = max((reference.year - best_record.fiscal_year) * 365, 0)

    lookup_failed = bool(lookup and lookup.lookup_failed)
    threshold = settings.explicit_language_confidence

    # 1. Current job-specific statement wins, including explicit "no sponsorship"
    #    even when historical company sponsorship exists.
    if (
        settings.use_current_posting_language
        and language.is_explicit_negative
        and language.confidence >= threshold
    ):
        return SponsorshipEvidence(
            status=VisaSponsorshipStatus.NOT_SUPPORTED,
            confidence=language.confidence,
            source="job_posting",
            evidence=EVIDENCE_TEMPLATES["not_supported"],
            scope=SponsorshipScope.JOB_SPECIFIC,
            historical_sponsor=historical_sponsor or None,
            match_strength=strength if historical_sponsor else H1BMatchStrength.NONE,
            last_verified=reference,
            evidence_age_days=evidence_age,
            lookup_failed=lookup_failed,
            decided_by=language.decided_by,
        )

    if (
        settings.use_current_posting_language
        and language.is_explicit_positive
        and language.confidence >= threshold
    ):
        return SponsorshipEvidence(
            status=VisaSponsorshipStatus.CONFIRMED,
            confidence=language.confidence,
            source="job_posting",
            evidence=EVIDENCE_TEMPLATES["confirmed"],
            scope=SponsorshipScope.JOB_SPECIFIC,
            historical_sponsor=historical_sponsor or None,
            match_strength=strength if historical_sponsor else None,
            last_verified=reference,
            evidence_age_days=evidence_age,
            lookup_failed=lookup_failed,
            decided_by=language.decided_by,
        )

    if language.polarity is LanguagePolarity.AMBIGUOUS:
        # Ambiguous wording stays UNKNOWN (or LIKELY if historical evidence is
        # strong). Gemini may refine polarity before this function is called,
        # but it is not allowed to emit CONFIRMED.
        if strength is H1BMatchStrength.STRONG:
            return SponsorshipEvidence(
                status=VisaSponsorshipStatus.LIKELY,
                confidence=0.62,
                source="h1bgrader",
                evidence=EVIDENCE_TEMPLATES["likely_role"] + " " + EVIDENCE_TEMPLATES["ambiguous"],
                scope=SponsorshipScope.HISTORICAL_ROLE,
                historical_sponsor=True,
                match_strength=strength,
                last_verified=reference,
                evidence_age_days=evidence_age,
                lookup_failed=lookup_failed,
                decided_by=DecisionSource.DETERMINISTIC,
            )
        return SponsorshipEvidence(
            status=VisaSponsorshipStatus.UNKNOWN,
            confidence=language.confidence,
            source="job_posting",
            evidence=EVIDENCE_TEMPLATES["ambiguous"],
            scope=SponsorshipScope.JOB_SPECIFIC,
            historical_sponsor=historical_sponsor or None,
            match_strength=strength,
            last_verified=reference,
            evidence_age_days=evidence_age,
            lookup_failed=lookup_failed,
            decided_by=language.decided_by,
        )

    if lookup_failed:
        return SponsorshipEvidence(
            status=VisaSponsorshipStatus.UNKNOWN,
            confidence=0.2,
            source="h1bgrader",
            evidence=EVIDENCE_TEMPLATES["unknown_lookup"],
            scope=SponsorshipScope.UNKNOWN,
            historical_sponsor=None,
            match_strength=H1BMatchStrength.UNKNOWN,
            last_verified=reference,
            lookup_failed=True,
        )

    if strength is H1BMatchStrength.STRONG:
        return SponsorshipEvidence(
            status=VisaSponsorshipStatus.LIKELY,
            confidence=0.78,
            source="h1bgrader",
            evidence=EVIDENCE_TEMPLATES["likely_role"],
            scope=SponsorshipScope.HISTORICAL_ROLE,
            historical_sponsor=True,
            match_strength=strength,
            last_verified=reference,
            evidence_age_days=evidence_age,
        )

    if strength is H1BMatchStrength.MODERATE:
        return SponsorshipEvidence(
            status=VisaSponsorshipStatus.LIKELY,
            confidence=0.64,
            source="h1bgrader",
            evidence=EVIDENCE_TEMPLATES["likely_role"],
            scope=SponsorshipScope.HISTORICAL_ROLE,
            historical_sponsor=True,
            match_strength=strength,
            last_verified=reference,
            evidence_age_days=evidence_age,
        )

    if strength is H1BMatchStrength.WEAK:
        # Company-level or stale records: visible as history, not a job-level claim.
        return SponsorshipEvidence(
            status=VisaSponsorshipStatus.UNKNOWN,
            confidence=0.35,
            source="h1bgrader",
            evidence=(
                EVIDENCE_TEMPLATES["unrelated_role"]
                if best_record and best_record.job_title
                else EVIDENCE_TEMPLATES["historical_old"]
            ),
            scope=SponsorshipScope.HISTORICAL_COMPANY,
            historical_sponsor=True,
            match_strength=strength,
            last_verified=reference,
            evidence_age_days=evidence_age,
        )

    return SponsorshipEvidence(
        status=VisaSponsorshipStatus.UNKNOWN,
        confidence=0.15,
        source="h1bgrader" if lookup is not None else None,
        evidence=EVIDENCE_TEMPLATES["unknown"],
        scope=SponsorshipScope.UNKNOWN,
        historical_sponsor=False if lookup and lookup.found is False else None,
        match_strength=strength,
        last_verified=reference,
        lookup_failed=lookup_failed,
    )

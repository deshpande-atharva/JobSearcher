"""Response models for structured LLM output.

Kept deliberately flat -- scalar fields and ``Literal`` unions only -- so the
generated JSON Schema stays inside the subset structured-output mode supports,
with no ``$ref`` indirection.

Every model carries a ``reasoning`` field. It is short, stored in the audit
trail, and makes a surprising classification explainable after the fact.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ExtractionResult",
    "RoleClassificationResult",
    "SeniorityResult",
    "SponsorshipLanguageResult",
]


class _LLMResult(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RoleClassificationResult(_LLMResult):
    """Is this posting genuinely a software-engineering role?"""

    is_software_engineering: bool = Field(
        description="True when the actual work described is software engineering."
    )
    role_family: Literal[
        "SOFTWARE_ENGINEER",
        "BACKEND_ENGINEER",
        "FRONTEND_ENGINEER",
        "FULLSTACK_ENGINEER",
        "PLATFORM_ENGINEER",
        "CLOUD_ENGINEER",
        "INFRASTRUCTURE_ENGINEER",
        "DEVOPS_ENGINEER",
        "SITE_RELIABILITY_ENGINEER",
        "SYSTEMS_ENGINEER",
        "PRODUCT_ENGINEER",
        "MOBILE_ENGINEER",
        "SECURITY_ENGINEER",
        "QA_AUTOMATION_ENGINEER",
        "DATA_ENGINEER",
        "MACHINE_LEARNING_ENGINEER",
        "OTHER_SOFTWARE",
        "NOT_SOFTWARE",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=600)


class SeniorityResult(_LLMResult):
    """Does the posting's *required* experience fit a new-grad/0-2yr profile?"""

    fits_entry_level: bool
    required_years_min: float | None = Field(
        default=None, description="Lowest number of years the REQUIRED qualifications demand."
    )
    required_years_max: float | None = Field(default=None)
    preferred_years_min: float | None = Field(
        default=None,
        description="Years mentioned only under preferred/nice-to-have. Must not drive the verdict.",
    )
    seniority_label: Literal[
        "INTERN", "NEW_GRAD", "ENTRY_LEVEL", "MID_LEVEL", "SENIOR", "LEAD_OR_ABOVE", "UNKNOWN"
    ] = "UNKNOWN"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=600)


class SponsorshipLanguageResult(_LLMResult):
    """Interpretation of ambiguous sponsorship wording in a posting.

    Constrained on purpose: the model classifies wording only. It cannot decide
    whether a job stays in the pipeline, and it must not upgrade historical
    employer data into a claim about this posting -- hence there is no
    ``CONFIRMED`` option and no access to historical records in the prompt.
    """

    polarity: Literal["POSITIVE", "NEGATIVE", "AMBIGUOUS", "ABSENT"]
    confidence: float = Field(ge=0.0, le=1.0)
    quote: str = Field(
        default="",
        max_length=400,
        description="Verbatim sentence from the posting that drove the decision. Empty if none.",
    )
    reasoning: str = Field(default="", max_length=600)


class ExtractionResult(_LLMResult):
    """Fields recovered from a posting whose markup defeated the parser.

    Every field is optional. ``null`` means "not stated" -- the model is
    instructed never to invent a value, because a fabricated date or job ID is
    worse than a missing one.
    """

    company: str | None = None
    job_title: str | None = None
    location: str | None = None
    employment_type: Literal[
        "Full-time", "Contract", "Internship", "Co-op", "Part-time", "Temporary", "Unknown"
    ] = "Unknown"
    remote_type: Literal["Remote", "Hybrid", "On-site", "Unknown"] = "Unknown"
    job_id: str | None = None
    posted_date: str | None = Field(
        default=None, description="ISO 8601 date/datetime exactly as stated in the posting, or null."
    )
    reasoning: str = Field(default="", max_length=600)

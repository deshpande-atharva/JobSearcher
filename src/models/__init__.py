"""Typed domain models shared across every agent and service."""

from src.models.job import (
    AppliedFlag,
    ApplicationStatus,
    DateSource,
    DecisionSource,
    EmploymentType,
    H1BMatchStrength,
    H1BRecord,
    Job,
    JobDecision,
    RawJobPosting,
    RejectionReason,
    RemoteType,
    SponsorshipEvidence,
    SponsorshipScope,
    VisaSponsorshipStatus,
)
from src.models.state import CompanyOutcome, PipelineState, RunSummary

__all__ = [
    "AppliedFlag",
    "ApplicationStatus",
    "CompanyOutcome",
    "DateSource",
    "DecisionSource",
    "EmploymentType",
    "H1BMatchStrength",
    "H1BRecord",
    "Job",
    "JobDecision",
    "PipelineState",
    "RawJobPosting",
    "RejectionReason",
    "RemoteType",
    "RunSummary",
    "SponsorshipEvidence",
    "SponsorshipScope",
    "VisaSponsorshipStatus",
]

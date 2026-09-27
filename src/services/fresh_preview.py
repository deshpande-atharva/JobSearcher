"""Diagnostic classification of timestamp-fresh postings.

This does not accept, reject, or reorder jobs. The production graph still
runs role, seniority, location, and employment before the freshness gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.job import EmploymentType, RejectionReason, RemoteType
from src.services.freshness import date_channel, freshness_timestamp, is_fresh
from src.services.rejection_codes import rejection_code
from src.services.roles import classify_role, semantic_family
from src.services.seniority import classify_seniority
from src.utils.normalization import detect_employment_type, normalize_location

if TYPE_CHECKING:
    from src.models.job import RawJobPosting
    from src.models.state import PipelineState, RejectedJob

__all__ = ["attach_fresh_preview", "build_fresh_preview"]

_AFTER_ROLE = frozenset({RejectionReason.EXTRACTION_FAILED, RejectionReason.ROLE})
_AFTER_SENIORITY = _AFTER_ROLE | {RejectionReason.SENIORITY}
_AFTER_LOCATION = _AFTER_SENIORITY | {RejectionReason.LOCATION}
_AFTER_EMPLOYMENT = _AFTER_LOCATION | {RejectionReason.EMPLOYMENT_TYPE}
_FRESHNESS_GATE = _AFTER_EMPLOYMENT | {RejectionReason.FRESHNESS}


def attach_fresh_preview(state: "PipelineState") -> None:
    """Store the fresh-preview audit on the discovery profile."""
    profile = dict(state.summary.discovery_profile or {})
    profile["fresh_preview_audit"] = build_fresh_preview(state)
    state.summary.discovery_profile = profile


def build_fresh_preview(state: "PipelineState") -> dict:
    """Classify every authoritative fresh raw posting without changing state."""
    hours = state.config.freshness_hours
    use_updated = state.config.settings.run.freshness_use_updated_when_posted_missing
    filters = state.config.settings.filters
    allowed = set(filters.employment_types)
    jobs_by_url = {
        (job.direct_application_url or "").strip(): job
        for job in state.jobs
        if job.direct_application_url
    }
    rows: list[dict] = []
    for posting in state.raw_postings:
        fresh, age = is_fresh(
            posting,
            hours,
            use_updated_when_posted_missing=use_updated,
        )
        if not fresh:
            continue
        rows.append(
            _row(
                state,
                posting,
                age=age,
                use_updated=use_updated,
                allowed=allowed,
                require_us=filters.require_us_location,
                max_years=filters.max_required_years,
                ignore_preferred=filters.ignore_preferred_experience,
                survived=jobs_by_url.get((posting.apply_url or "").strip()) is not None,
            )
        )
    return {
        "count": len(rows),
        "counters": _counters(rows),
        "funnel": _funnel(rows),
        "funnel_by_source": _funnel_by_source(rows),
        "by_pipeline_reason": _reason_counts(rows),
        "jobs": rows,
    }


def _row(
    state: "PipelineState",
    posting: "RawJobPosting",
    *,
    age: float | None,
    use_updated: bool,
    allowed: set[EmploymentType],
    require_us: bool,
    max_years: float,
    ignore_preferred: bool,
    survived: bool,
) -> dict:
    role = classify_role(posting.title, posting.description, state.config.roles)
    seniority = classify_seniority(
        posting.title,
        posting.description,
        state.config.roles,
        max_required_years=max_years,
        ignore_preferred=ignore_preferred,
    )
    loc = normalize_location(posting.location_raw, description=posting.description)
    remote = posting.remote_type_raw
    from src.utils.normalization import detect_remote_type

    remote_type = detect_remote_type(remote, posting.location_raw, posting.title)
    if remote_type is RemoteType.UNKNOWN:
        remote_type = loc.remote_type
    location_pass, location_reason = _location_decision(loc, remote_type, require_us=require_us)
    employment = detect_employment_type(
        posting.employment_type_raw, title=posting.title, description=posting.description
    )
    if employment is EmploymentType.UNKNOWN:
        employment = EmploymentType.FULL_TIME
    employment_pass = employment in allowed
    rejected = None if survived else _match_rejection(state, posting)
    pipeline_reason = None if rejected is None else rejected.reason.value
    code = None if rejected is None else rejection_code(rejected)
    moment = freshness_timestamp(posting, use_updated_when_posted_missing=use_updated)
    decisions = _gate_decisions(pipeline_reason, code)
    return {
        "company": posting.company_name or "",
        "job_id": posting.job_id or "",
        "source": posting.source or "",
        "job_title": (posting.title or "")[:180],
        "title": (posting.title or "")[:180],
        "location": (posting.location_raw or "")[:160],
        "posted_at": posting.posted_at.isoformat() if posting.posted_at else None,
        "updated_at": posting.updated_at.isoformat() if posting.updated_at else None,
        "date_source": posting.date_source.value if posting.date_source else "UNKNOWN",
        "official_url": (posting.apply_url or "")[:300],
        "employment": employment.value,
        "timestamp_source": date_channel(posting).value,
        "authoritative_timestamp": moment.isoformat() if moment else None,
        "age_hours": None if age is None else round(age, 2),
        "freshness": "fresh",
        "role_pass": bool(role.is_software_engineering and not role.needs_llm),
        "role_needs_llm": role.needs_llm,
        "classification_state": role.classification_state,
        "semantic_family": semantic_family(role) if role.needs_llm else "",
        "semantic_status": _semantic_status(role, rejected),
        "role_reason": role.detail[:180],
        "seniority_pass": bool(seniority.fits_entry_level and not seniority.needs_llm),
        "seniority_needs_llm": seniority.needs_llm,
        "seniority_reason": seniority.detail[:180],
        "experience_years": seniority.min_years,
        "experience_pass": bool(seniority.fits_entry_level and not seniority.needs_llm),
        "location_pass": location_pass,
        "location_reason": location_reason,
        "employment_pass": employment_pass,
        "employment_reason": "" if employment_pass else f"employment type {employment.value}",
        "pipeline_reason": pipeline_reason,
        "rejection_code": code,
        "final_rejection_reason": "final_candidate" if survived and rejected is None else (code or pipeline_reason or "unmatched"),
        "final_candidate": bool(survived and rejected is None),
        "role_decision": decisions["role"],
        "seniority_decision": decisions["seniority"],
        "experience_decision": decisions["experience"],
        "location_decision": decisions["location"],
        "employment_decision": decisions["employment"],
        "pipeline_detail": "" if rejected is None or not rejected.detail else rejected.detail[:180],
    }


def _location_decision(loc, remote_type: RemoteType, *, require_us: bool) -> tuple[bool, str]:
    """Mirror the location agent. Bare remote is not treated as U.S."""
    del remote_type
    if loc.is_international_only:
        return False, "international-only"
    us_ok = loc.is_us
    if require_us and not us_ok:
        return False, "could not confirm a U.S. location"
    return True, "us" if us_ok else "not required"


def _match_rejection(state: "PipelineState", posting: "RawJobPosting") -> "RejectedJob | None":
    url = (posting.apply_url or "").strip()
    company = posting.company_name or ""
    title = posting.title or ""
    for item in state.rejected:
        if url and item.url == url:
            return item
    for item in state.rejected:
        if item.company == company and item.title == title and (item.source or "") == (posting.source or ""):
            return item
    return None


def _gate_decisions(reason: str | None, code: str | None) -> dict[str, str]:
    """Pipeline outcome only. A later gate is not_reached when an earlier one rejected."""
    blocked = {
        "role": reason in {"role", "extraction_failed"},
        "seniority": code == "SENIORITY_TOO_HIGH",
        "experience": code == "EXPERIENCE_TOO_HIGH",
        "location": reason == "location",
        "employment": reason == "employment_type",
    }
    order = ("role", "seniority", "experience", "location", "employment")
    reached = True
    decisions = {}
    for name in order:
        if not reached:
            decisions[name] = "not_reached"
            continue
        if name == "role" and code and str(code).startswith("SEMANTIC_REVIEW"):
            decisions[name] = code
            reached = False
            continue
        if name == "seniority" and blocked["experience"]:
            decisions[name] = "pass"
            continue
        if name == "experience" and blocked["seniority"]:
            decisions[name] = "pass"
            continue
        if blocked[name]:
            decisions[name] = "reject"
            reached = False
            continue
        decisions[name] = "pass"
    return decisions


def _semantic_status(role, rejected: "RejectedJob | None") -> str:
    if rejected is not None and rejected.reason is RejectionReason.EXTRACTION_FAILED:
        return "not_reached"
    if not role.needs_llm:
        return role.classification_state
    if rejected is not None and rejected.report_code:
        return rejected.report_code
    if rejected is not None and rejected.reason is RejectionReason.ROLE:
        return "REJECT"
    return "ACCEPT"


def _counters(rows: list[dict]) -> dict[str, int]:
    def decision(row: dict, gate: str, outcome: str) -> bool:
        return row.get(f"{gate}_decision") == outcome

    return {
        "fresh_authoritative_total": len(rows),
        "fresh_role_pass": sum(1 for row in rows if decision(row, "role", "pass")),
        "fresh_role_reject": sum(1 for row in rows if decision(row, "role", "reject")),
        "fresh_seniority_pass": sum(1 for row in rows if decision(row, "seniority", "pass")),
        "fresh_seniority_reject": sum(1 for row in rows if decision(row, "seniority", "reject")),
        "fresh_experience_pass": sum(1 for row in rows if decision(row, "experience", "pass")),
        "fresh_experience_reject": sum(1 for row in rows if decision(row, "experience", "reject")),
        "fresh_location_pass": sum(1 for row in rows if decision(row, "location", "pass")),
        "fresh_location_reject": sum(1 for row in rows if decision(row, "location", "reject")),
        "fresh_employment_pass": sum(1 for row in rows if decision(row, "employment", "pass")),
        "fresh_employment_reject": sum(1 for row in rows if decision(row, "employment", "reject")),
        "fresh_final_candidates": sum(1 for row in rows if row.get("final_candidate")),
    }


def _funnel(rows: list[dict]) -> dict[str, int]:
    def cleared(reason: str | None, blocked: frozenset[RejectionReason]) -> bool:
        if reason is None:
            return True
        try:
            parsed = RejectionReason(reason)
        except ValueError:
            return False
        return parsed not in blocked

    return {
        "fresh_preview": len(rows),
        "fresh_after_role": sum(1 for row in rows if cleared(row["pipeline_reason"], _AFTER_ROLE)),
        "fresh_after_seniority": sum(
            1 for row in rows if cleared(row["pipeline_reason"], _AFTER_SENIORITY)
        ),
        "fresh_after_location": sum(
            1 for row in rows if cleared(row["pipeline_reason"], _AFTER_LOCATION)
        ),
        "fresh_after_employment": sum(
            1 for row in rows if cleared(row["pipeline_reason"], _AFTER_EMPLOYMENT)
        ),
        "freshness_gate": sum(1 for row in rows if cleared(row["pipeline_reason"], _FRESHNESS_GATE)),
    }


def _funnel_by_source(rows: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("source") or "unknown"), []).append(row)
    return {name: _funnel(items) for name, items in grouped.items()}


def _reason_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row["pipeline_reason"] or "reached_freshness_or_later"
        counts[key] = counts.get(key, 0) + 1
    return counts

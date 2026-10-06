"""Concise observability for a daily run. No secrets and no resume text."""

from __future__ import annotations

import json

from src.models.state import PipelineState
from src.services.rejection_codes import rejection_counts
from src.services.roles import REPORT_FAMILIES

__all__ = [
    "production_source_state",
    "render_pipeline_health",
    "source_completeness",
    "source_contract",
]


def _semantic_family_line(summary) -> str:
    buckets = summary.semantic_by_family or {}
    parts = []
    for name in REPORT_FAMILIES:
        row = buckets.get(name) or {}
        parts.append(
            f"{name}:required={row.get('required', 0)} "
            f"attempted={row.get('attempted', 0)} "
            f"accepted={row.get('accepted', 0)} "
            f"rejected={row.get('rejected', 0)} "
            f"uncertain={row.get('uncertain', 0)} "
            f"unavailable={row.get('unavailable', 0)}"
        )
    return "semantic_review_by_family: " + " | ".join(parts)


def _manifest_text(state: PipelineState) -> str:
    """In-memory run record for the log. Not a database and not a secret store."""
    summary = state.summary
    profile = summary.discovery_profile or {}
    preview = (profile.get("freshness_preview") or {}).get("by_source") or {}
    incomplete = [
        board.get("company")
        for board in (profile.get("workday_boards") or [])
        if board.get("estimated_incomplete")
    ]
    complete = partial = failed = empty = 0
    for name, health in summary.source_health.items():
        label = source_completeness(name, health, profile)
        if label == "COMPLETE":
            complete += 1
        elif label == "PARTIAL":
            partial += 1
        elif label == "FAILED":
            failed += 1
        elif label == "EMPTY":
            empty += 1
    payload = {
        "discovered_jobs": summary.jobs_discovered,
        "deduplicated_jobs": summary.jobs_after_cross_source_dedup,
        "fresh_jobs": summary.fresh_authoritative_jobs,
        "final_candidates": summary.jobs_accepted,
        "deterministic_accepts": summary.deterministic_accepts,
        "deterministic_rejects": summary.deterministic_rejects,
        "semantic_review_required": summary.semantic_review_required,
        "semantic_review_attempted": summary.semantic_review_attempted,
        "semantic_review_accepted": summary.semantic_review_accepted,
        "semantic_review_rejected": summary.semantic_review_rejected,
        "semantic_review_uncertain": summary.semantic_review_uncertain,
        "semantic_review_unavailable": summary.semantic_review_unavailable,
        "semantic_reviews_blocked_by_circuit": summary.semantic_reviews_blocked_by_circuit,
        "fresh_semantic_review_required": summary.fresh_semantic_review_required,
        "fresh_semantic_review_attempted": summary.fresh_semantic_review_attempted,
        "fresh_semantic_review_accepted": summary.fresh_semantic_review_accepted,
        "fresh_semantic_review_rejected": summary.fresh_semantic_review_rejected,
        "fresh_semantic_review_uncertain": summary.fresh_semantic_review_uncertain,
        "fresh_semantic_review_unavailable": summary.fresh_semantic_review_unavailable,
        "source_complete": complete,
        "source_partial": partial,
        "source_failed": failed,
        "source_empty": empty,
        "llm_state_before": summary.llm_state_before,
        "llm_state_after": summary.llm_state_after or summary.llm_circuit_state,
        "email_status": summary.email_status,
        "xlsx_status": summary.xlsx_status,
        "archive_status": summary.archive_status,
        "runtime_seconds": None if summary.duration_seconds is None else round(summary.duration_seconds, 3),
        "exit_code": summary.exit_code,
        "run_started_utc": summary.run_started_at.isoformat() if summary.run_started_at else None,
        "run_finished_utc": summary.run_finished_at.isoformat() if summary.run_finished_at else None,
        "freshness_hours": summary.freshness_hours_used,
        "source_status": {
            name: production_source_state(health) for name, health in sorted(summary.source_health.items())
        },
        "source_counts": {
            name: health.jobs_discovered for name, health in sorted(summary.source_health.items())
        },
        "fresh_counts": {
            "authoritative": summary.fresh_authoritative_jobs,
            "by_source": {
                name: {
                    "fresh": (bucket or {}).get("fresh", 0),
                    "stale": (bucket or {}).get("stale", 0),
                    "unknown": (bucket or {}).get("unknown", 0),
                }
                for name, bucket in sorted(preview.items())
            },
        },
        "semantic_review_counts": {
            "required": summary.semantic_review_required,
            "attempted": summary.semantic_review_attempted,
            "accepted": summary.semantic_review_accepted,
            "rejected": summary.semantic_review_rejected,
            "uncertain": summary.semantic_review_uncertain,
            "unavailable": summary.semantic_review_unavailable,
        },
        "llm_stats": {
            "state_before": summary.llm_state_before,
            "state_after": summary.llm_state_after or summary.llm_circuit_state,
            "calls": summary.llm_calls,
            "successes": summary.llm_successes,
            "failures": summary.llm_failures,
            "retries": summary.llm_retries,
            "circuit_skips": summary.llm_circuit_skips,
            "semantic_reviews_blocked_by_circuit": summary.semantic_reviews_blocked_by_circuit,
        },
        "workday_incomplete_boards": incomplete,
        "final_candidate_count": summary.jobs_accepted,
        "archive_path": summary.archive_path,
        "current_path": summary.workbook_path,
    }
    return json.dumps(payload, separators=(",", ":"), default=str)


def _stage_runtime_line(stages: dict, total: float) -> str:
    def pick(*names: str) -> float:
        return round(sum(float(stages.get(name) or 0) for name in names), 3)

    return (
        "stage_runtime: "
        f"total_runtime={total:.3f} "
        f"discovery_runtime={pick('run_discovery')} "
        f"normalization_runtime={pick('run_extraction')} "
        f"dedup_runtime={pick('run_dedup')} "
        f"role_runtime={pick('run_role_classification')} "
        f"seniority_runtime={pick('run_seniority')} "
        f"experience_runtime={pick('run_seniority')} "
        f"location_runtime={pick('location_gate')} "
        f"employment_runtime={pick('employment_gate')} "
        f"freshness_runtime={pick('run_freshness')} "
        f"url_runtime={pick('run_url_verification')} "
        f"h1b_runtime={pick('run_h1b_enrichment')} "
        f"job_intelligence_runtime={pick('job_intelligence_gate')} "
        f"critic_runtime={pick('critic_gate')} "
        f"xlsx_runtime={round(float(stages['xlsx']), 3) if 'xlsx' in stages else pick('run_output')} "
        f"email_runtime={pick('email')} "
        "note=experience_is_inside_seniority;company_times_overlap"
    )


def source_completeness(name: str, health, profile: dict | None = None) -> str:
    """COMPLETE only when discovery finished without failures or a list cap.

    A capped Workday board is PARTIAL even if every request succeeded.
    """
    state = production_source_state(health)
    if state in {"DISABLED", "FAILED", "EMPTY"}:
        return state
    caps = int(((profile or {}).get("by_source") or {}).get(name, {}).get("caps") or 0)
    if state == "PARTIAL" or caps:
        return "PARTIAL"
    if state == "SUCCESS":
        return "COMPLETE"
    return state


def source_contract(
    name: str,
    health,
    profile: dict | None = None,
    *,
    jobs_after_dedup: int = 0,
    fresh_jobs: int = 0,
    unknown_timestamp_jobs: int = 0,
    failed_companies: list[str] | None = None,
) -> dict:
    """One source row. A documented list cap makes an otherwise successful source PARTIAL."""
    profile = profile or {}
    caps = int(((profile.get("by_source") or {}).get(name) or {}).get("caps") or 0)
    boards = [
        board
        for board in (profile.get("workday_boards") or [])
        if name == "workday" and board.get("estimated_incomplete")
    ]
    incomplete = caps or len(boards)
    status = production_source_state(health)
    if status == "SUCCESS" and incomplete:
        status = "PARTIAL"
    return {
        "source": name,
        "status": status,
        "companies_attempted": health.attempted,
        "companies_succeeded": health.successful,
        "companies_failed": health.failed,
        "jobs_discovered": health.jobs_discovered,
        "jobs_after_dedup": jobs_after_dedup,
        "fresh_jobs": fresh_jobs,
        "unknown_timestamp_jobs": unknown_timestamp_jobs,
        "incomplete_boards": incomplete,
        "failed_companies": list(failed_companies or []),
    }


def production_source_state(health) -> str:
    """Map company outcomes to one deterministic source state.

    SUCCESS: the source responded and at least one record was processed, with no
    company failure.
    EMPTY: requests succeeded and zero records came back.
    PARTIAL: at least one company succeeded and at least one company failed.
    FAILED: every attempted company failed, so discovery produced nothing useful.
    DISABLED: the source was not attempted.
    """
    if health.attempted == 0:
        return "DISABLED"
    if health.failed and health.successful == 0:
        return "FAILED"
    if health.failed and health.successful:
        return "PARTIAL"
    if health.jobs_discovered:
        return "SUCCESS"
    return "EMPTY"


def _global_board_line(summary) -> str:
    profile = (summary.discovery_profile or {}).get("global_boards") or {}
    if not profile:
        return "global_boards: not_run coverage=not_queried"
    parts = [
        f"status={profile.get('status', 'unknown')}",
        f"coverage={profile.get('coverage', 'partial')}",
    ]
    for name in ("greenhouse", "ashby", "workday"):
        board = profile.get(name) or {}
        if not board or board.get("status") == "disabled":
            continue
        parts.append(
            f"{name}_discovered={board.get('discovered_boards', 0)} "
            f"{name}_valid={board.get('valid_boards', 0)} "
            f"{name}_invalid={board.get('invalid_boards', 0)} "
            f"{name}_failed={board.get('failed_boards', 0)} "
            f"{name}_empty={board.get('empty_boards', 0)} "
            f"{name}_complete={board.get('complete_boards', 0)} "
            f"{name}_partial={board.get('partial_boards', 0)} "
            f"{name}_duplicate={board.get('duplicate_boards', 0)} "
            f"{name}_configured_overlap={board.get('configured_overlap', 0)} "
            f"{name}_selected={board.get('selected_boards', 0)} "
            f"{name}_jobs={board.get('jobs', 0)}"
        )
    return "global_boards: " + " ".join(parts)


def _workday_global_line(state: PipelineState) -> str:
    """Configured boards plus the bounded archive sample. Coverage stays partial."""
    profile = (state.summary.discovery_profile or {}).get("global_boards") or {}
    board = profile.get("workday") or {}
    if not board or board.get("status") == "disabled":
        return "workday_global: disabled GLOBAL DISCOVERY: PARTIAL COVERAGE"
    skipped = board.get("duplicate_boards_skipped")
    if skipped is None:
        skipped = int(board.get("duplicate_boards") or 0) + int(
            board.get("configured_overlap") or 0
        )
    successful = int(board.get("complete_boards") or 0) + int(board.get("partial_boards") or 0)
    selected = board.get("boards_selected_for_collection", board.get("selected_boards", 0))
    collected = board.get(
        "unique_boards_collected",
        successful + int(board.get("empty_boards") or 0),
    )
    return (
        "workday_global: GLOBAL DISCOVERY: PARTIAL COVERAGE "
        f"configured_boards={board.get('configured_boards', 0)} "
        f"globally_discovered_boards={board.get('discovered_boards', 0)} "
        f"duplicate_boards_skipped={skipped} "
        f"boards_selected_for_collection={selected} "
        f"unique_boards_collected={collected} "
        f"successful_boards={successful} "
        f"complete_boards={board.get('complete_boards', 0)} "
        f"failed_boards={board.get('failed_boards', 0)} "
        f"empty_boards={board.get('empty_boards', 0)} "
        f"partial_boards={board.get('partial_boards', 0)} "
        f"jobs_discovered={board.get('jobs', 0)} "
        f"index_seconds={profile.get('index_seconds', 0)} "
        f"collection_seconds={board.get('collection_seconds', 0)}"
    )


def _freshness_tier_line(state: PipelineState) -> str:
    policy = state.config.settings.freshness
    tiers = policy.tiers
    counts = state.summary.freshness_tiers or {}
    rendered = " ".join(f"{name}={counts.get(name, 0)}" for name in (
        "VERY_FRESH",
        "FRESH",
        "RECENT",
        "AGING",
        "STALE",
        "OLD",
        "UNKNOWN",
    ))
    return (
        "freshness_tiers: "
        f"enabled={str(policy.enabled).lower()} "
        f"eligible_through={policy.eligible_through} "
        f"cuts={tiers.very_fresh_hours:g}/{tiers.fresh_hours:g}/{tiers.recent_hours:g}/"
        f"{tiers.aging_hours:g}/{tiers.stale_hours:g} "
        + rendered
    )


def render_pipeline_health(state: PipelineState) -> str:
    summary = state.summary
    sources = state.config.settings.discovery.sources
    reachable = sum(1 for job in state.jobs if job.url_verified)
    accepted_urls = sum(1 for job in state.jobs if job.direct_application_url)
    finished = summary.run_finished_at or summary.run_started_at
    runtime = (finished - summary.run_started_at).total_seconds()
    after_role = summary.remaining_after("role")
    after_seniority = summary.remaining_after("role", "seniority")
    after_location = summary.remaining_after("role", "seniority", "location")
    after_employment = summary.remaining_after("role", "seniority", "location", "employment")
    after_fresh = summary.remaining_after("role", "seniority", "location", "employment", "freshness")
    after_url = summary.remaining_after("role", "seniority", "location", "employment", "freshness", "url")
    lines = [
        "MULTI-SOURCE PRODUCTION READINESS",
        f"run_timestamp_utc: {finished.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"run_started_utc: {summary.run_started_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"runtime_seconds: {runtime:.1f}",
        f"freshness_hours: {state.config.freshness_hours:g}",
        f"fixture_mode: {str(state.config.fixture_mode).lower()}",
        f"dry_run: {str(state.config.dry_run).lower()}",
        f"workday_browser_enabled: {str(sources.workday.browser_enabled).lower()}",
        f"jobright_enabled: {str(sources.jobright.enabled).lower()}",
        _global_board_line(summary),
        _workday_global_line(state),
        _freshness_tier_line(state),
        (
            "source_caps: "
            f"greenhouse=full-board "
            f"workday_max_jobs={sources.workday.max_jobs} "
            f"lever_max_jobs={sources.lever.max_jobs} "
            f"ashby_max_jobs={sources.ashby.max_jobs} "
            f"max_jobs_per_run={state.config.settings.run.max_jobs_per_run} "
            f"lever_timestamp_order=when_cap_binds "
            f"ashby_timestamp_order=when_cap_binds"
        ),
        (
            "qualification: "
            f"discovered={summary.jobs_discovered} "
            f"normalized={summary.jobs_extracted} "
            f"deduplicated={summary.jobs_after_cross_source_dedup} "
            f"role={after_role} "
            f"seniority={after_seniority} "
            f"location={after_location} "
            f"employment={after_employment} "
            f"freshness={after_fresh} "
            f"url={after_url} "
            f"h1b={after_url} "
            f"intelligence={summary.intelligence_evaluated} "
            f"critic={summary.critic_reviewed} "
            f"final={summary.jobs_accepted}"
        ),
        (
            "gate_counts: "
            f"location_pass={after_location} location_reject={summary.rejected_by_location} "
            f"employment_pass={after_employment} employment_reject={summary.rejected_by_employment_type}"
        ),
        f"rejections: {rejection_counts(state.rejected) or {}}",
        f"url_accepted: {accepted_urls}",
        f"url_verified: {reachable}",
        (
            "h1b: "
            f"confirmed={summary.h1b_confirmed} likely={summary.h1b_likely} "
            f"unknown={summary.h1b_unknown} not_supported={summary.h1b_not_supported} "
            f"lookup_failures={summary.h1b_lookup_failures}"
        ),
        (
            "llm: "
            f"calls={summary.llm_calls} successes={summary.llm_successes} "
            f"failures={summary.llm_failures} retries={summary.llm_retries} "
            f"deterministic_avoided={summary.llm_deterministic_avoided} "
            f"circuit_skips={summary.llm_circuit_skips} fallbacks={summary.llm_fallbacks} "
            f"state={summary.llm_circuit_state} by_purpose={summary.llm_by_purpose or {}}"
        ),
        "llm_circuit_scope: in-process; a new run starts CLOSED",
        f"email: {summary.email_status}",
        (
            "email_status: "
            f"email_skipped={str(summary.email_skipped).lower()} "
            f"email_failed={str(summary.email_failed).lower()} "
            f"email_reason={summary.email_reason or 'none'}"
        ),
        (
            "semantic_review: "
            f"semantic_review_required={summary.semantic_review_required} "
            f"semantic_review_attempted={summary.semantic_review_attempted} "
            f"semantic_review_accepted={summary.semantic_review_accepted} "
            f"semantic_review_rejected={summary.semantic_review_rejected} "
            f"semantic_review_uncertain={summary.semantic_review_uncertain} "
            f"semantic_review_unavailable={summary.semantic_review_unavailable}"
        ),
        _semantic_family_line(summary),
        (
            "fresh_semantic_review: "
            f"fresh_semantic_review_required={summary.fresh_semantic_review_required} "
            f"fresh_semantic_review_attempted={summary.fresh_semantic_review_attempted} "
            f"fresh_semantic_review_accepted={summary.fresh_semantic_review_accepted} "
            f"fresh_semantic_review_rejected={summary.fresh_semantic_review_rejected} "
            f"fresh_semantic_review_uncertain={summary.fresh_semantic_review_uncertain} "
            f"fresh_semantic_review_unavailable={summary.fresh_semantic_review_unavailable}"
        ),
        (
            "role_decisions: "
            f"deterministic_accepts={summary.deterministic_accepts} "
            f"deterministic_rejects={summary.deterministic_rejects}"
        ),
        (
            "semantic_outcome_split: "
            f"semantic_review_unavailable={summary.semantic_review_unavailable} "
            "is a hold for missing model review and is not a role mismatch; "
            f"deterministic_rejects={summary.deterministic_rejects}; "
            f"fresh_semantic_review_required={summary.fresh_semantic_review_required}"
        ),
        (
            "llm_circuit: "
            f"llm_state_before={summary.llm_state_before} "
            f"llm_state_after={summary.llm_state_after or summary.llm_circuit_state} "
            f"llm_calls={summary.llm_calls} "
            f"llm_successes={summary.llm_successes} "
            f"llm_failures={summary.llm_failures} "
            f"llm_retries={summary.llm_retries} "
            f"llm_circuit_skips={summary.llm_circuit_skips} "
            f"semantic_reviews_blocked_by_circuit={summary.semantic_reviews_blocked_by_circuit}"
        ),
        f"workbook: {summary.workbook_path or 'not written'}",
        f"archive: {summary.archive_path or 'not written'}",
        f"xlsx_status: {summary.xlsx_status}",
        f"archive_status: {summary.archive_status}",
        f"stage_seconds: {summary.stage_seconds or {}}",
        _stage_runtime_line(summary.stage_seconds or {}, runtime),
        "run_manifest: " + _manifest_text(state),
    ]
    profile = summary.discovery_profile or {}
    if profile:
        lines.append(
            "discovery_profile: "
            f"http_requests={profile.get('http_requests', 0)} "
            f"cache_hits={profile.get('http_cache_hits', 0)} "
            f"retries={profile.get('http_retries', 0)} "
            f"playwright_renders={profile.get('playwright_renders', 0)} "
            f"playwright_failures={profile.get('playwright_failures', 0)} "
            f"playwright_seconds={profile.get('playwright_seconds', 0)}"
        )
        by_source = profile.get("by_source") or {}
        if by_source:
            summed = []
            walls = []
            for name in sorted(by_source):
                bucket = by_source[name]
                summed.append(
                    f"{name}={bucket.get('seconds_sum', bucket.get('seconds', 0))}s/{bucket.get('jobs', 0)}jobs"
                )
                walls.append(f"{name}={bucket.get('wall_seconds', 0)}s")
            lines.append("source_seconds_sum: " + " ".join(summed))
            lines.append("source_wall_seconds: " + " ".join(walls))
            lines.append(
                "timing_note: source_seconds_sum overlaps across companies; "
                "source_wall_seconds is the union of those intervals and still overlaps other sources"
            )
        reasons = profile.get("fallback_reasons") or {}
        if reasons:
            lines.append(
                "career_fallback_reasons: "
                + " ".join(f"{key}={reasons[key]}" for key in sorted(reasons))
            )
        partitions = profile.get("workday_partitions") or {}
        if partitions:
            lines.append(
                "workday_partitions: "
                f"capped_companies={partitions.get('companies_capped', 0)} "
                f"attempted={partitions.get('attempted', 0)} "
                f"successful={partitions.get('successful', 0)} "
                f"failed={partitions.get('failed', 0)} "
                f"new_jobs={partitions.get('jobs', 0)} "
                f"duplicates={partitions.get('duplicates', 0)} "
                f"requests={partitions.get('requests', 0)} "
                f"recovered_fresh={partitions.get('recovered_fresh_jobs', 0)}"
            )
        lines.append(
            "workday_filters: appliedFacets={} searchText=only limit=20 "
            "unsupported_filters=none capped_boards_are_incomplete=true"
        )
        for board in profile.get("workday_boards") or []:
            lines.append(
                "workday_board: "
                f"company={board.get('company')} "
                f"unfiltered={board.get('unfiltered_jobs')} "
                f"reported_total={board.get('reported_total')} "
                f"cap_reached={str(bool(board.get('cap_reached'))).lower()} "
                f"partitions={board.get('partitions_attempted')} "
                f"partition_requests={board.get('partition_requests')} "
                f"new_jobs={board.get('partition_jobs')} "
                f"duplicates={board.get('duplicates')} "
                f"unique={board.get('unique_count')} "
                f"incomplete={str(bool(board.get('estimated_incomplete'))).lower()} "
                f"unfiltered_fresh={board.get('unfiltered_fresh_jobs')} "
                f"recovered_fresh={board.get('recovered_fresh_jobs')}"
            )
        preview = profile.get("freshness_preview") or {}
        if preview:
            lines.append(
                "freshness_preview: "
                f"fresh={preview.get('fresh', 0)} "
                f"stale={preview.get('stale', 0)} "
                f"unknown={preview.get('unknown', 0)}"
            )
            by_source = preview.get("by_source") or {}
            for name in sorted(by_source):
                bucket = by_source[name] or {}
                lines.append(
                    "freshness_by_source: "
                    f"{name} fresh={bucket.get('fresh', 0)} "
                    f"stale={bucket.get('stale', 0)} "
                    f"unknown={bucket.get('unknown', 0)}"
                )
        attribution = profile.get("freshness_attribution") or {}
        for name in sorted(attribution):
            row = attribution[name] or {}
            lines.append(
                "freshness_attribution: "
                f"source={name} "
                f"discovered={row.get('discovered', 0)} "
                f"posted_known={row.get('posted_known', 0)} "
                f"updated_known={row.get('updated_known', 0)} "
                f"posted_unknown={row.get('posted_unknown', 0)} "
                f"fresh={row.get('fresh', 0)} "
                f"stale={row.get('stale', 0)} "
                f"unknown={row.get('unknown', 0)} "
                f"fresh_rate={row.get('fresh_rate', 0)} "
                f"unknown_rate={row.get('unknown_rate', 0)}"
            )
        date_sources = profile.get("date_source_counts") or {}
        if date_sources:
            lines.append(
                "freshness_date_sources: "
                + " ".join(f"{key}={date_sources.get(key, 0)}" for key in (
                    "POSTED_DATE",
                    "UPDATED_DATE",
                    "DISCOVERED_DATE",
                    "UNKNOWN",
                ))
            )
        audit = profile.get("fresh_preview_audit") or {}
        counters = audit.get("counters") or {}
        if counters:
            lines.append(
                "fresh_job_audit: "
                + " ".join(f"{key}={counters.get(key, 0)}" for key in (
                    "fresh_authoritative_total",
                    "fresh_role_pass",
                    "fresh_role_reject",
                    "fresh_seniority_pass",
                    "fresh_seniority_reject",
                    "fresh_experience_pass",
                    "fresh_experience_reject",
                    "fresh_location_pass",
                    "fresh_location_reject",
                    "fresh_employment_pass",
                    "fresh_employment_reject",
                    "fresh_final_candidates",
                ))
            )
        for job in audit.get("jobs") or []:
            lines.append(
                "fresh_job: "
                f"company={job.get('company')} "
                f"source={job.get('source')} "
                f"job_id={job.get('job_id')} "
                f"title={job.get('job_title') or job.get('title')} "
                f"location={job.get('location')} "
                f"posted_at={job.get('posted_at')} "
                f"updated_at={job.get('updated_at')} "
                f"date_source={job.get('date_source')} "
                f"role={job.get('role_decision')} "
                f"seniority={job.get('seniority_decision')} "
                f"experience={job.get('experience_decision')} "
                f"location_decision={job.get('location_decision')} "
                f"employment={job.get('employment_decision')} "
                f"reason={job.get('final_rejection_reason')} "
                f"url={job.get('official_url')}"
            )
            if job.get("role_needs_llm"):
                lines.append(
                    "semantic_fresh_job: "
                    f"company={job.get('company')} "
                    f"job_id={job.get('job_id')} "
                    f"title={job.get('job_title') or job.get('title')} "
                    f"location={job.get('location')} "
                    f"date_source={job.get('date_source')} "
                    f"semantic_family={job.get('semantic_family')} "
                    f"semantic_status={job.get('semantic_status')} "
                    f"final_reason={job.get('final_rejection_reason')}"
                )
        funnel = audit.get("funnel") or {}
        if funnel:
            lines.append(
                "freshness_stage_funnel: "
                f"preview={funnel.get('fresh_preview', 0)} "
                f"after_role={funnel.get('fresh_after_role', 0)} "
                f"after_seniority={funnel.get('fresh_after_seniority', 0)} "
                f"after_location={funnel.get('fresh_after_location', 0)} "
                f"after_employment={funnel.get('fresh_after_employment', 0)} "
                f"freshness_gate={funnel.get('freshness_gate', 0)}"
            )
        stage_by_source = audit.get("funnel_by_source") or {}
        source_names = sorted(set((preview.get("by_source") or {})) | set(stage_by_source))
        for name in source_names:
            stage = stage_by_source.get(name) or {}
            lines.append(
                "freshness_stage_by_source: "
                f"{name} "
                f"preview={stage.get('fresh_preview', 0)} "
                f"after_role={stage.get('fresh_after_role', 0)} "
                f"after_seniority={stage.get('fresh_after_seniority', 0)} "
                f"after_location={stage.get('fresh_after_location', 0)} "
                f"after_employment={stage.get('fresh_after_employment', 0)} "
                f"freshness_gate={stage.get('freshness_gate', 0)}"
            )
        slowest = profile.get("slowest_companies") or []
        if slowest:
            lines.append(
                "slowest_companies: "
                + " | ".join(
                    f"{item.get('company')} wall={item.get('seconds')}s source={item.get('source')} "
                    f"requests={item.get('requests', 0)} retries={item.get('retries', 0)} "
                    f"playwright_renders={item.get('playwright_renders', 0)} "
                    f"status={item.get('status') or ('ok' if item.get('ok') else 'failed')} "
                    f"jobs={item.get('jobs')} cap={str(bool(item.get('cap'))).lower()}"
                    for item in slowest[:8]
                )
            )
        page_outcomes = profile.get("career_page_outcomes") or {}
        if page_outcomes:
            lines.append(
                "career_page_outcomes: "
                + " ".join(f"{key}={page_outcomes[key]}" for key in sorted(page_outcomes))
            )
        detection = profile.get("ats_detection") or {}
        if detection:
            lines.append(
                "ats_detection: "
                f"automatic={detection.get('automatic', 0)} "
                f"uncertain={detection.get('uncertain', 0)} "
                f"rejected_skipped={detection.get('rejected_skipped', 0)}"
            )
    if summary.source_health:
        rendered = []
        for name in sorted(summary.source_health):
            health = summary.source_health[name]
            rendered.append(
                f"{name}={production_source_state(health)}:{health.jobs_discovered}"
            )
        lines.append("source_health: " + " ".join(rendered))
        completeness = [
            f"{name}={source_completeness(name, summary.source_health[name], profile)}"
            for name in sorted(summary.source_health)
        ]
        lines.append("source_completeness: " + " ".join(completeness))
        deduped: dict[str, int] = {}
        for posting in state.raw_postings:
            source_name = posting.source or "unknown"
            deduped[source_name] = deduped.get(source_name, 0) + 1
        failed_by_source: dict[str, list[str]] = {}
        for outcome in state.company_outcomes:
            if not outcome.succeeded:
                failed_by_source.setdefault(outcome.source, []).append(outcome.company)
        for name in sorted(summary.source_health):
            health = summary.source_health[name]
            attr = (profile.get("freshness_attribution") or {}).get(name) or {}
            preview_bucket = ((profile.get("freshness_preview") or {}).get("by_source") or {}).get(name) or {}
            contract = source_contract(
                name,
                health,
                profile,
                jobs_after_dedup=deduped.get(name, 0),
                fresh_jobs=int(attr.get("fresh") or preview_bucket.get("fresh") or 0),
                unknown_timestamp_jobs=int(attr.get("unknown") or preview_bucket.get("unknown") or 0),
                failed_companies=failed_by_source.get(name, []),
            )
            failed = ",".join(contract["failed_companies"]) or "none"
            lines.append(
                "source_contract: "
                f"source={contract['source']} status={contract['status']} "
                f"companies_attempted={contract['companies_attempted']} "
                f"companies_succeeded={contract['companies_succeeded']} "
                f"companies_failed={contract['companies_failed']} "
                f"jobs_discovered={contract['jobs_discovered']} "
                f"jobs_after_dedup={contract['jobs_after_dedup']} "
                f"fresh_jobs={contract['fresh_jobs']} "
                f"unknown_timestamp_jobs={contract['unknown_timestamp_jobs']} "
                f"incomplete_boards={contract['incomplete_boards']} "
                f"failed_companies={failed}"
            )
            label = source_completeness(name, health, profile)
            lines.append(
                "source_freshness: "
                f"source={name} "
                f"discovered={attr.get('discovered', health.jobs_discovered)} "
                f"posted_known={attr.get('posted_known', 0)} "
                f"updated_known={attr.get('updated_known', 0)} "
                f"unknown={attr.get('unknown', preview_bucket.get('unknown', 0))} "
                f"fresh={attr.get('fresh', preview_bucket.get('fresh', 0))} "
                f"stale={attr.get('stale', preview_bucket.get('stale', 0))} "
                f"fresh_rate={attr.get('fresh_rate', 0)} "
                f"unknown_rate={attr.get('unknown_rate', 0)} "
                f"complete={label}"
            )
        career = summary.source_health.get("company_career")
        if career is not None:
            outcomes = profile.get("career_page_outcomes") or {}
            lines.append(
                "career_fallback: "
                f"companies_attempted={career.attempted} "
                f"success={career.successful} "
                f"hard_failures={career.failed} "
                f"jobs_discovered={career.jobs_discovered} "
                f"jobs_after_dedup={deduped.get('company_career', 0)} "
                f"fresh_jobs={int(((profile.get('freshness_attribution') or {}).get('company_career') or {}).get('fresh') or 0)} "
                f"unsupported_structure={outcomes.get('UNSUPPORTED_STRUCTURE', 0)} "
                f"forbidden={outcomes.get('HTTP_FORBIDDEN', 0)} "
                f"robots_disallowed={outcomes.get('ROBOTS_DISALLOWED', 0)} "
                f"not_found={outcomes.get('HTTP_NOT_FOUND', 0)} "
                f"http_error={outcomes.get('HTTP_ERROR', 0)} "
                f"js_shell={outcomes.get('JS_SHELL', 0)}"
            )
    if summary.jobs_truncated:
        lines.append(f"coverage_cap: max_jobs_per_run truncated={summary.jobs_truncated}")
    if summary.learning_report:
        lines.append(summary.learning_report)
    for attempt in list(state.resources.get("board_search_attempts") or []):
        if attempt.get("skipped") or "error" in attempt:
            continue
        lines.append(
            "board_search_slot: "
            f"key={attempt.get('key')} "
            f"board={attempt.get('board')} "
            f"strategy_id={attempt.get('strategy_id')} "
            f"method={attempt.get('method')} "
            f"raw_jobs={attempt.get('raw_jobs', 0)} "
            f"qualified={attempt.get('qualified_jobs', 0)} "
            f"new={attempt.get('new_qualified_jobs', 0)} "
            f"repeat={attempt.get('repeat_qualified_jobs', 0)} "
            f"exact_urls={attempt.get('exact_official_url_count', 0)} "
            f"url_failures={attempt.get('url_failure_count', 0)} "
            f"duration_seconds={attempt.get('duration_seconds')} "
            f"search_success={str(attempt.get('search_success', '')).lower()}"
        )
    return "\n".join(lines)

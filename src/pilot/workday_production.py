"""Opt-in Workday production smoke.

Exercises browser discovery through the existing qualification, intelligence,
and critic path. The production workbook is not written. Email is not sent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.agents.critic_agent import review_fit
from src.agents.job_intelligence_agent import evaluate_job
from src.llm.base import build_llm_provider
from src.models.config import AppConfig
from src.models.job import RejectionReason
from src.navigation.memory import NavigationMemory
from src.pilot.workday import WORKDAY_TENANTS
from src.pilot.workday_qualify import qualify_workday_postings
from src.pilot.workday_target import discover_target_jobs, run_recorded_keyword_passes, target_queries
from src.services.candidate_profile import load_candidate_profile, merge_resume_profile
from src.services.resume_store import load_profile
from src.services.xlsx import write_workbooks
from src.sources.base import HttpClient

__all__ = ["run_workday_live_e2e", "run_workday_production_smoke"]


async def run_workday_production_smoke(config: AppConfig) -> int:
    config = config.model_copy(update={"dry_run": True, "send_email": False})
    company, board = WORKDAY_TENANTS[0]
    production = config.project_root / config.settings.output.current_workbook
    before = production.stat().st_mtime_ns if production.exists() else None

    print("WORKDAY PHASE 3 PRODUCTION SMOKE")
    print(f"company: {company}")
    print(f"board: {board}")
    print(f"browser_enabled setting: {config.settings.discovery.sources.workday.browser_enabled}")
    print("dry-run: workbook will not be written")

    try:
        found = await discover_target_jobs(config, board_url=board, company_name=company)
    except Exception as exc:
        print(f"BLOCKED: {exc}")
        return 2
    if found.blocked and not found.discovered:
        print(f"BLOCKED: {found.blocked}")
        return 2

    provider = build_llm_provider(config)
    funnel = await qualify_workday_postings(config, found.detailed, llm=provider)
    close = getattr(provider, "aclose", None)
    if callable(close):
        await close()

    profile = merge_resume_profile(
        load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml"),
        config.project_root / config.settings.candidate.profile_path,
    )
    canonical = load_profile(config.project_root / config.settings.candidate.profile_path)
    by_id = {posting.job_id: posting for posting in found.detailed if posting.job_id}
    intelligence = 0
    critic = 0
    unsupported = 0
    if canonical is None or profile.profile_version is None:
        print("resume profile: MISSING")
    else:
        print(f"profile_version: {canonical.profile_version}")
        print(f"profile_sha256: {canonical.resume.sha256}")
        for job in funnel.state.jobs:
            posting = by_id.get(job.job_id or "")
            if posting is None:
                continue
            fit = evaluate_job(posting, profile, config.roles)
            critique = review_fit(fit, posting, profile)
            intelligence += 1
            critic += 1
            unsupported += len(critique.unsupported_claims)
            print(
                f"fit {job.job_id}: decision={critique.fit.decision} "
                f"role={critique.fit.role_alignment} experience={critique.fit.experience_alignment} "
                f"education={critique.fit.education_alignment} sponsorship={critique.fit.sponsorship} "
                f"freshness={critique.fit.freshness} issues={len(critique.issues)}"
            )

    isolated = _isolated_workbook(funnel.state.jobs)
    after = production.stat().st_mtime_ns if production.exists() else None
    workbook_unchanged = before == after

    print("discovered_titles:")
    for posting in found.discovered[:12]:
        print(f"  {posting.job_id or '-'} | {posting.title}")
    print("detail_titles:")
    for posting in found.detailed:
        print(f"  {posting.job_id} | {posting.title} | {posting.location_raw or 'UNKNOWN'}")
    print(f"queries: {', '.join(found.queries)}")
    print(f"queries_applied: {found.queries_applied}")
    print(f"search: {found.search}")
    print(f"strategy_reused: {found.strategy_reused}")
    print(f"recovery: {found.recovery}")
    print(f"navigation_model_calls: {found.navigation_model_calls}")
    print(f"discovered_count: {found.discovered_count}")
    print(f"title_skipped: {found.title_skipped}")
    print(f"unopened_plausible: {found.unopened_plausible}")
    print(f"detail_pages: {len(found.detailed)}")
    print(f"detail_extraction_failures: {found.detail_failures}")
    print(f"browser_pages_opened: {found.browser_pages_opened}")
    print(f"role_candidates: {funnel.role_candidates}")
    print(f"experience_candidates: {funnel.experience_candidates}")
    print(f"location_candidates: {funnel.location_candidates}")
    print(f"employment_candidates: {funnel.employment_candidates}")
    print(f"fresh_candidates: {funnel.fresh_candidates}")
    print(f"url_verified: {funnel.url_verified}")
    print(f"deduped_count: {funnel.deduped_count}")
    print(f"intelligence_evaluated: {intelligence}")
    print(f"critic_evaluated: {critic}")
    print(f"unsupported_claims: {unsupported}")
    print("rejections:")
    for item in funnel.state.rejected:
        print(f"  {item.reason.value} | {item.title} | {item.detail or ''}")
    print(f"final_count: {funnel.final_count}")
    print(f"llm_calls: {provider.stats.calls}")
    print(f"llm_failures: {provider.stats.failures}")
    print(f"llm_provider_ready: {provider.available}")
    print(f"isolated_xlsx: {isolated}")
    print(f"production_workbook_unchanged: {workbook_unchanged}")
    print(f"email: {funnel.state.summary.email_status}")
    if found.blocked:
        print(f"blocked_note: {found.blocked}")
    if not workbook_unchanged:
        print("FAIL: production workbook changed")
        return 1
    return 0 if found.search == "PASS" or found.discovered_count else 2


async def run_workday_live_e2e(config: AppConfig) -> int:
    """Two real keyword passes, then the bounded qualification chain."""
    config = config.model_copy(update={"dry_run": True, "send_email": False, "fixture_mode": False})
    company, board = WORKDAY_TENANTS[0]
    production = config.project_root / config.settings.output.current_workbook
    before = production.stat().st_mtime_ns if production.exists() else None
    keyword = next(query for query in target_queries(config.roles) if "new grad" in query.lower())

    print("WORKDAY PHASE 4 LIVE E2E")
    print(f"company: {company}")
    print(f"board: {board}")
    print(f"keyword: {keyword}")
    print(f"browser_enabled setting: {config.settings.discovery.sources.workday.browser_enabled}")
    print("dry-run: workbook will not be written")

    try:
        first, second = await run_recorded_keyword_passes(
            config, board_url=board, company_name=company, keyword=keyword
        )
    except Exception as exc:
        print(f"BLOCKED: {exc}")
        return 2
    _print_pass("RUN 1", first)
    _print_pass("RUN 2", second)

    try:
        found = await discover_target_jobs(config, board_url=board, company_name=company)
    except Exception as exc:
        print(f"BLOCKED: {exc}")
        return 2

    http = HttpClient(config)
    provider = build_llm_provider(config)
    try:
        funnel = await qualify_workday_postings(config, found.detailed, llm=provider, http=http)
    finally:
        close = getattr(provider, "aclose", None)
        if callable(close):
            await close()
        await http.aclose()

    memory = NavigationMemory(config.project_root / config.settings.greenhouse_pilot.navigation_dir)
    stored = memory.load(company, "workday")
    if stored is not None:
        for version in stored.versions:
            print(
                f"stored_version: {version.version} status={version.status} "
                f"actions={[step.action for step in version.actions]} "
                f"success={version.success_count} failure={version.failure_count} "
                f"created_at={version.created_at or '-'}"
            )

    profile = merge_resume_profile(
        load_candidate_profile(config.project_root / "config" / "candidate_profile.yaml"),
        config.project_root / config.settings.candidate.profile_path,
    )
    canonical = load_profile(config.project_root / config.settings.candidate.profile_path)
    print(f"profile_version: {getattr(canonical, 'profile_version', None)}")
    print(f"profile_sha256: {canonical.resume.sha256 if canonical is not None else 'MISSING'}")
    _print_stages(found.detailed, funnel)
    intelligence = 0
    critic_count = 0
    unsupported = 0
    if funnel.final_count and canonical is not None and profile.profile_version is not None:
        by_id = {posting.job_id: posting for posting in found.detailed if posting.job_id}
        for job in funnel.state.jobs:
            posting = by_id.get(job.job_id or "")
            if posting is None:
                continue
            fit = evaluate_job(posting, profile, config.roles)
            critique = review_fit(fit, posting, profile)
            intelligence += 1
            critic_count += 1
            unsupported += len(critique.unsupported_claims)
            print(
                f"fit {job.job_id}: decision={critique.fit.decision} "
                f"role={critique.fit.role_alignment} experience={critique.fit.experience_alignment} "
                f"technical={critique.fit.technical_alignment} education={critique.fit.education_alignment} "
                f"location={critique.fit.location_alignment} freshness={critique.fit.freshness} "
                f"sponsorship={critique.fit.sponsorship} confidence={critique.fit.confidence} "
                f"matched={len(critique.fit.matched_requirements)} "
                f"missing={len(critique.fit.missing_requirements)} "
                f"issues={len(critique.issues)} unsupported={len(critique.unsupported_claims)}"
            )
            print(f"  sponsorship_evidence: {job.sponsorship_display}")
    else:
        print("LIVE FRESH CANDIDATE: NOT AVAILABLE")
        print("LIVE URL VERIFICATION: BLOCKED — no job survived freshness")
        print("LIVE H1B: BLOCKED — no fresh candidate")
        print("LIVE JOB INTELLIGENCE: BLOCKED — no job survived 24-hour freshness")
        print("LIVE CRITIC: BLOCKED — no job survived 24-hour freshness")

    isolated = _isolated_workbook(funnel.state.jobs)
    after = production.stat().st_mtime_ns if production.exists() else None
    unchanged = before == after
    print(f"discovered_count: {found.discovered_count}")
    print(f"strategy_version: {found.strategy_version}")
    print(f"strategy_reused: {found.strategy_reused}")
    print(f"navigation_model_calls: {first.navigation_model_calls + second.navigation_model_calls}")
    print(f"detail_pages: {len(found.detailed)}")
    print(f"fresh_candidates: {funnel.fresh_candidates}")
    print(f"url_verified: {funnel.url_verified}")
    print(f"intelligence_evaluated: {intelligence}")
    print(f"critic_evaluated: {critic_count}")
    print(f"unsupported_claims: {unsupported}")
    print(f"semantic_llm_calls: {provider.stats.calls}")
    print(f"semantic_llm_failures: {provider.stats.failures}")
    print(f"llm_provider_ready: {provider.available}")
    print(f"final_count: {funnel.final_count}")
    print(f"isolated_xlsx: {isolated}")
    print(f"production_workbook_unchanged: {unchanged}")
    print(f"email: {funnel.state.summary.email_status}")
    if not unchanged:
        print("FAIL: production workbook changed")
        return 1
    if not (first.applied and second.applied and second.reused and second.navigation_model_calls == 0):
        print("FAIL: keyword strategy was not replayed")
        return 1
    return 0


def _print_pass(label: str, result) -> None:
    print(
        f"{label}: applied={result.applied} reused={result.reused} promoted={result.promoted} "
        f"version={result.strategy_version} recovery={result.recovery} "
        f"cards={result.cards} navigation_model_calls={result.navigation_model_calls} "
        f"blocked={result.blocked or '-'}"
    )


_STAGE_ORDER = (
    ("role", RejectionReason.ROLE),
    ("experience", RejectionReason.SENIORITY),
    ("location", RejectionReason.LOCATION),
    ("employment", RejectionReason.EMPLOYMENT_TYPE),
    ("freshness", RejectionReason.FRESHNESS),
    ("url", RejectionReason.INVALID_URL),
)


def _print_stages(detailed, funnel) -> None:
    print("job_stages:")
    for posting in detailed:
        item = next(
            (
                rejected
                for rejected in funnel.state.rejected
                if posting.job_id and posting.job_id in (rejected.url or "")
            ),
            None,
        )
        parts = [f"{posting.job_id}: discovery PASS", "detail PASS"]
        if item is None:
            parts.extend(f"{name} PASS" for name, _reason in _STAGE_ORDER)
            print(" | ".join(parts))
            continue
        for name, reason in _STAGE_ORDER:
            if item.reason is reason:
                parts.append(f"{name} REJECT")
                if item.detail:
                    parts.append(item.detail)
                break
            parts.append(f"{name} PASS")
        else:
            parts.append(f"{item.reason.value} REJECT")
        print(" | ".join(parts))


def _isolated_workbook(jobs) -> str:
    with tempfile.TemporaryDirectory(prefix="workday-phase3-") as directory:
        root = Path(directory)
        current, _archive = write_workbooks(
            jobs,
            current_path=root / "jobs.xlsx",
            archive_dir=root / "archive",
            existing_rows=[],
            write_archive=True,
        )
        existed = current.exists()
    return "written-and-removed" if existed else "failed"

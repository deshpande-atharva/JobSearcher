"""Write the XLSX tracker and send the email summary."""

from __future__ import annotations

import time

from src.models.state import PipelineState
from src.services.freshness import is_fresh
from src.services.notifications import send_run_summary
from src.services.xlsx import write_workbooks
from src.utils.dates import utcnow
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_output(state: PipelineState) -> None:
    from src.services.discovery_learning import persist_learning

    persist_learning(state)
    config = state.config
    current = config.path(config.settings.output.current_workbook)
    archive_dir = config.path(config.settings.output.archive_dir)

    if config.dry_run:
        state.summary.workbook_path = str(current) + " (dry-run, not written)"
        state.summary.xlsx_status = "skipped"
        state.summary.archive_status = "skipped"
        state.summary.note("dry-run: workbook was not written")
        state.summary.stage_seconds["xlsx"] = 0.0
    else:
        started = time.perf_counter()
        current_path, archive_path = write_workbooks(
            state.jobs,
            current_path=current,
            archive_dir=archive_dir,
            existing_rows=state.resources.get("existing_rows") or [],
            write_archive=config.settings.output.write_archive,
        )
        state.summary.stage_seconds["xlsx"] = round(time.perf_counter() - started, 3)
        state.summary.workbook_path = str(current_path)
        state.summary.archive_path = str(archive_path) if archive_path else None
        state.summary.xlsx_status = "written"
        state.summary.archive_status = "written" if archive_path else "skipped"

    state.summary.fresh_authoritative_jobs = _authoritative_fresh_count(state)
    email_started = time.perf_counter()
    status = send_run_summary(config, state.summary, state.jobs)
    state.summary.email_status = status
    if "SMTP_NOT_CONFIGURED" in status:
        state.summary.email_skipped = True
        state.summary.email_reason = "SMTP_NOT_CONFIGURED"
    elif status.startswith("failed"):
        state.summary.email_failed = True
        state.summary.email_reason = status
    else:
        state.summary.email_reason = status
    state.summary.stage_seconds["email"] = round(time.perf_counter() - email_started, 3)
    state.summary.run_finished_at = utcnow()
    log.info("output complete", workbook=state.summary.workbook_path, email=state.summary.email_status)


def _authoritative_fresh_count(state: PipelineState) -> int:
    hours = state.config.freshness_hours
    use_updated = state.config.settings.run.freshness_use_updated_when_posted_missing
    count = 0
    for posting in state.raw_postings:
        fresh, _age = is_fresh(posting, hours, use_updated_when_posted_missing=use_updated)
        if fresh:
            count += 1
    return count

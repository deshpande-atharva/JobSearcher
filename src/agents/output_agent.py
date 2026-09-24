"""Write the XLSX tracker and send the email summary."""

from __future__ import annotations

from src.models.state import PipelineState
from src.services.notifications import send_run_summary
from src.services.xlsx import write_workbooks
from src.utils.dates import utcnow
from src.utils.logging import get_logger

log = get_logger(__name__)


async def run_output(state: PipelineState) -> None:
    config = state.config
    current = config.path(config.settings.output.current_workbook)
    archive_dir = config.path(config.settings.output.archive_dir)

    if config.dry_run:
        state.summary.workbook_path = str(current) + " (dry-run, not written)"
        state.summary.note("dry-run: workbook was not written")
    else:
        current_path, archive_path = write_workbooks(
            state.jobs,
            current_path=current,
            archive_dir=archive_dir,
            existing_rows=state.resources.get("existing_rows") or [],
            write_archive=config.settings.output.write_archive,
        )
        state.summary.workbook_path = str(current_path)
        state.summary.archive_path = str(archive_path) if archive_path else None

    state.summary.email_status = send_run_summary(config, state.summary, state.jobs)
    state.summary.run_finished_at = utcnow()
    log.info("output complete", workbook=state.summary.workbook_path, email=state.summary.email_status)

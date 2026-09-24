"""Read existing workbooks so reruns preserve tracking and skip known jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from src.models.config import AppConfig
from src.models.job import AppliedFlag, ApplicationStatus
from src.utils.dates import parse_datetime, utcnow
from src.utils.logging import get_logger

__all__ = ["HistoryIndex", "load_history"]

log = get_logger(__name__)


@dataclass
class HistoryIndex:
    known_keys: set[str] = field(default_factory=set)
    tracking: dict[str, dict[str, str]] = field(default_factory=dict)
    existing_rows: list[dict[str, str]] = field(default_factory=list)


def load_history(config: AppConfig) -> HistoryIndex:
    """Load the current tracker plus recent daily archives."""
    from src.services.xlsx import iter_workbook_rows, row_dedup_key

    index = HistoryIndex()
    current = config.path(config.settings.output.current_workbook)
    if current.is_file():
        _ingest(index, current, preserve_rows=True)

    archive_dir = config.path(config.settings.output.archive_dir)
    if archive_dir.is_dir():
        cutoff = utcnow() - timedelta(days=config.settings.output.history_lookback_days)
        for path in sorted(archive_dir.glob("*.xlsx")):
            if path.name.startswith("~$"):
                continue
            stamp = _archive_stamp(path)
            if stamp is not None and stamp < cutoff:
                continue
            _ingest(index, path, preserve_rows=False)

    log.info(
        "history loaded",
        known=len(index.known_keys),
        tracking=len(index.tracking),
        existing_rows=len(index.existing_rows),
    )
    return index


def _ingest(index: HistoryIndex, path: Path, *, preserve_rows: bool) -> None:
    from src.services.xlsx import iter_workbook_rows, row_dedup_key

    try:
        rows = list(iter_workbook_rows(path))
    except Exception as exc:
        log.warning("could not read workbook", path=str(path), error=str(exc))
        return

    for row in rows:
        key = row_dedup_key(row)
        if not key:
            continue
        index.known_keys.add(key)
        applied = row.get("Applied") or ""
        status = row.get("Status") or ""
        if applied or status:
            index.tracking.setdefault(key, {})
            if applied:
                index.tracking[key]["applied"] = _coerce_applied(applied)
            if status:
                index.tracking[key]["status"] = _coerce_status(status)
        if preserve_rows:
            index.existing_rows.append(row)


def _archive_stamp(path: Path):
    return parse_datetime(path.stem)


def _coerce_applied(value: str) -> str:
    text = value.strip()
    for flag in AppliedFlag:
        if text == flag.value or text.lower() == flag.value.lower():
            return flag.value
    lowered = text.lower()
    if lowered in {"applied", "yes", "true", "1", "☑ applied"}:
        return AppliedFlag.APPLIED.value
    return AppliedFlag.NOT_APPLIED.value


def _coerce_status(value: str) -> str:
    text = value.strip()
    for status in ApplicationStatus:
        if text.lower() == status.value.lower():
            return status.value
    return text or ApplicationStatus.NOT_STARTED.value

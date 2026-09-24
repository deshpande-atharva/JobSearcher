"""XLSX tracker generation and reading.

The workbook is the user's application tracker. Automation may add rows and
refresh informational columns; it must never overwrite ``Applied`` or
``Status`` once the user has set them.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from src.models.job import AppliedFlag, ApplicationStatus, Job
from src.utils.dates import format_date, format_datetime, utcnow
from src.utils.logging import get_logger
from src.utils.normalization import normalize_company_name, normalize_location_key, normalize_title
from src.utils.urls import canonicalize_url

__all__ = [
    "BASE_COLUMNS",
    "OPTIONAL_COLUMNS",
    "iter_workbook_rows",
    "row_dedup_key",
    "sanitize_cell",
    "write_workbooks",
]

log = get_logger(__name__)

BASE_COLUMNS: tuple[str, ...] = (
    "Company",
    "Job Title",
    "Normalized Role",
    "Location",
    "Remote Type",
    "Employment Type",
    "Posted Date",
    "Updated Date",
    "Job ID",
    "Source",
    "Direct Application URL",
    "Applied",
    "Status",
    "Found At",
    "Visa Sponsorship",
    "Sponsorship Evidence",
)

OPTIONAL_COLUMNS: tuple[str, ...] = ("H-1B Last Verified",)

ALL_COLUMNS: tuple[str, ...] = BASE_COLUMNS + OPTIONAL_COLUMNS

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

_HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_WRAP = Alignment(wrap_text=True, vertical="top")
_THIN = Border(
    left=Side(style="thin", color="D0D7DE"),
    right=Side(style="thin", color="D0D7DE"),
    top=Side(style="thin", color="D0D7DE"),
    bottom=Side(style="thin", color="D0D7DE"),
)
_ALT_FILL = PatternFill("solid", fgColor="F6F8FA")

_WIDTHS: dict[str, float] = {
    "Company": 22,
    "Job Title": 36,
    "Normalized Role": 24,
    "Location": 22,
    "Remote Type": 12,
    "Employment Type": 16,
    "Posted Date": 14,
    "Updated Date": 14,
    "Job ID": 18,
    "Source": 16,
    "Direct Application URL": 42,
    "Applied": 16,
    "Status": 18,
    "Found At": 20,
    "Visa Sponsorship": 16,
    "Sponsorship Evidence": 48,
    "H-1B Last Verified": 18,
}

_APPLIED_VALUES = [flag.value for flag in AppliedFlag]
_STATUS_VALUES = [status.value for status in ApplicationStatus]


def sanitize_cell(value: object) -> str:
    """Neutralise spreadsheet formula injection.

    Values beginning with ``=``, ``+``, ``-`` or ``@`` are stored as text by
    prefixing a single quote. Excel still displays the original characters.
    """
    if value is None:
        return ""
    text = str(value)
    if text and text[0] in _FORMULA_PREFIXES:
        return f"'{text}"
    return text


def job_to_row(job: Job) -> dict[str, str]:
    return {
        "Company": job.company,
        "Job Title": job.job_title,
        "Normalized Role": job.normalized_role,
        "Location": job.location,
        "Remote Type": job.remote_type.value,
        "Employment Type": job.employment_type.value,
        "Posted Date": format_date(job.posted_at),
        "Updated Date": format_date(job.updated_at),
        "Job ID": job.job_id or "",
        "Source": job.source,
        "Direct Application URL": job.direct_application_url,
        "Applied": job.applied.value,
        "Status": job.status.value,
        "Found At": format_datetime(job.found_at),
        "Visa Sponsorship": job.sponsorship_display,
        "Sponsorship Evidence": job.visa_sponsorship_evidence or "",
        "H-1B Last Verified": format_date(job.h1b_last_verified),
    }


def apply_tracking(job: Job, tracking: dict[str, dict[str, str]]) -> None:
    """Copy user-owned Applied/Status values onto a job. Never invent them."""
    saved = tracking.get(job.dedup_key)
    if not saved:
        return
    if saved.get("applied"):
        try:
            job.applied = AppliedFlag(saved["applied"])
        except ValueError:
            pass
    if saved.get("status"):
        try:
            job.status = ApplicationStatus(saved["status"])
        except ValueError:
            pass


def row_dedup_key(row: dict[str, str]) -> str:
    """Reproduce :meth:`Job.dedup_key` from a stored workbook row."""
    company = normalize_company_name(row.get("Company"))
    job_id = (row.get("Job ID") or "").strip()
    if company and job_id:
        return f"{company}|id:{job_id.lower()}"
    parts = [
        company,
        normalize_title(row.get("Job Title")),
        normalize_location_key(row.get("Location")),
        canonicalize_url(row.get("Direct Application URL")),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{company}|fp:{digest}"


def iter_workbook_rows(path: Path) -> list[dict[str, str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        rows = list(sheet.iter_rows(values_only=True))
    finally:
        workbook.close()
    if not rows:
        return []
    headers = [str(cell or "").strip() for cell in rows[0]]
    results: list[dict[str, str]] = []
    for raw in rows[1:]:
        if raw is None or all(cell is None or str(cell).strip() == "" for cell in raw):
            continue
        item = {
            headers[i]: ("" if cell is None else str(cell))
            for i, cell in enumerate(raw)
            if i < len(headers) and headers[i]
        }
        if item.get("Company") or item.get("Job Title"):
            results.append(item)
    return results


def write_workbooks(
    jobs: list[Job],
    *,
    current_path: Path,
    archive_dir: Path,
    existing_rows: list[dict[str, str]] | None = None,
    write_archive: bool = True,
) -> tuple[Path, Path | None]:
    """Write today's snapshot to ``data/current/jobs.xlsx`` and a daily archive.

    The current workbook contains only ``jobs`` (this run's accepted postings),
    in that order. Jobs that disappeared are left in older archives and are not
    copied forward. ``Applied`` and ``Status`` are copied from ``existing_rows``
    by company + job id (or the fingerprint fallback), never by row number.

    Both files are written to temporary paths first. The previous current
    workbook is replaced only after both workbooks have been built, so a crash
    or a failed archive write leaves the last good ``jobs.xlsx`` in place.
    """
    previous: dict[str, dict[str, str]] = {}
    for row in existing_rows or []:
        key = row_dedup_key(row)
        if key and key not in previous:
            previous[key] = row

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for job in jobs:
        key = job.dedup_key
        if key in seen:
            continue
        seen.add(key)
        incoming = job_to_row(job)
        prior = previous.get(key)
        if prior:
            if prior.get("Applied"):
                incoming["Applied"] = prior["Applied"]
            if prior.get("Status"):
                incoming["Status"] = prior["Status"]
        rows.append(incoming)

    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_tmp = current_path.with_name(current_path.name + ".tmp")
    archive_path: Path | None = None
    archive_tmp: Path | None = None
    if write_archive:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{utcnow().strftime('%Y-%m-%d')}.xlsx"
        archive_tmp = archive_path.with_name(archive_path.name + ".tmp")

    try:
        _write_sheet(current_tmp, rows)
        if archive_tmp is not None:
            _write_sheet(archive_tmp, rows)
        os.replace(current_tmp, current_path)
        if archive_tmp is not None and archive_path is not None:
            os.replace(archive_tmp, archive_path)
    finally:
        for tmp in (current_tmp, archive_tmp):
            if tmp is not None and tmp.exists():
                tmp.unlink(missing_ok=True)

    log.info("wrote current workbook", path=str(current_path), rows=len(rows))
    if archive_path is not None:
        log.info("wrote archive workbook", path=str(archive_path), rows=len(rows))
    return current_path, archive_path


def _write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    workbook = Workbook()
    sheet: Worksheet = workbook.active
    sheet.title = "Jobs"

    for col, header in enumerate(ALL_COLUMNS, start=1):
        cell = sheet.cell(1, col, header)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _THIN
        sheet.column_dimensions[get_column_letter(col)].width = _WIDTHS.get(header, 16)

    url_col = ALL_COLUMNS.index("Direct Application URL") + 1

    for r_idx, row in enumerate(rows, start=2):
        for c_idx, header in enumerate(ALL_COLUMNS, start=1):
            raw = row.get(header, "")
            value = sanitize_cell(raw)
            cell = sheet.cell(r_idx, c_idx, value)
            cell.alignment = _WRAP
            cell.border = _THIN
            if r_idx % 2 == 0:
                cell.fill = _ALT_FILL
            if header == "Direct Application URL" and raw and not raw.startswith("'"):
                if raw.startswith(("http://", "https://")):
                    cell.hyperlink = raw
                    cell.style = "Hyperlink"
                    cell.alignment = _WRAP

        # Re-apply hyperlink after style assignment (openpyxl quirk).
        url = row.get("Direct Application URL") or ""
        if url.startswith(("http://", "https://")):
            url_cell = sheet.cell(r_idx, url_col)
            url_cell.hyperlink = url
            url_cell.font = Font(color="0563C1", underline="single")

    last_row = max(len(rows) + 1, 2)
    last_col = get_column_letter(len(ALL_COLUMNS))
    sheet.auto_filter.ref = f"A1:{last_col}{last_row}"
    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 22
    sheet.auto_filter.ref = f"A1:{last_col}{last_row}"

    applied_col = get_column_letter(ALL_COLUMNS.index("Applied") + 1)
    status_col = get_column_letter(ALL_COLUMNS.index("Status") + 1)
    applied_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(_APPLIED_VALUES) + '"',
        allow_blank=False,
        showDropDown=False,
    )
    applied_dv.error = "Select a value from the list"
    applied_dv.errorTitle = "Applied"
    applied_dv.add(f"{applied_col}2:{applied_col}{max(last_row, 500)}")
    status_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(_STATUS_VALUES) + '"',
        allow_blank=False,
        showDropDown=False,
    )
    status_dv.add(f"{status_col}2:{status_col}{max(last_row, 500)}")
    sheet.add_data_validation(applied_dv)
    sheet.add_data_validation(status_dv)

    workbook.save(path)
    workbook.close()

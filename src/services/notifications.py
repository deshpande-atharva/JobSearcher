"""SMTP run-summary notifications.

Missing email configuration is a warning, never a pipeline failure. The message
is a concise summary -- never a dump of every job.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from typing import Any

from src.models.config import AppConfig, Secrets
from src.models.state import RunSummary
from src.utils.logging import get_logger, redact

__all__ = ["build_email_body", "email_configured", "send_run_summary"]

log = get_logger(__name__)


def email_configured(secrets: Secrets) -> bool:
    return secrets.smtp_configured


def build_email_body(summary: RunSummary, jobs: list[Any] | None = None) -> str:
    """Concise notification. Never the full workbook or the diagnostic report."""
    from src.utils.dates import utcnow

    when = summary.run_finished_at or utcnow()
    date = when.strftime("%Y-%m-%d")
    final = summary.jobs_accepted
    lines = [f"Daily Job Discovery — {date}", f"Final jobs: {final}"]
    if final == 0:
        lines.extend(
            [
                "The pipeline completed successfully.",
                "No jobs matched the production 24-hour criteria.",
                "This is a valid zero-result run.",
            ]
        )
        return "\n".join(lines)

    fresh = summary.remaining_after("role", "seniority", "location", "employment", "freshness")
    lines.extend(
        [
            f"Companies scanned: {summary.companies_attempted}",
            f"Jobs discovered: {summary.jobs_discovered}",
            f"Deduplicated: {summary.jobs_after_cross_source_dedup}",
            f"Fresh <{summary.freshness_hours_used:g}h: {fresh}",
            "New jobs:",
        ]
    )
    new_jobs = [job for job in (jobs or []) if getattr(job, "is_new", True)]
    if not new_jobs:
        lines.append("• (none)")
    else:
        for job in new_jobs:
            lines.append(f"• {job.company} — {job.job_title}")
    lines.append("Workbook updated successfully.")
    return "\n".join(lines)


def send_run_summary(config: AppConfig, summary: RunSummary, jobs: list[Any] | None = None) -> str:
    """Send the run summary. Returns a short status string for the report."""
    if not config.settings.notifications.enabled:
        return "disabled"
    if config.dry_run and not config.send_email:
        return "skipped (dry-run)"
    if not config.send_email:
        return "skipped"

    secrets = config.secrets
    if not secrets.smtp_configured:
        missing = ", ".join(secrets.missing_smtp_fields())
        log.warning("WARNING: email notifications disabled", missing=missing)
        return f"skipped (missing {missing})"

    prefix = config.settings.notifications.subject_prefix
    accepted = summary.jobs_accepted
    subject = f"{prefix} {accepted} accepted · {summary.new_jobs} new · {summary.companies_failed} companies failed"

    body = build_email_body(summary, jobs)
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = secrets.notification_from or secrets.smtp_username or ""
    message["To"] = secrets.notification_email or ""
    message.set_content(body)

    port = secrets.smtp_port or 587
    host = secrets.smtp_host or ""
    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.ehlo()
            if port != 25:
                try:
                    smtp.starttls()
                    smtp.ehlo()
                except smtplib.SMTPException:
                    pass
            smtp.login(secrets.smtp_username or "", secrets.smtp_password or "")
            smtp.send_message(message)
    except Exception as exc:
        log.warning("failed to send notification email", error=redact(str(exc)))
        return f"failed ({type(exc).__name__})"

    log.info("notification email sent", to=secrets.notification_email)
    return "sent"

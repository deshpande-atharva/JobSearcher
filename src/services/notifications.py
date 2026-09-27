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

__all__ = ["build_email_body", "build_email_subject", "email_configured", "send_run_summary"]

log = get_logger(__name__)


def email_configured(secrets: Secrets) -> bool:
    return secrets.smtp_configured


def _source_health_lines(summary: RunSummary) -> list[str]:
    if not summary.source_health:
        return []
    from src.services.production_report import production_source_state

    lines = ["Source health:"]
    for name in sorted(summary.source_health):
        health = summary.source_health[name]
        lines.append(
            f"- {name}: {production_source_state(health)} "
            f"jobs={health.jobs_discovered} company_failures={health.failed}"
        )
    return lines


def build_email_subject(summary: RunSummary, prefix: str = "[job-agent]") -> str:
    from src.utils.dates import utcnow

    when = summary.run_finished_at or summary.run_started_at or utcnow()
    date = when.strftime("%Y-%m-%d")
    return f"{prefix} {date} status=completed final={summary.jobs_accepted}"


def build_email_body(summary: RunSummary, jobs: list[Any] | None = None) -> str:
    """Concise notification. Titles only. No descriptions, resume text, or secrets."""
    from src.utils.dates import utcnow

    when = summary.run_finished_at or utcnow()
    date = when.strftime("%Y-%m-%d")
    final = summary.jobs_accepted
    runtime = summary.duration_seconds
    runtime_text = "n/a" if runtime is None else f"{runtime:.1f}s"
    lines = [
        f"Daily Job Discovery — {date}",
        "Status: completed",
        f"Final jobs: {final}",
        f"Discovered jobs: {summary.jobs_discovered}",
        f"Fresh jobs: {summary.fresh_authoritative_jobs}",
        f"Final candidates: {final}",
        f"Runtime: {runtime_text}",
        f"LLM state: {summary.llm_state_after or summary.llm_circuit_state}",
    ]
    lines.extend(_source_health_lines(summary))
    if summary.failed_sources:
        lines.append("Failed sources: " + ", ".join(summary.failed_sources))
    if final == 0:
        lines.extend(
            [
                "The pipeline completed successfully.",
                "No jobs matched the production 24-hour criteria.",
                "This is a valid zero-result run.",
            ]
        )
        return "\n".join(lines)

    lines.append("New jobs:")
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
        return "skipped (SMTP_NOT_CONFIGURED)"

    subject = build_email_subject(summary, config.settings.notifications.subject_prefix)
    port = secrets.smtp_port or 587
    host = secrets.smtp_host or ""
    try:
        body = build_email_body(summary, jobs)
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = secrets.notification_from or secrets.smtp_username or ""
        message["To"] = secrets.notification_email or ""
        message.set_content(body)
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

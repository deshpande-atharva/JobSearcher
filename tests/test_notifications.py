import hashlib
import smtplib
from pathlib import Path

from src.models.config import Secrets
from src.models.state import RunSummary
from src.services.notifications import build_email_body, send_run_summary
from tests.conftest import make_job


def test_missing_smtp_warns_and_does_not_fail(tmp_config, caplog) -> None:
    tmp_config.send_email = True
    tmp_config.dry_run = False
    status = send_run_summary(tmp_config, RunSummary())
    assert status.startswith("skipped")
    assert "WARNING: email notifications disabled" in caplog.text


def test_smtp_failure_warns_and_does_not_fail(tmp_config, monkeypatch, caplog) -> None:
    tmp_config.send_email = True
    tmp_config.dry_run = False
    tmp_config.secrets = Secrets(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user",
        smtp_password="secret",
        notification_email="jobs@example.com",
    )

    class BrokenSMTP:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise OSError("connection refused")

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr("src.services.notifications.smtplib.SMTP", BrokenSMTP)
    status = send_run_summary(tmp_config, RunSummary(jobs_accepted=1), [make_job()])
    assert status.startswith("failed")
    assert "failed to send notification email" in caplog.text


def test_auth_failure_does_not_log_the_password(tmp_config, monkeypatch, caplog) -> None:
    password = "smtp-secret-value"
    tmp_config.send_email = True
    tmp_config.dry_run = False
    tmp_config.secrets = Secrets(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user",
        smtp_password=password,
        notification_email="jobs@example.com",
    )

    class AuthSMTP:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise smtplib.SMTPAuthenticationError(535, f"rejected {password}".encode())

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr("src.services.notifications.smtplib.SMTP", AuthSMTP)
    status = send_run_summary(tmp_config, RunSummary(jobs_accepted=0))
    assert status == "failed (SMTPAuthenticationError)"
    assert password not in caplog.text
    for record in caplog.records:
        error = getattr(record, "context", {}).get("error", "")
        assert password not in str(error)


def test_notification_does_not_change_the_workbook(tmp_config, monkeypatch) -> None:
    current = Path("data/current/jobs.xlsx")
    archive = Path("data/archive/2026-09-26.xlsx")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (current, archive)}
    tmp_config.send_email = True
    tmp_config.dry_run = False
    tmp_config.secrets = Secrets(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user",
        smtp_password="smtp-secret-value",
        notification_email="jobs@example.com",
    )

    class AuthSMTP:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise smtplib.SMTPAuthenticationError(535, b"authentication unsuccessful")

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr("src.services.notifications.smtplib.SMTP", AuthSMTP)
    assert send_run_summary(tmp_config, RunSummary()).startswith("failed")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert after == before


def test_email_body_lists_new_jobs_not_the_workbook() -> None:
    summary = RunSummary(
        jobs_accepted=1,
        companies_attempted=35,
        jobs_discovered=10,
        jobs_after_cross_source_dedup=9,
        freshness_hours_used=24,
    )
    job = make_job(company="Scale AI", job_title="Software Engineer, Public Sector - New Grad")
    body = build_email_body(summary, [job])
    assert "Final jobs: 1" in body
    assert "Scale AI — Software Engineer, Public Sector - New Grad" in body
    assert "Workbook updated successfully." in body
    assert "Direct Application URL" not in body


def test_zero_result_email_is_a_valid_run() -> None:
    body = build_email_body(RunSummary(jobs_accepted=0), [])
    assert "Final jobs: 0" in body
    assert "valid zero-result run" in body

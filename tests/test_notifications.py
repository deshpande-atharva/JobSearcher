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

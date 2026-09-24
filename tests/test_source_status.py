"""SOURCE_OK / EMPTY / ERROR / BLOCKED / UNSUPPORTED stay distinct."""

from src.models.job import RawJobPosting
from src.models.state import RunSummary
from src.sources.base import SourceResult, classify_source_status


def test_classify_source_status_matrix() -> None:
    assert classify_source_status(None) == "ERROR"
    assert classify_source_status("HTTP 400", http_status=400) == "ERROR"
    assert classify_source_status("TimeoutException: timed out") == "ERROR"
    assert classify_source_status("disallowed by robots.txt") == "BLOCKED"
    assert classify_source_status("access denied (HTTP 403)", http_status=403) == "BLOCKED"
    assert classify_source_status("CAPTCHA interstitial") == "BLOCKED"
    assert classify_source_status("WORKDAY_UNSUPPORTED_CONFIGURATION") == "UNSUPPORTED"
    assert classify_source_status("HTTP 404", http_status=404) == "ERROR"


def test_ok_empty_error_are_not_interchangeable() -> None:
    ok = SourceResult.ok(
        "workday",
        "Adobe",
        [RawJobPosting(source="workday", company_name="Adobe", title="Software Engineer")],
    )
    # Construct via discovered_count path: empty list is EMPTY, not ERROR.
    empty = SourceResult.ok("workday", "Adobe", [])
    error = SourceResult.fail("workday", "Adobe", "Workday CXS HTTP 400", http_status=400)
    blocked = SourceResult.fail("smartrecruiters", "Example", "disallowed by robots.txt")
    unsupported = SourceResult.fail(
        "workday", "Elevance Health", "WORKDAY_UNSUPPORTED_CONFIGURATION"
    )

    assert empty.success is True
    assert empty.status == "EMPTY"
    assert empty.discovered_count == 0

    assert error.success is False
    assert error.status == "ERROR"
    assert error.http_status == 400
    assert error.status != "EMPTY"

    assert blocked.status == "BLOCKED"
    assert unsupported.status == "UNSUPPORTED"
    assert ok.status == "OK"


def test_record_source_does_not_count_http_400_as_empty() -> None:
    summary = RunSummary()
    summary.record_source(
        "workday",
        success=False,
        jobs=0,
        error="Workday CXS HTTP 400",
        status="ERROR",
        http_status=400,
        company="Adobe",
    )
    summary.record_source("greenhouse", success=True, jobs=0, company="Quiet Corp")
    health = summary.source_health["workday"]
    assert health.empty == 0
    assert health.error == 1
    assert health.failed == 1
    assert health.http_400 == 1
    assert health.successful == 0
    greenhouse = summary.source_health["greenhouse"]
    assert greenhouse.empty == 1
    assert greenhouse.error == 0
    report = summary.discovery_health_report()
    assert "error: 1" in report
    assert "HTTP 400: 1" in report
    summary_text = summary.source_summary_report()
    assert "SOURCE SUMMARY" in summary_text
    assert "Workday" in summary_text

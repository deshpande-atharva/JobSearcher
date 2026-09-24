from src.models.config import load_config
from src.models.job import RawJobPosting
from src.services.url_verification import pick_direct_url
from src.utils.urls import classify_url, extract_job_id, is_generic_careers_page
from tests.conftest import ROOT


def _policy():
    return load_config(ROOT / "config", env={}, fixture_mode=True, send_email=False).url_policy


def test_ats_urls_accepted() -> None:
    policy = _policy()
    for url in (
        "https://boards.greenhouse.io/acme/jobs/4012345",
        "https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "https://jobs.ashbyhq.com/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "https://jobs.smartrecruiters.com/Acme/743999123456789-software-engineer",
        "https://example.wd1.myworkdayjobs.com/en-US/External/job/Software-Engineer_R-12345",
        "https://careers-acme.icims.com/jobs/12345/software-engineer/job",
    ):
        assert classify_url(url, policy).acceptable_as_final is True


def test_aggregator_urls_rejected_as_final() -> None:
    policy = _policy()
    for url in (
        "https://www.linkedin.com/jobs/view/123",
        "https://www.indeed.com/viewjob?jk=abc",
        "https://www.glassdoor.com/job-listing/software-engineer",
        "https://jobright.ai/jobs/abc",
        "https://www.ziprecruiter.com/jobs/abc",
        "https://www.dice.com/job-detail/abc",
        "https://www.google.com/search?q=jobs",
    ):
        verdict = classify_url(url, policy)
        assert verdict.acceptable_as_final is False


def test_job_id_preserved_from_url() -> None:
    assert extract_job_id("https://boards.greenhouse.io/acme/jobs/4012345") == "4012345"
    assert extract_job_id("https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee") == (
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )


def test_generic_careers_homepage() -> None:
    assert is_generic_careers_page("https://example.com/careers") is True
    assert is_generic_careers_page("https://boards.greenhouse.io/acme/jobs/4012345") is False


def test_pick_direct_url_rejects_aggregator_only() -> None:
    policy = _policy()
    raw = RawJobPosting(
        source="jobright",
        company_name="Initech",
        title="Backend Engineer",
        apply_url="https://www.linkedin.com/jobs/view/999999",
    )
    check = pick_direct_url(raw, policy)
    assert check.accepted is False
    assert "aggregator" in check.reason.lower()


def test_unwrap_prefers_ats_over_aggregator() -> None:
    policy = _policy()
    raw = RawJobPosting(
        source="jobright",
        company_name="Acme",
        title="Software Engineer",
        apply_url="https://jobright.ai/jobs/abc?url=https%3A%2F%2Fboards.greenhouse.io%2Facme%2Fjobs%2F4012345",
    )
    check = pick_direct_url(raw, policy)
    assert check.accepted is True
    assert "greenhouse.io" in check.url
    assert check.job_id == "4012345"

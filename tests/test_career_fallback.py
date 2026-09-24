"""Career-page fallback must not fabricate dates or drop direct URLs."""

from src.models.config import CompanyConfig
from src.models.job import DateSource
from src.sources.base import SourceContext
from src.sources.company_career import CompanyCareerSource
from src.utils.logging import get_logger


def _source(tmp_config) -> CompanyCareerSource:
    return CompanyCareerSource(
        SourceContext(config=tmp_config, http=None, logger=get_logger("test.career"))
    )


def test_career_json_ld_keeps_posted_date(tmp_config) -> None:
    company = CompanyConfig(name="Example", careers_url="https://example.com/careers")
    html = (
        '<script type="application/ld+json">'
        '{"@type":"JobPosting","title":"Software Engineer",'
        '"url":"https://example.com/jobs/99",'
        '"datePosted":"2026-09-22T12:00:00Z"}'
        "</script>"
    )
    postings = _source(tmp_config)._parse_html(html, "https://example.com/careers", company)
    assert len(postings) == 1
    assert postings[0].title == "Software Engineer"
    assert postings[0].apply_url == "https://example.com/jobs/99"
    assert postings[0].date_source is DateSource.POSTED_DATE
    assert postings[0].posted_at is not None


def test_career_link_without_date_is_unknown(tmp_config) -> None:
    company = CompanyConfig(name="Example", careers_url="https://example.com/careers")
    html = '<a href="https://example.com/jobs/backend-engineer-1">Backend Engineer</a>'
    postings = _source(tmp_config)._parse_html(html, "https://example.com/careers", company)
    assert len(postings) == 1
    assert postings[0].title == "Backend Engineer"
    assert postings[0].apply_url == "https://example.com/jobs/backend-engineer-1"
    assert postings[0].job_id is None
    assert postings[0].posted_at is None
    assert postings[0].date_source is DateSource.UNKNOWN


def test_career_missing_job_id_still_keeps_direct_url(tmp_config) -> None:
    company = CompanyConfig(name="Example", careers_url="https://example.com/careers")
    html = '<a href="https://example.com/job/platform-engineer">Platform Engineer</a>'
    postings = _source(tmp_config)._parse_html(html, "https://example.com/careers", company)
    assert postings[0].job_id is None
    assert postings[0].apply_url.endswith("/job/platform-engineer")

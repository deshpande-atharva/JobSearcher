"""Workday application URLs keep the career site from the public CXS path."""

from __future__ import annotations

import json

import pytest

from src.agents.extraction_agent import _to_job
from src.models.config import CompanyConfig
from src.models.job import RawJobPosting
from src.services.url_verification import pick_direct_url, verify_url
from src.sources.base import FetchResult, SourceContext
from src.sources.greenhouse import GreenhouseSource
from src.sources.workday import WorkdaySource, public_workday_job_url
from src.utils.logging import get_logger

BAH_HOST = "bah.wd1.myworkdayjobs.com"
BAH_SITE = "BAH_Jobs"
BAH_PATH = "/job/El-Segundo-CA/Cybersecurity-Engineer--Junior_R0250290"
BAH_ID = "R0250290"
BAH_BOARD = f"https://{BAH_HOST}/{BAH_SITE}"
BAH_URL = f"https://{BAH_HOST}/{BAH_SITE}{BAH_PATH}"
BAH_BROKEN = f"https://{BAH_HOST}{BAH_PATH}"

GLOBAL_HOST = "easterseals.wd5.myworkdayjobs.com"
GLOBAL_SITE = "ESNH_VTcareers"
GLOBAL_PATH = "/job/Manchester-NH/Software-Engineer_R100"
GLOBAL_BOARD = f"https://{GLOBAL_HOST}/{GLOBAL_SITE}"
GLOBAL_URL = f"https://{GLOBAL_HOST}/{GLOBAL_SITE}{GLOBAL_PATH}"

GREENHOUSE_URL = "https://boards.greenhouse.io/figma/jobs/6143238004"


def _company(name: str, board: str) -> CompanyConfig:
    return CompanyConfig(name=name, ats_type="workday", ats_identifier=board, careers_url=board)


def _ctx(config, http) -> SourceContext:
    return SourceContext(config=config, http=http, logger=get_logger("test.workday.urls"))


class _Cxs:
    def __init__(self, path: str, job_id: str, title: str = "Software Engineer") -> None:
        self.path = path
        self.job_id = job_id
        self.title = title
        self.bodies: list[dict] = []

    async def request(self, method: str, url: str, **kwargs):
        if method == "POST":
            self.bodies.append(dict(kwargs.get("json_body") or {}))
            payload = {
                "total": 1,
                "jobPostings": [
                    {
                        "title": self.title,
                        "externalPath": self.path,
                        "bulletFields": [self.job_id],
                        "locationsText": "El Segundo, CA",
                        "postedOn": "Posted Today",
                    }
                ],
            }
            return FetchResult(url=url, status=200, text=json.dumps(payload))
        return FetchResult(url=url, status=200, text="<html></html>")


def _live(config):
    return config.model_copy(update={"fixture_mode": False})


@pytest.mark.asyncio
async def test_configured_workday_url_keeps_site_path(tmp_config) -> None:
    http = _Cxs(BAH_PATH, BAH_ID, title="Cybersecurity Engineer, Junior")
    source = WorkdaySource(_ctx(_live(tmp_config), http))
    result = await source.discover_result(_company("Booz Allen Hamilton", BAH_BOARD))
    assert result.success is True
    posting = result.jobs[0]
    assert posting.job_id == BAH_ID
    assert posting.apply_url == BAH_URL
    assert posting.apply_url != BAH_BROKEN
    assert BAH_SITE in (posting.apply_url or "")
    job = _to_job(posting)
    assert job is not None
    assert job.direct_application_url == BAH_URL
    check = pick_direct_url(posting, tmp_config.url_policy)
    assert check.accepted is True
    assert check.url == BAH_URL
    assert check.url.startswith("https://")


@pytest.mark.asyncio
async def test_global_workday_url_uses_discovered_site(tmp_config) -> None:
    http = _Cxs(GLOBAL_PATH, "R100")
    source = WorkdaySource(_ctx(_live(tmp_config), http))
    result = await source.discover_result(_company("Easterseals", GLOBAL_BOARD))
    assert result.success is True
    posting = result.jobs[0]
    assert posting.apply_url == GLOBAL_URL
    assert GLOBAL_SITE in (posting.apply_url or "")
    assert posting.apply_url != f"https://{GLOBAL_HOST}{GLOBAL_PATH}"
    check = pick_direct_url(posting, tmp_config.url_policy)
    assert check.accepted is True
    assert check.url == GLOBAL_URL
    assert check.url.startswith("https://")


def test_missing_workday_path_is_not_manufactured() -> None:
    assert public_workday_job_url(BAH_HOST, BAH_SITE, None) is None
    assert public_workday_job_url(BAH_HOST, BAH_SITE, "") is None
    assert public_workday_job_url(BAH_HOST, BAH_SITE, "/search") is None
    assert public_workday_job_url(BAH_HOST, BAH_SITE, "/jobs") is None
    assert public_workday_job_url("", BAH_SITE, BAH_PATH) is None
    assert public_workday_job_url(BAH_HOST, "", BAH_PATH) is None
    manufactured = f"https://{BAH_HOST}/{BAH_SITE}/job/{BAH_ID}"
    assert public_workday_job_url(BAH_HOST, BAH_SITE, None) != manufactured


@pytest.mark.asyncio
async def test_workday_posting_without_path_has_no_application_url(tmp_config) -> None:
    class Bare(_Cxs):
        async def request(self, method: str, url: str, **kwargs):
            if method == "POST":
                payload = {
                    "total": 1,
                    "jobPostings": [
                        {
                            "title": "Cybersecurity Engineer, Junior",
                            "bulletFields": [BAH_ID],
                            "locationsText": "El Segundo, CA",
                        }
                    ],
                }
                return FetchResult(url=url, status=200, text=json.dumps(payload))
            return FetchResult(url=url, status=200, text="")

    source = WorkdaySource(_ctx(_live(tmp_config), Bare(BAH_PATH, BAH_ID)))
    result = await source.discover_result(_company("Booz Allen Hamilton", BAH_BOARD))
    assert result.jobs[0].apply_url is None
    assert result.jobs[0].job_id == BAH_ID
    check = pick_direct_url(result.jobs[0], tmp_config.url_policy)
    assert check.accepted is False
    assert "no application URL" in check.reason


def test_site_stripped_workday_url_is_rejected(tmp_config) -> None:
    raw = RawJobPosting(
        source="workday",
        company_name="Booz Allen Hamilton",
        title="Cybersecurity Engineer, Junior",
        job_id=BAH_ID,
        apply_url=BAH_BROKEN,
    )
    check = pick_direct_url(raw, tmp_config.url_policy)
    assert check.accepted is False
    assert "career site" in check.reason


def test_explicit_workday_url_is_kept() -> None:
    explicit = BAH_URL
    assert public_workday_job_url(BAH_HOST, BAH_SITE, explicit) == explicit


def test_greenhouse_absolute_url_is_unchanged(tmp_config) -> None:
    source = GreenhouseSource(_ctx(tmp_config, object()))
    company = CompanyConfig(
        name="Figma",
        ats_type="greenhouse",
        ats_identifier="figma",
        careers_url="https://boards.greenhouse.io/figma",
    )
    posting = source._to_posting(
        {
            "title": "Software Engineer Intern",
            "id": 6143238004,
            "absolute_url": GREENHOUSE_URL,
            "location": {"name": "San Francisco, CA"},
        },
        company,
        "figma",
    )
    assert posting is not None
    assert posting.apply_url == GREENHOUSE_URL
    check = pick_direct_url(posting, tmp_config.url_policy)
    assert check.accepted is True
    assert check.url == GREENHOUSE_URL
    assert check.url.startswith("https://")


@pytest.mark.asyncio
async def test_workday_page_not_found_document_is_rejected(tmp_config) -> None:
    live = _live(tmp_config)
    raw = RawJobPosting(
        source="workday",
        company_name="Booz Allen Hamilton",
        title="Cybersecurity Engineer, Junior",
        job_id=BAH_ID,
        apply_url=BAH_URL,
    )
    check = pick_direct_url(raw, live.url_policy)
    assert check.accepted is True

    class Missing:
        async def head(self, url: str) -> FetchResult:
            return FetchResult(
                url=url,
                status=200,
                text=(
                    '<wml:Application_Error xmlns:wml="http://www.workday.com/ns/model/1.0">'
                    "<wml:Message>Requested page not found</wml:Message>"
                ),
            )

    missed = await verify_url(check, live, Missing())
    assert missed.accepted is False
    assert missed.reachable is False

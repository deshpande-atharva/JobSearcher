"""Workday CXS adapter: page size, status classification, pagination, fallback."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.models.config import CompanyConfig
from src.models.job import DateSource
from src.models.state import PipelineState
from src.sources.base import FetchResult, SourceContext, SourceResult
from src.sources.workday import CXS_PAGE_SIZE, WorkdaySource, parse_workday_site
from src.utils.logging import get_logger

ADOBE_URL = "https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced"


def _company() -> CompanyConfig:
    return CompanyConfig(
        name="Adobe",
        ats_type="workday",
        ats_identifier=ADOBE_URL,
        careers_url=ADOBE_URL,
    )


def _ctx(config, http) -> SourceContext:
    return SourceContext(config=config, http=http, logger=get_logger("test.workday"))


def _live_config(tmp_config):
    return tmp_config.model_copy(update={"fixture_mode": False})


class FakeWorkdayHttp:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.bodies: list[dict] = []

    async def request(self, method: str, url: str, **kwargs):
        body = kwargs.get("json_body") or {}
        if method == "POST":
            self.bodies.append(dict(body))
            limit = body.get("limit", 20)
            offset = int(body.get("offset") or 0)
            if self.mode == "timeout":
                return FetchResult(url=url, status=None, error="TimeoutException: timed out")
            if self.mode == "malformed":
                return FetchResult(url=url, status=200, text="not-json")
            if self.mode in {"http400", "html_fallback"} or (isinstance(limit, int) and limit > CXS_PAGE_SIZE):
                return FetchResult(
                    url=url,
                    status=400,
                    text='{"errorCode":"HTTP_400","message":""}',
                    error="HTTP 400",
                )
            if self.mode == "source_cap":
                if offset >= 2000:
                    return FetchResult(url=url, status=200, text='{"total":0,"jobPostings":[]}')
                jobs = [
                    {
                        "title": f"Software Engineer {offset + i}",
                        "externalPath": f"/job/swe_{offset + i}",
                        "id": f"R{offset + i}",
                        "postedOn": "Posted 30+ Days Ago",
                    }
                    for i in range(20)
                ]
                total = 2000 if offset == 0 else 0
                return FetchResult(
                    url=url,
                    status=200,
                    text=json.dumps({"total": total, "jobPostings": jobs}),
                )
            if self.mode == "paginate":
                if offset >= 60:
                    return FetchResult(url=url, status=200, text='{"total":0,"jobPostings":[]}')
                jobs = [
                    {
                        "title": f"Software Engineer {offset + i}",
                        "externalPath": f"/job/swe_{offset + i}",
                        "id": f"R{offset + i}",
                        "postedOn": "Posted Today",
                    }
                    for i in range(20)
                ]
                total = 60 if offset == 0 else 0
                return FetchResult(
                    url=url,
                    status=200,
                    text=json.dumps({"total": total, "jobPostings": jobs}),
                )
            jobs = [
                {
                    "title": "Software Engineer",
                    "externalPath": "/job/swe_1",
                    "id": "R1",
                    "postedOn": "Posted Today",
                }
            ]
            return FetchResult(url=url, status=200, text=json.dumps({"total": 1, "jobPostings": jobs}))
        return FetchResult(url=url, status=200, text="<html></html>")

    async def get_text(self, url: str, **kwargs):
        if self.mode == "html_fallback":
            return FetchResult(
                url=url,
                status=200,
                text=(
                    '<script type="application/ld+json">'
                    '{"@type":"JobPosting","title":"Backend Engineer",'
                    '"url":"https://adobe.wd5.myworkdayjobs.com/job/1",'
                    '"datePosted":"2026-09-22T12:00:00Z"}'
                    "</script>"
                ),
            )
        return FetchResult(url=url, status=200, text="<html><body>no jobs</body></html>")


@pytest.mark.asyncio
async def test_workday_successful_request(tmp_config) -> None:
    http = FakeWorkdayHttp("ok")
    source = WorkdaySource(_ctx(_live_config(tmp_config), http))
    result = await source.discover_result(_company())
    assert result.success is True
    assert result.status == "OK"
    assert result.http_status == 200
    assert result.discovered_count == 1
    assert result.jobs[0].title == "Software Engineer"
    assert result.jobs[0].date_source is DateSource.POSTED_DATE
    assert http.bodies[0]["limit"] == CXS_PAGE_SIZE


@pytest.mark.asyncio
async def test_workday_http_400_is_error_not_empty(tmp_config) -> None:
    source = WorkdaySource(_ctx(_live_config(tmp_config), FakeWorkdayHttp("http400")))
    result = await source.discover_result(_company())
    assert result.success is False
    assert result.status == "ERROR"
    assert result.http_status == 400
    assert result.discovered_count == 0
    assert result.fallback_used is True
    assert result.fallback_status == "EMPTY"
    assert result.status != "EMPTY"


@pytest.mark.asyncio
async def test_workday_timeout(tmp_config) -> None:
    source = WorkdaySource(_ctx(_live_config(tmp_config), FakeWorkdayHttp("timeout")))
    result = await source.discover_result(_company())
    assert result.success is False
    assert result.status == "ERROR"
    assert "TimeoutException" in (result.error or "")


@pytest.mark.asyncio
async def test_workday_malformed_response(tmp_config) -> None:
    source = WorkdaySource(_ctx(_live_config(tmp_config), FakeWorkdayHttp("malformed")))
    result = await source.discover_result(_company())
    assert result.success is False
    assert result.status == "ERROR"
    assert "JSON" in (result.error or "")


@pytest.mark.asyncio
async def test_workday_pagination_ignores_zero_total_on_later_pages(tmp_config) -> None:
    http = FakeWorkdayHttp("paginate")
    source = WorkdaySource(_ctx(_live_config(tmp_config), http))
    result = await source.discover_result(_company())
    assert result.success is True
    assert result.discovered_count == 60
    assert result.diagnostics.get("pages") == 3
    assert result.diagnostics.get("source_list_cap_reached") is False
    assert all(body["limit"] == 20 for body in http.bodies)


@pytest.mark.asyncio
async def test_workday_source_list_cap_is_reported_not_treated_as_exact(tmp_config) -> None:
    source = WorkdaySource(_ctx(_live_config(tmp_config), FakeWorkdayHttp("source_cap")))
    result = await source.discover_result(_company())
    assert result.success is True
    assert result.discovered_count == 2000
    assert result.diagnostics.get("source_list_cap_reached") is True
    assert result.diagnostics.get("reported_total") == 2000
    assert all(body["limit"] == 20 for body in source.http.bodies)


@pytest.mark.asyncio
async def test_workday_html_fallback_after_cxs_failure(tmp_config) -> None:
    source = WorkdaySource(_ctx(_live_config(tmp_config), FakeWorkdayHttp("html_fallback")))
    result = await source.discover_result(_company())
    assert result.success is True
    assert result.fallback_used is True
    assert result.fallback_status == "OK"
    assert result.http_status == 400
    assert result.jobs[0].title == "Backend Engineer"
    assert result.jobs[0].date_source is DateSource.POSTED_DATE


@pytest.mark.asyncio
async def test_workday_unsupported_identifier(tmp_config) -> None:
    source = WorkdaySource(_ctx(_live_config(tmp_config), FakeWorkdayHttp("ok")))
    company = CompanyConfig(name="Nope", ats_type="workday", ats_identifier="adobe")
    result = await source.discover_result(company)
    assert result.success is False
    assert result.status == "UNSUPPORTED"


@pytest.mark.asyncio
async def test_workday_bounded_concurrency(tmp_config) -> None:
    from src.agents.discovery_agent import _discover_workday

    class StubSource:
        in_flight = 0
        max_in_flight = 0

        def __init__(self, ctx) -> None:
            pass

        async def discover_result(self, company):
            StubSource.in_flight += 1
            StubSource.max_in_flight = max(StubSource.max_in_flight, StubSource.in_flight)
            await asyncio.sleep(0.04)
            StubSource.in_flight -= 1
            return SourceResult.ok("workday", company.name, [])

    StubSource.in_flight = 0
    StubSource.max_in_flight = 0
    state = PipelineState(config=_live_config(tmp_config))
    state.resources["workday_semaphore"] = asyncio.Semaphore(2)
    ctx = _ctx(state.config, FakeWorkdayHttp("ok"))
    company = _company()
    await asyncio.gather(
        *[_discover_workday(state, ctx, StubSource, company) for _ in range(6)]
    )
    assert StubSource.max_in_flight <= 2
    assert StubSource.max_in_flight >= 1


def test_parse_workday_site_rejects_name_only() -> None:
    assert parse_workday_site("adobe") is None
    parsed = parse_workday_site(ADOBE_URL)
    assert parsed is not None
    host, tenant, site = parsed
    assert tenant == "adobe"
    assert site == "external_experienced"
    assert "wd5" in host

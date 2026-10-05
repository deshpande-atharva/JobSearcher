"""CDX observability tests.

Verifies that each HTTP failure category is distinguished from a successful
HTTP 200 with zero records, and that discovery behavior is unchanged when CDX
is unavailable.

Previous test count: 503.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from src.services.global_boards import load_public_board_index
from src.services.global_workday import load_workday_archive_sample
from src.sources.base import FetchResult

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Shared mock HTTP client
# ---------------------------------------------------------------------------


class _CdxMockHTTP:
    """Returns a fixed CDX response regardless of which CDX URL is requested."""

    def __init__(self, mode: str, records: list | None = None):
        self.mode = mode
        self._records = records or []
        self.calls: list[tuple[str, str]] = []

    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url))
        mode = self.mode
        if mode == "timeout":
            raise asyncio.TimeoutError()
        if mode == "connection_error":
            raise OSError("Connection refused")
        if mode == "http_429":
            return FetchResult(url=url, status=429, text="", error="HTTP 429")
        if mode == "http_500":
            return FetchResult(url=url, status=500, text="", error="HTTP 500")
        if mode == "zero_records":
            # HTTP 200 with CDX header row only (zero matching records)
            body = json.dumps([["original", "statuscode"]])
            return FetchResult(url=url, status=200, text=body)
        if mode == "with_records":
            body = json.dumps([["original", "statuscode"]] + self._records)
            return FetchResult(url=url, status=200, text=body)
        if mode == "malformed":
            # HTTP 200 with non-JSON body (e.g. CDX proxy returns an HTML error page)
            return FetchResult(url=url, status=200, text="<html>Service Error</html>")
        raise ValueError(f"Unknown mock mode: {mode!r}")


class _FakeCtx:
    """Minimal context duck-type: _cdx_body only uses ctx.http."""

    def __init__(self, http):
        self.http = http


def _ctx(mode: str, records: list | None = None) -> _FakeCtx:
    return _FakeCtx(_CdxMockHTTP(mode, records))


# ---------------------------------------------------------------------------
# Workday CDX path — load_workday_archive_sample
# ---------------------------------------------------------------------------

_WORKDAY_RECORD = ["https://testfirm.wd5.myworkdayjobs.com/en-US/External", "200"]


@pytest.mark.asyncio
async def test_workday_cdx_http200_zero_records_returns_empty_boards():
    """HTTP 200 with a CDX header row and no data rows → empty board list."""
    urls, detail, _ = await load_workday_archive_sample(
        _ctx("zero_records"),
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []
    assert "GLOBAL DISCOVERY" in detail


@pytest.mark.asyncio
async def test_workday_cdx_http200_with_records_returns_boards():
    """HTTP 200 with valid Workday board URL rows → board list is non-empty."""
    urls, *_ = await load_workday_archive_sample(
        _ctx("with_records", records=[_WORKDAY_RECORD]),
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert len(urls) >= 1
    assert any("testfirm.wd5.myworkdayjobs.com" in u for u in urls)


@pytest.mark.asyncio
async def test_workday_cdx_http429_returns_empty_boards():
    """HTTP 429 (rate limited) → empty board list, no crash, no fake empty success."""
    urls, *_ = await load_workday_archive_sample(
        _ctx("http_429"),
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []


@pytest.mark.asyncio
async def test_workday_cdx_http500_returns_empty_boards():
    """HTTP 500 (server error) → empty board list, no crash."""
    urls, *_ = await load_workday_archive_sample(
        _ctx("http_500"),
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []


@pytest.mark.asyncio
async def test_workday_cdx_timeout_returns_empty_boards():
    """asyncio.TimeoutError from CDX request → empty board list, no crash."""
    urls, *_ = await load_workday_archive_sample(
        _ctx("timeout"),
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []


@pytest.mark.asyncio
async def test_workday_cdx_connection_error_returns_empty_boards():
    """OSError (connection refused / DNS failure) → empty board list, no crash."""
    urls, *_ = await load_workday_archive_sample(
        _ctx("connection_error"),
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []


@pytest.mark.asyncio
async def test_workday_cdx_malformed_response_returns_empty_boards():
    """HTTP 200 with non-JSON body (e.g. HTML error page) → empty board list, no crash."""
    urls, *_ = await load_workday_archive_sample(
        _ctx("malformed"),
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []


# ---------------------------------------------------------------------------
# Greenhouse/Ashby CDX path — load_public_board_index
# ---------------------------------------------------------------------------

_GH_RECORD = ["https://boards.greenhouse.io/testco/jobs/1234", "200"]


@pytest.mark.asyncio
async def test_board_index_cdx_http200_zero_records_returns_empty_index():
    """HTTP 200 zero-records CDX response → empty greenhouse/ashby lists."""
    index = await load_public_board_index(
        _ctx("zero_records"),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert index.greenhouse == []
    assert index.ashby == []


@pytest.mark.asyncio
async def test_board_index_cdx_http200_with_records_returns_tokens():
    """HTTP 200 with a valid Greenhouse URL row → greenhouse list is non-empty."""
    index = await load_public_board_index(
        _ctx("with_records", records=[_GH_RECORD]),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert "testco" in index.greenhouse


@pytest.mark.asyncio
async def test_board_index_cdx_http429_marks_index_unavailable():
    """HTTP 429 → index.status 'unavailable', no tokens, no crash."""
    index = await load_public_board_index(
        _ctx("http_429"),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert index.greenhouse == []
    assert index.ashby == []
    assert index.status == "unavailable"


@pytest.mark.asyncio
async def test_board_index_cdx_http500_marks_index_unavailable():
    """HTTP 500 → index.status 'unavailable', no tokens."""
    index = await load_public_board_index(
        _ctx("http_500"),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert index.status == "unavailable"
    assert index.greenhouse == []


@pytest.mark.asyncio
async def test_board_index_cdx_timeout_marks_index_unavailable():
    """Timeout → index.status 'unavailable', no tokens."""
    index = await load_public_board_index(
        _ctx("timeout"),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert index.status == "unavailable"


@pytest.mark.asyncio
async def test_board_index_cdx_connection_error_marks_index_unavailable():
    """Connection error → index.status 'unavailable'."""
    index = await load_public_board_index(
        _ctx("connection_error"),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert index.status == "unavailable"


@pytest.mark.asyncio
async def test_board_index_cdx_malformed_returns_empty_tokens_without_crash():
    """Malformed HTTP 200 body (e.g. HTML error page) → empty token lists, no crash.

    The CDX server responded with HTTP 200 so index.status is 'ok' (the server
    was reachable), but the non-JSON body produces zero parseable tokens.
    """
    index = await load_public_board_index(
        _ctx("malformed"),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    # CDX responded (HTTP 200) so we treat it as "up" even though body was garbage
    assert index.status == "ok"
    assert index.greenhouse == []
    assert index.ashby == []


# ---------------------------------------------------------------------------
# Behavioral invariant: all CDX error modes produce empty board targets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_cdx_error_modes_produce_no_workday_board_targets():
    """Every CDX failure mode must return an empty URL list.

    Empty URL list → board_urls empty → board-search slots skipped with
    no_global_board_target → learning memory unchanged.
    (The skip and learning behavior is tested in test_board_search_v2_new_tests.py.)
    """
    error_modes = ("http_429", "http_500", "timeout", "connection_error", "malformed")
    for mode in error_modes:
        urls, *_ = await load_workday_archive_sample(
            _ctx(mode),
            clusters=("wd5",),
            limit=40,
            now=NOW,
            prefixes_per_run=1,
        )
        assert urls == [], f"CDX mode {mode!r} must produce no board targets; got {urls}"


# ---------------------------------------------------------------------------
# myworkdaysite.com domain regression — confirmed root cause of wasted queries
# ---------------------------------------------------------------------------


def test_myworkdaysite_student_portal_urls_rejected_by_parser():
    """myworkdaysite.com CDX returns student-portal hosts, not wdN career boards.

    Internet Archive CDX for myworkdaysite.com returns URLs like:
      https://wd1-student.myworkdaysite.com/institution/SiteName/...
    These have host structure 'wd1-student.myworkdaysite.com' (only two labels
    before the TLD: 'wd1-student' and 'myworkdaysite.com').  The _WD_HOST regex
    requires three labels: tenant.wdN.myworkdaysite.com.  None of these match.
    """
    from src.services.public_board_index import parse_public_workday_board

    # Actual CDX rows observed for wd1+myworkdaysite.com prefix='k'
    non_career_urls = [
        "https://wd1-student.myworkdaysite.com/adams/ASU_Student_Site",
        "https://wd1-student.myworkdaysite.com/adams/ASU_Student_Site/assets/banner",
        "https://wd1-student.myworkdaysite.com/adams/ASU_Student_Site/login",
        "https://wd1-student.myworkdaysite.com/adams/ASU_Student_Site/passwordreset/abc123",
        "https://wd1-student.myworkdaysite.com/aims/Aims_External_Student_Site",
        "https://wd1-student.myworkdaysite.com/aims/Aims_External_Student_Site/assets/logo",
        "https://wd1-student.myworkdaysite.com/aims/Aims_External_Student_Site/events/def456",
    ]
    for url in non_career_urls:
        result = parse_public_workday_board(url)
        assert result is None, (
            f"myworkdaysite.com student portal must be rejected; got {result!r} for {url}"
        )


@pytest.mark.asyncio
async def test_load_workday_archive_sample_queries_only_myworkdayjobs():
    """CDX queries must target myworkdayjobs.com only, never myworkdaysite.com.

    myworkdaysite.com CDX returns student portals that produce 0 career boards and
    waste CDX quota (4 of 12 queries per run historically).
    """
    ctx = _ctx("zero_records")
    await load_workday_archive_sample(
        ctx,
        clusters=("wd5", "wd1"),
        limit=40,
        now=NOW,
        prefixes_per_run=2,
    )
    for _method, url in ctx.http.calls:
        assert "myworkdaysite.com" not in url, (
            f"Must not query myworkdaysite.com (produces 0 boards); found: {url}"
        )
        assert "myworkdayjobs.com" in url, (
            f"Should query myworkdayjobs.com; found: {url}"
        )


@pytest.mark.asyncio
async def test_workday_cdx_query_count_is_clusters_times_prefixes():
    """Query count = len(clusters) × prefixes_per_run (no domain doubling).

    After removing myworkdaysite.com, the query count is exactly
    2 clusters × 2 prefixes = 4 (previously was 3 × 2 = 6 due to first-cluster
    getting both WORKDAY_DOMAINS).
    """
    ctx = _ctx("zero_records")
    await load_workday_archive_sample(
        ctx,
        clusters=("wd5", "wd1"),
        limit=40,
        now=NOW,
        prefixes_per_run=2,
        snapshot_fallback_enabled=False,
    )
    expected = 2 * 2  # 2 clusters × 2 prefixes × 1 domain
    assert len(ctx.http.calls) == expected, (
        f"Expected {expected} CDX queries; got {len(ctx.http.calls)}"
    )

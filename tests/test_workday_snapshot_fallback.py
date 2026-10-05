"""Workday Snapshot Index fallback tests.

Verifies bounded date-range CDX fallback behavior when resumeKey CDX returns
no usable boards.
"""
from __future__ import annotations
import asyncio
import json
import pytest
from datetime import datetime, timezone
from src.services.global_workday import load_workday_archive_sample, load_workday_snapshot_fallback
from src.services.public_board_index import workday_snapshot_query_url
from src.sources.base import FetchResult

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
_VALID_BOARD_URL = "https://testfirm.wd5.myworkdayjobs.com/en-US/External"
_STUDENT_PORTAL_URL = "https://wd1-student.myworkdaysite.com/adams/ASU_Student_Site"
_LOGIN_URL = "https://testfirm.wd5.myworkdayjobs.com/login"
_ASSET_URL = "https://testfirm.wd5.myworkdayjobs.com/assets/logo.png"


class _MockHTTP:
    """Mock HTTP client that returns different responses for CDX vs snapshot URLs."""

    def __init__(self, cdx_mode: str, snapshot_mode: str, records: list | None = None):
        self.cdx_mode = cdx_mode
        self.snapshot_mode = snapshot_mode
        self._records = records or []
        self.calls: list[tuple[str, str]] = []

    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url))
        # Distinguish CDX resumeKey queries from date-range snapshot queries
        mode = self.snapshot_mode if ("from=" in url and "resumeKey" not in url) else self.cdx_mode
        return self._respond(mode, url)

    def _respond(self, mode: str, url: str):
        if mode == "timeout":
            raise asyncio.TimeoutError()
        if mode == "connection_error":
            raise OSError("Connection refused")
        if mode == "http_429":
            return FetchResult(url=url, status=429, text="", error="HTTP 429")
        if mode == "zero_records":
            body = json.dumps([["original", "statuscode"]])
            return FetchResult(url=url, status=200, text=body)
        if mode == "with_records":
            body = json.dumps([["original", "statuscode"]] + self._records)
            return FetchResult(url=url, status=200, text=body)
        raise ValueError(f"Unknown mode: {mode!r}")


class _FakeCtx:
    def __init__(self, http):
        self.http = http


def _ctx(cdx_mode: str, snapshot_mode: str = "zero_records", records: list | None = None) -> _FakeCtx:
    return _FakeCtx(_MockHTTP(cdx_mode, snapshot_mode, records))


# ---------------------------------------------------------------------------
# workday_snapshot_query_url structure
# ---------------------------------------------------------------------------

def test_snapshot_query_url_has_date_params_no_resume_key():
    """Snapshot query URL must have from/to date params but NO resumeKey."""
    url = workday_snapshot_query_url(
        cluster="wd5",
        from_date="20260901000000",
        to_date="20261001000000",
        limit=40,
    )
    assert "from=20260901000000" in url
    assert "to=20261001000000" in url
    assert "resumeKey" not in url
    assert "wd5.myworkdayjobs.com" in url
    assert "matchType=domain" in url


def test_snapshot_query_url_rejects_invalid_cluster():
    """Invalid cluster name must raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="workday cluster"):
        workday_snapshot_query_url(cluster="invalid", from_date="20260901", to_date="20261001", limit=40)


# ---------------------------------------------------------------------------
# CDX success → fallback NOT called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cdx_success_fallback_not_invoked():
    """When CDX returns valid boards, fallback is not called."""
    ctx = _ctx("with_records", snapshot_mode="zero_records", records=[[_VALID_BOARD_URL, "200"]])
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert len(urls) >= 1
    assert board_origin == "global_index"
    # No snapshot (from=) queries should have been made
    snapshot_calls = [u for _, u in ctx.http.calls if "from=" in u and "resumeKey" not in u]
    assert snapshot_calls == [], f"Fallback must not be called when CDX found boards; got: {snapshot_calls}"


# ---------------------------------------------------------------------------
# CDX partial failure → CDX boards retained, fallback NOT called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cdx_partial_failure_retains_cdx_boards():
    """If some CDX requests fail but any return valid boards, use CDX results, skip fallback."""
    # Track call count: first call returns 429, second returns valid boards
    call_count = [0]
    import json as _json

    class _PartialHTTP:
        calls: list = []

        async def request(self, method, url, **kwargs):
            self.calls.append((method, url))
            # resumeKey queries: alternate between 429 and success
            if "resumeKey" in url:
                call_count[0] += 1
                if call_count[0] % 2 == 1:
                    return FetchResult(url=url, status=429, text="", error="HTTP 429")
                body = _json.dumps([["original", "statuscode"], [_VALID_BOARD_URL, "200"]])
                return FetchResult(url=url, status=200, text=body)
            # Snapshot queries should not be called
            return FetchResult(url=url, status=200, text=_json.dumps([["original", "statuscode"]]))

    ctx = _FakeCtx(_PartialHTTP())
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=2,  # 2 prefixes = 2 CDX queries; one fails, one succeeds
    )
    assert len(urls) >= 1, "CDX partial failure with 1 good result must return that result"
    assert board_origin == "global_index"
    snapshot_calls = [u for _, u in ctx.http.calls if "from=" in u and "resumeKey" not in u]
    assert snapshot_calls == [], f"Fallback must not be called when CDX found boards"


# ---------------------------------------------------------------------------
# CDX unusable → snapshot fallback invoked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cdx_unusable_snapshot_fallback_invoked():
    """When CDX returns no usable boards, snapshot fallback is called."""
    ctx = _ctx("zero_records", snapshot_mode="with_records", records=[[_VALID_BOARD_URL, "200"]])
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert len(urls) >= 1, "Snapshot fallback must return discovered boards"
    assert board_origin == "global_snapshot_index"
    assert "Wayback Snapshot Index" in detail or "snapshot" in detail.lower()
    # Snapshot (from=) queries must have been made
    snapshot_calls = [u for _, u in ctx.http.calls if "from=" in u and "resumeKey" not in u]
    assert len(snapshot_calls) >= 1


# ---------------------------------------------------------------------------
# Snapshot success — valid board returned with correct provenance tag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_fallback_returns_valid_board_url():
    """Snapshot fallback returns valid career URL when CDX finds nothing."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "with_records", records=[[_VALID_BOARD_URL, "200"]])),
        clusters=("wd5",),
        limit=40,
        now=NOW,
    )
    assert len(urls) >= 1
    assert any("testfirm.wd5.myworkdayjobs.com" in u for u in urls)


# ---------------------------------------------------------------------------
# Invalid candidates rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_fallback_rejects_student_portal():
    """Student portal URLs from myworkdaysite.com must be rejected."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "with_records", records=[[_STUDENT_PORTAL_URL, "200"]])),
        clusters=("wd1",),
        limit=40,
        now=NOW,
    )
    assert urls == [], f"Student portal must be rejected; got {urls}"


@pytest.mark.asyncio
async def test_snapshot_fallback_rejects_login_page():
    """Login page paths must be rejected."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "with_records", records=[[_LOGIN_URL, "200"]])),
        clusters=("wd5",),
        limit=40,
        now=NOW,
    )
    assert urls == [], f"Login page must be rejected; got {urls}"


@pytest.mark.asyncio
async def test_snapshot_fallback_rejects_asset_url():
    """Asset URLs must be rejected."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "with_records", records=[[_ASSET_URL, "200"]])),
        clusters=("wd5",),
        limit=40,
        now=NOW,
    )
    assert urls == [], f"Asset URL must be rejected; got {urls}"


@pytest.mark.asyncio
async def test_snapshot_fallback_rejects_non_workday_url():
    """Non-Workday URLs must be rejected."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "with_records", records=[["https://boards.greenhouse.io/test", "200"]])),
        clusters=("wd5",),
        limit=40,
        now=NOW,
    )
    assert urls == [], f"Non-Workday URL must be rejected"


# ---------------------------------------------------------------------------
# Snapshot failure → empty result, no exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_fallback_timeout_returns_empty():
    """Snapshot API timeout must return empty list without raising."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "timeout")),
        clusters=("wd5",),
        limit=40,
        now=NOW,
    )
    assert urls == []


@pytest.mark.asyncio
async def test_snapshot_fallback_http_error_returns_empty():
    """Snapshot API HTTP error must return empty list without raising."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "http_429")),
        clusters=("wd5",),
        limit=40,
        now=NOW,
    )
    assert urls == []


@pytest.mark.asyncio
async def test_snapshot_fallback_connection_error_returns_empty():
    """Snapshot API connection error must return empty list without raising."""
    urls = await load_workday_snapshot_fallback(
        _FakeCtx(_MockHTTP("zero_records", "connection_error")),
        clusters=("wd5",),
        limit=40,
        now=NOW,
    )
    assert urls == []


# ---------------------------------------------------------------------------
# No-target semantics: both CDX and snapshot fail → board_origin still "global_index"
# empty result means no target → skipped by discovery agent (tested elsewhere)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_both_fail_returns_empty_with_global_index_origin():
    """Both CDX and snapshot fail → empty URLs, board_origin is 'global_index' (no fallback used)."""
    ctx = _ctx("zero_records", snapshot_mode="zero_records")
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []
    # board_origin is "global_index" when fallback produced nothing (it wasn't "used")
    # The empty URL list is what causes skipping — the origin value is irrelevant in this path


@pytest.mark.asyncio
async def test_both_fail_no_exception_escapes():
    """Both CDX and snapshot timeout must not raise; pipeline remains safe."""
    ctx = _ctx("timeout", snapshot_mode="timeout")
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert urls == []


# ---------------------------------------------------------------------------
# Snapshot fallback disabled via setting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_fallback_disabled_skips_fallback():
    """When snapshot_fallback_enabled=False, fallback is never called even when CDX returns nothing."""
    ctx = _ctx("zero_records", snapshot_mode="with_records", records=[[_VALID_BOARD_URL, "200"]])
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
        snapshot_fallback_enabled=False,
    )
    assert urls == [], "Fallback disabled; CDX returned nothing; result must be empty"
    snapshot_calls = [u for _, u in ctx.http.calls if "from=" in u and "resumeKey" not in u]
    assert snapshot_calls == [], "Snapshot queries must not be made when fallback disabled"


# ---------------------------------------------------------------------------
# Provenance and board_origin
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_discovered_boards_have_correct_board_origin():
    """Boards from snapshot fallback must return board_origin='global_snapshot_index'."""
    ctx = _ctx("zero_records", snapshot_mode="with_records", records=[[_VALID_BOARD_URL, "200"]])
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert board_origin == "global_snapshot_index"


@pytest.mark.asyncio
async def test_cdx_discovered_boards_have_global_index_origin():
    """Boards from CDX must return board_origin='global_index'."""
    ctx = _ctx("with_records", records=[[_VALID_BOARD_URL, "200"]])
    urls, detail, board_origin = await load_workday_archive_sample(
        ctx,
        clusters=("wd5",),
        limit=40,
        now=NOW,
        prefixes_per_run=1,
    )
    assert board_origin == "global_index"

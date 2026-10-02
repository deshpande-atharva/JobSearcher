"""Comprehensive tests for V2 Hybrid Global Job Discovery board search.

All tests are deterministic - no network calls, no real browser.
"""

from __future__ import annotations

import pytest

from src.browser.page_state import JobCard, PageElement, PageState
from src.browser.session import InMemoryBrowser, PageFixture
from src.models.job import RawJobPosting
from src.services.board_search import (
    API_CAPABILITIES,
    BoardSearchSlot,
    observe_capabilities,
    official_job_url,
    run_profile_search,
    search_key,
)
from tests.conftest import make_job


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _learning_settings(**overrides):
    from src.models.config import LearningSettings

    data = {
        "exploration_share": 0.20,
        "revisit_share": 0.10,
        "cooldown_runs": 3,
        "revisit_runs": 2,
        "min_runs_before_suppression": 2,
        "novelty_weight": 1.0,
        "exploration_bonus": 0.35,
        "revisit_bonus": 0.20,
    }
    data.update(overrides)
    return LearningSettings(**data)


def _board_job(job_id, strategy_id, method, board, *, is_new, exact_url=True):
    """Job with board-search provenance for learning tests."""
    return make_job(
        company="GlobalCorp",
        job_title=f"Engineer {job_id}",
        job_id=job_id,
        source=board,
        is_new=is_new,
        provenance={
            "search_strategy_id": strategy_id,
            "method": method,
            "board": board,
            "query": strategy_id.replace("_", " "),
            "exact_url": exact_url,
            "discovered_from": [board],
        },
        direct_application_url=f"https://example.test/{board}/{job_id}",
    )


# ---------------------------------------------------------------------------
# API_CAPABILITIES tests
# ---------------------------------------------------------------------------


def test_api_capabilities_documented():
    """API_CAPABILITIES has workday/greenhouse/ashby/lever; workday keyword_search=supported."""
    assert "workday" in API_CAPABILITIES
    assert "greenhouse" in API_CAPABILITIES
    assert "ashby" in API_CAPABILITIES
    assert "lever" in API_CAPABILITIES
    assert API_CAPABILITIES["workday"]["keyword_search"] == "supported"


# ---------------------------------------------------------------------------
# observe_capabilities tests
# ---------------------------------------------------------------------------


def test_observe_capabilities_with_search_input():
    """PageState with input text containing 'search' → keyword_search=supported."""
    page = PageState(
        inputs=[PageElement(id="kw", text="Search for jobs", role="searchbox")]
    )
    caps = observe_capabilities(page)
    assert caps["keyword_search"] == "supported"


def test_observe_capabilities_without_search_input():
    """Empty PageState → keyword_search=unknown."""
    page = PageState()
    caps = observe_capabilities(page)
    assert caps["keyword_search"] == "unknown"


def test_observe_capabilities_with_location_select():
    """PageState with select labeled 'location' → location_filter=supported."""
    page = PageState(
        inputs=[PageElement(id="kw", text="search", role="searchbox")],
        selects=[PageElement(id="loc", text="Location", options=["Remote", "US"])],
    )
    caps = observe_capabilities(page)
    assert caps["location_filter"] == "supported"


def test_observe_capabilities_unknown_when_no_selects():
    """PageState with inputs but no selects → location_filter=unknown."""
    page = PageState(
        inputs=[PageElement(id="kw", text="search", role="searchbox")],
    )
    caps = observe_capabilities(page)
    assert caps["location_filter"] == "unknown"


# ---------------------------------------------------------------------------
# requested vs applied filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requested_vs_applied_filters_distinguished():
    """run_profile_search with no location select → requested includes 'location', applied does NOT."""
    search_page = PageState(
        url="https://boards.greenhouse.io/acmecorp",
        inputs=[PageElement(id="kw", text="Search for jobs", role="searchbox")],
        job_cards=[
            JobCard(id="j1", title="Backend Engineer", url="https://boards.greenhouse.io/acmecorp/jobs/123")
        ],
        links=[PageElement(id="link1", text="Backend Engineer", href="https://boards.greenhouse.io/acmecorp/jobs/123")],
    )
    result_page = PageState(
        url="https://boards.greenhouse.io/acmecorp/jobs/123",
        job_cards=[
            JobCard(id="j1", title="Backend Engineer", url="https://boards.greenhouse.io/acmecorp/jobs/123")
        ],
        links=[],
    )
    pages = {
        "search": PageFixture(id="search", state=search_page, transitions={"kw": "result"}),
        "result": PageFixture(id="result", state=result_page, transitions={}),
    }
    browser = InMemoryBrowser(pages, "search")
    execution = await run_profile_search(
        browser,
        board="greenhouse",
        strategy_id="backend_engineer",
        query="backend engineer",
        company_name="AcmeCorp",
    )
    assert "location" in execution.requested_filters
    assert "location" not in execution.applied_filters


# ---------------------------------------------------------------------------
# official_job_url tests
# ---------------------------------------------------------------------------


def test_official_job_url_accepts_posting():
    """/jobs/123 path → True."""
    assert official_job_url("https://boards.greenhouse.io/acmecorp/jobs/123") is True


def test_official_job_url_rejects_search_page():
    """/jobs path → False."""
    assert official_job_url("https://boards.greenhouse.io/acmecorp/jobs") is False


def test_official_job_url_rejects_aggregator():
    """linkedin.com → False."""
    assert official_job_url("https://www.linkedin.com/jobs/123") is False


def test_official_job_url_rejects_empty():
    """None, '' → False."""
    assert official_job_url(None) is False
    assert official_job_url("") is False


# ---------------------------------------------------------------------------
# Playwright fixture test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_playwright_search_with_fixture():
    """InMemoryBrowser fixture with search→results→job_detail transition."""
    job_url = "https://boards.greenhouse.io/acmecorp/jobs/123"

    search_page = PageState(
        url="https://boards.greenhouse.io/acmecorp",
        inputs=[PageElement(id="kw", text="Search for jobs", role="searchbox")],
        buttons=[],
        links=[],
        job_cards=[],
    )
    results_page = PageState(
        url="https://boards.greenhouse.io/acmecorp/search",
        job_cards=[JobCard(id="j1", title="Backend Engineer", company="AcmeCorp", url=job_url)],
        links=[PageElement(id="link1", href=job_url, text="Backend Engineer")],
    )
    job_detail_page = PageState(
        url=job_url,
        job_cards=[JobCard(id="j1", title="Backend Engineer", company="AcmeCorp", url=job_url)],
        links=[],
    )

    pages = {
        "search": PageFixture(id="search", state=search_page, transitions={"kw": "results"}),
        "results": PageFixture(id="results", state=results_page, transitions={"link1": "job_detail"}),
        "job_detail": PageFixture(id="job_detail", state=job_detail_page, transitions={}),
    }
    browser = InMemoryBrowser(pages, "search")

    execution = await run_profile_search(
        browser,
        board="greenhouse",
        strategy_id="backend_engineer",
        query="backend engineer",
        company_name="AcmeCorp",
    )

    assert execution.search_success is True
    assert len(execution.postings) > 0
    posting = execution.postings[0]
    assert posting.apply_url == job_url


# ---------------------------------------------------------------------------
# provenance stamping tests
# ---------------------------------------------------------------------------


def test_unknown_company_not_filtered():
    """RawJobPosting with company absent from companies.yaml has board_origin=global_index and is not filtered."""
    from src.agents.discovery_agent import _stamp_board_search_provenance
    from src.models.config import BoardSearchSettings

    settings = BoardSearchSettings()
    posting = RawJobPosting(
        source="greenhouse",
        company_name="UnknownGlobalCorp",
        title="Engineer",
        provenance={"board_origin": "global_index", "board_token": "unknownglobalcorp"},
    )
    postings = [posting]
    _stamp_board_search_provenance(postings, settings)
    # Should not crash and should not stamp (greenhouse doesn't get stamped by this function - only workday partitions)
    assert posting.provenance.get("search_strategy_id") is None


def test_board_search_api_stamping():
    """Calling _stamp_board_search_provenance on a Workday partition posting stamps search_strategy_id, method=api, board=workday."""
    from src.agents.discovery_agent import _stamp_board_search_provenance
    from src.models.config import BoardSearchSettings

    settings = BoardSearchSettings()
    posting = RawJobPosting(
        source="workday",
        company_name="TechCorp",
        title="Backend Engineer",
        job_id="wd-123",
        provenance={
            "workday_partitions": ["backend engineer"],
        },
    )
    _stamp_board_search_provenance([posting], settings)
    assert posting.provenance.get("search_strategy_id") == "backend_engineer"
    assert posting.provenance.get("method") == "api"
    assert posting.provenance.get("board") == "workday"


def test_board_search_api_stamping_skips_unfiltered():
    """Unfiltered partition postings are NOT stamped."""
    from src.agents.discovery_agent import _stamp_board_search_provenance
    from src.models.config import BoardSearchSettings

    settings = BoardSearchSettings()
    posting = RawJobPosting(
        source="workday",
        company_name="TechCorp",
        title="Engineer",
        provenance={
            "workday_partitions": ["unfiltered", "backend engineer"],
        },
    )
    _stamp_board_search_provenance([posting], settings)
    assert posting.provenance.get("search_strategy_id") is None


def test_board_search_api_stamping_skips_already_stamped():
    """Postings with existing search_strategy_id are not overwritten."""
    from src.agents.discovery_agent import _stamp_board_search_provenance
    from src.models.config import BoardSearchSettings

    settings = BoardSearchSettings()
    posting = RawJobPosting(
        source="workday",
        company_name="TechCorp",
        title="Backend Engineer",
        provenance={
            "workday_partitions": ["backend engineer"],
            "search_strategy_id": "existing_strategy",
        },
    )
    _stamp_board_search_provenance([posting], settings)
    assert posting.provenance["search_strategy_id"] == "existing_strategy"


# ---------------------------------------------------------------------------
# _extract_board_urls tests
# ---------------------------------------------------------------------------


def test_extract_board_urls_workday():
    """Workday apply_url → host + first path segment."""
    from src.agents.discovery_agent import _extract_board_urls

    posting = RawJobPosting(
        source="workday",
        company_name="TechCorp",
        title="Engineer",
        apply_url="https://techcorp.wd1.myworkdayjobs.com/External/job/Remote/Backend-Engineer_JR-001",
    )
    urls = _extract_board_urls([posting])
    assert "workday" in urls
    assert len(urls["workday"]) == 1
    assert urls["workday"][0] == "https://techcorp.wd1.myworkdayjobs.com/External"


def test_extract_board_urls_greenhouse():
    """Greenhouse board_token → boards.greenhouse.io URL."""
    from src.agents.discovery_agent import _extract_board_urls

    posting = RawJobPosting(
        source="greenhouse",
        company_name="AcmeCorp",
        title="Engineer",
        provenance={"board_token": "acmecorp", "board_origin": "global_index"},
    )
    urls = _extract_board_urls([posting])
    assert "greenhouse" in urls
    assert "https://boards.greenhouse.io/acmecorp" in urls["greenhouse"]


def test_extract_board_urls_ashby():
    """Ashby board_token → jobs.ashbyhq.com URL."""
    from src.agents.discovery_agent import _extract_board_urls

    posting = RawJobPosting(
        source="ashby",
        company_name="StartupCo",
        title="Engineer",
        provenance={"board_token": "startupco"},
    )
    urls = _extract_board_urls([posting])
    assert "ashby" in urls
    assert "https://jobs.ashbyhq.com/startupco" in urls["ashby"]


# ---------------------------------------------------------------------------
# allocate_board_searches / exploration test
# ---------------------------------------------------------------------------


def test_exploration_includes_unseen_strategy():
    """Empty memory → allocate_board_searches includes unseen strategies."""
    from src.models.config import BoardSearchSettings
    from src.services.board_search import allocate_board_searches
    from src.services.discovery_learning import empty_memory

    memory = empty_memory()
    settings = BoardSearchSettings(max_strategies=4)
    slots = allocate_board_searches(
        memory,
        settings,
        exploration_share=0.20,
        revisit_share=0.10,
        novelty_weight=1.0,
        exploration_bonus=0.35,
        revisit_bonus=0.20,
    )
    assert len(slots) > 0
    keys = [s.key for s in slots]
    # All slots should be "explore" since memory is empty
    assert all(s.reason == "explore" for s in slots)
    # At least one slot per seen strategy
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# apply_board_search_outcomes - exact_url=False not counted as novelty
# ---------------------------------------------------------------------------


def test_url_failure_not_counted_as_novelty():
    """Job with exact_url=False in provenance is NOT counted as new."""
    from src.services.board_search import apply_board_search_outcomes
    from src.services.discovery_learning import empty_memory

    memory = empty_memory()
    job = make_job(
        company="GlobalCorp",
        job_id="pw-001",
        source="workday",
        is_new=True,
        provenance={
            "search_strategy_id": "backend_engineer",
            "method": "playwright",
            "board": "workday",
            "query": "backend engineer",
            "exact_url": False,
        },
    )
    settings = _learning_settings()
    attempts = [
        {
            "key": search_key("workday", "backend_engineer", "playwright"),
            "board": "workday",
            "strategy_id": "backend_engineer",
            "method": "playwright",
            "query": "backend engineer",
        }
    ]
    apply_board_search_outcomes(memory, [job], attempts, settings)

    key = search_key("workday", "backend_engineer", "playwright")
    record = memory["board_searches"][key]
    # is_new=True but exact_url=False → new_count should be 0
    assert record["new_qualified_jobs"] == 0
    assert record["qualified_jobs"] == 1


# ---------------------------------------------------------------------------
# persist_learning dry_run test
# ---------------------------------------------------------------------------


def test_memory_dry_run_does_not_write(tmp_config, tmp_path):
    """persist_learning with dry_run=True does not write to filesystem."""
    from src.models.state import PipelineState, RunSummary

    # tmp_config already has dry_run=True
    state = PipelineState(config=tmp_config, resources={})
    state.jobs = []
    state.summary = RunSummary()

    from src.services.discovery_learning import persist_learning

    memory_path = tmp_path / "learning" / "memory.json"
    # The file should not exist
    assert not memory_path.exists()
    # persist_learning in fixture mode or dry_run should not fail
    persist_learning(state)
    # Still no file written (dry_run=True in tmp_config)
    assert not memory_path.exists()


# ---------------------------------------------------------------------------
# browser failure non-fatal test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_failure_non_fatal(tmp_config, monkeypatch):
    """_run_board_searches with playwright slot and broken browser → attempt recorded with error, no exception."""
    from src.agents.discovery_agent import _run_board_searches
    from src.models.state import PipelineState
    from src.services.discovery_learning import DiscoveryPlan

    slot = BoardSearchSlot(
        key="workday:backend_engineer:playwright",
        board="workday",
        strategy_id="backend_engineer",
        method="playwright",
        query="backend engineer",
        score=0.5,
        reason="test",
    )
    plan = DiscoveryPlan(active=True, board_searches=[slot])

    # Override fixture_mode=False so playwright path is attempted
    config = tmp_config.model_copy(update={"fixture_mode": False})
    # We need board_search.playwright_enabled=True
    from src.models.config import BoardSearchSettings

    new_settings = tmp_config.settings.model_copy(
        update={"board_search": BoardSearchSettings(playwright_enabled=True)}
    )
    config = config.model_copy(update={"settings": new_settings})

    state = PipelineState(config=config, resources={"discovery_plan": plan})

    import src.browser.playwright_runtime as brt

    class BrokenBrowser:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            raise RuntimeError("browser unavailable")

        async def aclose(self):
            pass

    monkeypatch.setattr(brt, "PlaywrightBrowser", BrokenBrowser)

    # Should not raise
    postings, attempts = await _run_board_searches(
        state, None, {"workday": ["https://company.wd1.myworkdayjobs.com/External"]}
    )
    assert postings == []
    assert len(attempts) == 1
    assert attempts[0]["error"] == "RuntimeError"
    assert attempts[0]["board"] == "workday"


# ---------------------------------------------------------------------------
# Workday limits unchanged test
# ---------------------------------------------------------------------------


def test_workday_limits_unchanged():
    """WorkdaySourceSettings defaults preserved; page_size=20."""
    from src.models.config import WorkdaySourceSettings

    ws = WorkdaySourceSettings()
    assert ws.page_size == 20
    assert ws.enabled is True


# ---------------------------------------------------------------------------
# Deduplication test
# ---------------------------------------------------------------------------


def test_duplicate_api_and_playwright_becomes_one():
    """Two RawJobPostings same company+job_id (one API, one playwright) → _dedupe_raw produces 1 posting."""
    from src.agents.discovery_agent import _dedupe_raw

    posting_api = RawJobPosting(
        source="workday",
        company_name="TechCorp",
        title="Backend Engineer",
        job_id="wd-999",
        apply_url="https://techcorp.wd1.myworkdayjobs.com/External/job/Backend-Engineer_JR-999",
    )
    posting_playwright = RawJobPosting(
        source="workday",
        company_name="TechCorp",
        title="Backend Engineer",
        job_id="wd-999",
        apply_url="https://techcorp.wd1.myworkdayjobs.com/External/job/Backend-Engineer_JR-999",
        provenance={"method": "playwright"},
    )
    result = _dedupe_raw([posting_api, posting_playwright])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# THE CRITICAL TEST: Run 1 → Run 2 transition
# ---------------------------------------------------------------------------


def test_learning_run1_to_run2_transition():
    """Run 2 plan differs from Run 1 based on Run 1 outcomes. The critical test."""
    from src.models.config import BoardSearchSettings
    from src.services.board_search import search_key
    from src.services.discovery_learning import (
        build_discovery_plan,
        empty_memory,
        update_memory,
    )

    settings = _learning_settings()
    board_search = BoardSearchSettings(
        enabled=True,
        playwright_enabled=False,
        max_strategies=6,  # Wide enough to see reordering
    )

    # Empty memory → Run 1 plan in config order
    memory = empty_memory()
    plan_run1 = build_discovery_plan(memory, settings, board_search)
    run1_keys = [s.key for s in plan_run1.board_searches]

    # Define the keys we'll measure in Run 1
    api_key = search_key("workday", "backend_engineer", "api")
    pw_backend_key = search_key("workday", "backend_engineer", "playwright")
    pw_frontend_key = search_key("workday", "frontend_engineer", "playwright")

    # Simulate Run 1 outcomes:
    #   Workday API backend → 0 new (repeat job)
    #   Workday Playwright backend → 4 new qualified
    #   Workday Playwright frontend → 1 new qualified
    run1_jobs = [
        _board_job("api-repeat-1", "backend_engineer", "api", "workday", is_new=False),
        _board_job("pw-be-1", "backend_engineer", "playwright", "workday", is_new=True),
        _board_job("pw-be-2", "backend_engineer", "playwright", "workday", is_new=True),
        _board_job("pw-be-3", "backend_engineer", "playwright", "workday", is_new=True),
        _board_job("pw-be-4", "backend_engineer", "playwright", "workday", is_new=True),
        _board_job("pw-fe-1", "frontend_engineer", "playwright", "workday", is_new=True),
    ]
    run1_attempts = [
        {
            "key": api_key,
            "board": "workday",
            "strategy_id": "backend_engineer",
            "method": "api",
            "query": "backend engineer",
        },
        {
            "key": pw_backend_key,
            "board": "workday",
            "strategy_id": "backend_engineer",
            "method": "playwright",
            "query": "backend engineer",
        },
        {
            "key": pw_frontend_key,
            "board": "workday",
            "strategy_id": "frontend_engineer",
            "method": "playwright",
            "query": "frontend engineer",
        },
    ]

    memory_after_run1 = update_memory(
        memory,
        jobs=run1_jobs,
        settings=settings,
        run_id="run1",
        attempts=run1_attempts,
    )

    # Build Run 2 plan from Run 1 memory
    plan_run2 = build_discovery_plan(memory_after_run1, settings, board_search)
    run2_keys = [s.key for s in plan_run2.board_searches]

    # CRITICAL ASSERTIONS:
    # 1. The plans are different
    assert run2_keys != run1_keys, (
        f"Run 2 plan must differ from Run 1.\nRun 1: {run1_keys}\nRun 2: {run2_keys}"
    )

    # 2. Memory records the Run 1 outcomes correctly
    searches = memory_after_run1.get("board_searches", {})
    assert pw_backend_key in searches, f"pw_backend missing from memory: {list(searches.keys())}"
    assert searches[pw_backend_key]["new_qualified_jobs"] == 4
    assert searches[pw_backend_key]["recent_novelty_rate"] == 1.0  # 4/4
    assert searches[api_key]["new_qualified_jobs"] == 0
    assert searches[api_key]["recent_novelty_rate"] == 0.0  # 0/1

    # 3. pw_backend_key must appear in Run 2 plan
    assert pw_backend_key in run2_keys, (
        f"High-performing playwright backend slot must be in Run 2 plan: {run2_keys}"
    )

    # 4. pw_backend_key appears earlier in Run 2 than in Run 1 (or wasn't in Run 1 at all)
    if pw_backend_key in run1_keys:
        idx1 = run1_keys.index(pw_backend_key)
        idx2 = run2_keys.index(pw_backend_key)
        assert idx2 <= idx1, (
            f"pw_backend moved from position {idx1} in Run 1 to {idx2} in Run 2 - should not get worse"
        )

    # 5. pw_backend slot in Run 2 has higher score than api_backend slot
    pw_be_slot = next((s for s in plan_run2.board_searches if s.key == pw_backend_key), None)
    api_be_slot = next((s for s in plan_run2.board_searches if s.key == api_key), None)
    assert pw_be_slot is not None
    if api_be_slot is not None:
        assert pw_be_slot.score >= api_be_slot.score, (
            f"Playwright backend (score={pw_be_slot.score}) must score >= API backend (score={api_be_slot.score})"
        )

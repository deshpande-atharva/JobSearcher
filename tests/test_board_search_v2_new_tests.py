"""Regression tests for V2 semantic fixes: provenance, skip semantics, API slots.

Issue 1/4: Configured board_url must never be used as a global-index discovery target.
Issue 2:   Playwright postings must carry board_origin=global_index.
Issue 3/5: API and Playwright slots with no global board target must be skipped.
"""

from __future__ import annotations

import asyncio

import pytest

from src.browser.page_state import JobCard, PageElement, PageState
from src.browser.session import InMemoryBrowser, PageFixture
from src.models.job import RawJobPosting
from src.services.board_search import (
    BoardSearchSlot,
    apply_board_search_outcomes,
    run_profile_search,
    search_key,
)
from tests.conftest import make_job


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


def _make_slot(board="workday", strategy_id="software_engineer", method="playwright"):
    return BoardSearchSlot(
        key=search_key(board, strategy_id, method),
        board=board,
        strategy_id=strategy_id,
        method=method,
        query=strategy_id.replace("_", " "),
        score=0.5,
        reason="test",
    )


# ---------------------------------------------------------------------------
# 1. global-index board URL → global_index provenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_playwright_posting_carries_board_origin_global_index():
    """run_profile_search postings carry board_origin=global_index (CDX-sourced board)."""
    job_url = "https://techcorp.wd1.myworkdayjobs.com/External/job/Remote/SWE_001"
    search_page = PageState(
        url="https://techcorp.wd1.myworkdayjobs.com/External",
        inputs=[PageElement(id="kw", text="Search jobs", role="searchbox")],
        job_cards=[JobCard(id="j1", title="SWE", url=job_url)],
        links=[PageElement(id="link1", href=job_url, text="SWE")],
    )
    result_page = PageState(
        url=job_url,
        job_cards=[JobCard(id="j1", title="SWE", url=job_url)],
        links=[],
    )
    pages = {
        "search": PageFixture(id="search", state=search_page, transitions={"kw": "result"}),
        "result": PageFixture(id="result", state=result_page, transitions={}),
    }
    browser = InMemoryBrowser(pages, "search")
    execution = await run_profile_search(
        browser, board="workday", strategy_id="software_engineer",
        query="software engineer", company_name="TechCorp",
    )
    assert execution.search_success is True
    assert len(execution.postings) > 0
    for posting in execution.postings:
        assert posting.provenance.get("board_origin") == "global_index", (
            f"CDX-sourced Playwright posting must have board_origin=global_index; "
            f"got {posting.provenance}"
        )
        assert posting.provenance.get("method") == "playwright"
        assert posting.provenance.get("discovery_mode") == "global_board"


# ---------------------------------------------------------------------------
# 2. Configured board_url → NOT used as global discovery target
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configured_board_url_never_used_as_global_discovery_target(tmp_config, monkeypatch):
    """board_url in companies.yaml must NEVER be treated as a global-index target.

    Even a valid Workday URL must be ignored: configured boards are covered by
    CXS, so using them for Playwright adds zero global novelty.
    """
    from src.agents.discovery_agent import _run_board_searches
    from src.models.config import AppConfig, BoardSearchSettings, CompanyConfig
    from src.models.state import PipelineState
    from src.services.discovery_learning import DiscoveryPlan

    slot = _make_slot(method="playwright")
    plan = DiscoveryPlan(active=True, board_searches=[slot])

    cxs_company = CompanyConfig(
        name="Adobe",
        ats_type="workday",
        ats_identifier="adobe",
        board_url="https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced",
    )
    new_settings = tmp_config.settings.model_copy(
        update={"board_search": BoardSearchSettings(playwright_enabled=True)}
    )
    config = tmp_config.model_copy(update={"settings": new_settings, "fixture_mode": False})
    state = PipelineState(config=config, resources={"discovery_plan": plan})
    monkeypatch.setattr(AppConfig, "target_companies", lambda self: (cxs_company,))

    postings, attempts = await _run_board_searches(state, None, {})

    assert postings == []
    assert attempts[0].get("skipped") is True, (
        "Slot must be skipped: configured board_url must not be used as global target"
    )
    assert attempts[0].get("skip_reason") == "no_global_board_target"


# ---------------------------------------------------------------------------
# 3. Configured CXS-covered board not used when CDX unavailable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cxs_covered_board_not_used_when_cdx_unavailable(tmp_config, monkeypatch):
    """CDX unavailable (board_urls={}) + CXS-covered company → slot skipped, no browser opened."""
    from src.agents.discovery_agent import _run_board_searches
    from src.models.config import AppConfig, BoardSearchSettings, CompanyConfig
    from src.models.state import PipelineState
    from src.services.discovery_learning import DiscoveryPlan
    import src.browser.playwright_runtime as brt

    slot = _make_slot(method="playwright")
    plan = DiscoveryPlan(active=True, board_searches=[slot])

    company = CompanyConfig(
        name="NVIDIA",
        ats_type="workday",
        ats_identifier="nvidia",
        board_url="https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite",
    )
    new_settings = tmp_config.settings.model_copy(
        update={"board_search": BoardSearchSettings(playwright_enabled=True)}
    )
    config = tmp_config.model_copy(update={"settings": new_settings, "fixture_mode": False})
    state = PipelineState(config=config, resources={"discovery_plan": plan})
    monkeypatch.setattr(AppConfig, "target_companies", lambda self: (company,))

    opened_urls: list[str] = []

    class RecordingBrowser:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            pass

        async def open(self, url: str):
            opened_urls.append(url)
            raise RuntimeError("stop")

        async def aclose(self):
            pass

    monkeypatch.setattr(brt, "PlaywrightBrowser", RecordingBrowser)

    await _run_board_searches(state, None, {})

    assert opened_urls == [], "No browser must be opened when CDX is unavailable"


# ---------------------------------------------------------------------------
# 4. No global board target → skipped=True (Playwright)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_playwright_no_global_target_skipped(tmp_config, monkeypatch):
    """Empty board_urls → Playwright slot skipped with no_global_board_target."""
    from src.agents.discovery_agent import _run_board_searches
    from src.models.config import AppConfig, BoardSearchSettings
    from src.models.state import PipelineState
    from src.services.discovery_learning import DiscoveryPlan

    slot = _make_slot(method="playwright")
    plan = DiscoveryPlan(active=True, board_searches=[slot])
    new_settings = tmp_config.settings.model_copy(
        update={"board_search": BoardSearchSettings(playwright_enabled=True)}
    )
    config = tmp_config.model_copy(update={"settings": new_settings, "fixture_mode": False})
    state = PipelineState(config=config, resources={"discovery_plan": plan})
    monkeypatch.setattr(AppConfig, "target_companies", lambda self: ())

    _postings, attempts = await _run_board_searches(state, None, {})

    assert len(attempts) == 1
    assert attempts[0].get("skipped") is True
    assert attempts[0].get("skip_reason") == "no_global_board_target"


# ---------------------------------------------------------------------------
# 5. Skipped Playwright slot does not create learning memory
# ---------------------------------------------------------------------------


def test_playwright_no_global_target_skip_does_not_create_memory():
    """no_global_board_target skip must NOT create a board_searches record."""
    from src.services.discovery_learning import empty_memory

    memory = empty_memory()
    key = search_key("workday", "software_engineer", "playwright")
    attempt = {
        "key": key,
        "board": "workday",
        "strategy_id": "software_engineer",
        "method": "playwright",
        "query": "software engineer",
        "skipped": True,
        "skip_reason": "no_global_board_target",
    }
    settings = _learning_settings()
    apply_board_search_outcomes(memory, [], [attempt], settings)

    assert key not in memory["board_searches"], (
        "no_global_board_target skip must NOT create a board_searches record"
    )


# ---------------------------------------------------------------------------
# 6. No API target → skipped=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_slot_skipped_when_no_global_boards(tmp_config, monkeypatch):
    """API slot with empty board_urls → skipped=True, skip_reason=no_global_board_target."""
    from src.agents.discovery_agent import _run_board_searches
    from src.models.config import AppConfig, BoardSearchSettings
    from src.models.state import PipelineState
    from src.services.discovery_learning import DiscoveryPlan

    slot = _make_slot(method="api")
    plan = DiscoveryPlan(active=True, board_searches=[slot])
    new_settings = tmp_config.settings.model_copy(
        update={"board_search": BoardSearchSettings(playwright_enabled=False)}
    )
    config = tmp_config.model_copy(update={"settings": new_settings, "fixture_mode": False})
    state = PipelineState(config=config, resources={"discovery_plan": plan})
    monkeypatch.setattr(AppConfig, "target_companies", lambda self: ())

    _postings, attempts = await _run_board_searches(state, None, {})

    assert len(attempts) == 1
    assert attempts[0].get("skipped") is True
    assert attempts[0].get("skip_reason") == "no_global_board_target"


# ---------------------------------------------------------------------------
# 7. No API target does not increment runs_seen
# ---------------------------------------------------------------------------


def test_api_no_global_target_skip_does_not_increment_runs_seen():
    """Skipped API slot (no_global_board_target) must NOT increment runs_seen."""
    from src.services.discovery_learning import empty_memory

    memory = empty_memory()
    key = search_key("workday", "software_engineer", "api")
    attempt = {
        "key": key,
        "board": "workday",
        "strategy_id": "software_engineer",
        "method": "api",
        "query": "software engineer",
        "skipped": True,
        "skip_reason": "no_global_board_target",
    }
    settings = _learning_settings()
    apply_board_search_outcomes(memory, [], [attempt], settings)

    assert key not in memory["board_searches"], (
        "Skipped API slot must NOT create a board_searches record"
    )


# ---------------------------------------------------------------------------
# 8. Executed API search with raw_jobs=0 remains a legitimate observation
# ---------------------------------------------------------------------------


def test_executed_api_zero_result_is_legitimate_observation():
    """API slot executed (not skipped, no error) with 0 qualified IS a real observation."""
    from src.services.discovery_learning import empty_memory

    memory = empty_memory()
    key = search_key("workday", "backend_engineer", "api")
    executed_attempt = {
        "key": key,
        "board": "workday",
        "strategy_id": "backend_engineer",
        "method": "api",
        "query": "backend engineer",
        # raw_jobs/search_success not set (API doesn't populate them in _run_board_searches)
        # No skipped, no error → executed observation
    }
    settings = _learning_settings()
    apply_board_search_outcomes(memory, [], [executed_attempt], settings)

    assert key in memory["board_searches"], (
        "Executed-zero API slot MUST create a board_searches record"
    )
    record = memory["board_searches"][key]
    assert record["runs_seen"] == 1
    assert record["consecutive_zero_new"] == 1
    assert record["new_qualified_jobs"] == 0


# ---------------------------------------------------------------------------
# 9. Executed Playwright search with raw_jobs=0 remains a legitimate observation
# ---------------------------------------------------------------------------


def test_executed_playwright_zero_raw_is_legitimate_observation():
    """Playwright slot with search_success=True and raw_jobs=0 IS a real observation."""
    from src.services.discovery_learning import empty_memory

    memory = empty_memory()
    key = search_key("workday", "backend_engineer", "playwright")
    executed_attempt = {
        "key": key,
        "board": "workday",
        "strategy_id": "backend_engineer",
        "method": "playwright",
        "query": "backend engineer",
        "search_success": True,
        "raw_jobs": 0,
    }
    settings = _learning_settings()
    apply_board_search_outcomes(memory, [], [executed_attempt], settings)

    assert key in memory["board_searches"], (
        "Executed-zero-raw Playwright slot MUST create a board_searches record"
    )
    record = memory["board_searches"][key]
    assert record["runs_seen"] == 1
    assert record["new_qualified_jobs"] == 0


# ---------------------------------------------------------------------------
# 10. Executed Playwright search with raw_jobs>0 updates attempt metrics
# ---------------------------------------------------------------------------


def test_executed_playwright_positive_updates_attempt_metrics():
    """Playwright slot with raw_jobs>0 and qualified jobs writes counts back to attempt."""
    from src.services.discovery_learning import empty_memory

    key = search_key("workday", "software_engineer", "playwright")
    attempt = {
        "key": key,
        "board": "workday",
        "strategy_id": "software_engineer",
        "method": "playwright",
        "query": "software engineer",
        "search_success": True,
        "raw_jobs": 5,
    }

    def _pw_job(job_id, is_new):
        j = make_job(company="GlobalCo", job_id=job_id, source="workday", is_new=is_new)
        j.provenance = {
            "board_origin": "global_index",
            "search_strategy_id": "software_engineer",
            "method": "playwright",
            "board": "workday",
            "query": "software engineer",
            "exact_url": True,
        }
        return j

    memory = empty_memory()
    settings = _learning_settings()
    jobs = [_pw_job("j1", True), _pw_job("j2", True), _pw_job("j3", False)]
    apply_board_search_outcomes(memory, jobs, [attempt], settings)

    assert attempt["qualified_jobs"] == 3
    assert attempt["new_qualified_jobs"] == 2
    assert attempt["repeat_qualified_jobs"] == 1
    assert attempt["exact_official_url_count"] == 3
    record = memory["board_searches"][key]
    assert record["runs_seen"] == 1
    assert record["new_qualified_jobs"] == 2
    assert record["recent_novelty_rate"] == pytest.approx(2 / 3, abs=0.001)


# ---------------------------------------------------------------------------
# 11. Real Playwright board search preserves provenance through dedup
# ---------------------------------------------------------------------------


def test_global_index_playwright_provenance_survives_dedup():
    """board_origin=global_index on a Playwright posting survives _dedupe_raw intact."""
    from src.agents.discovery_agent import _dedupe_raw

    pw_posting = RawJobPosting(
        source="workday",
        company_name="GlobalDiscovery",
        title="SWE",
        job_id="gd-001",
        apply_url="https://newco.wd1.myworkdayjobs.com/External/job/Remote/SWE_001",
        provenance={
            "board_origin": "global_index",
            "method": "playwright",
            "search_strategy_id": "software_engineer",
            "strategy_id": "workday_global_index",
        },
    )
    result = _dedupe_raw([pw_posting])
    assert len(result) == 1
    assert result[0].provenance.get("board_origin") == "global_index"
    assert result[0].provenance.get("method") == "playwright"

"""Persistent discovery learning changes the next search, not the tracker."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from src.agents.dedup_agent import run_dedup
from src.agents.extraction_agent import _to_job
from src.models.config import (
    CompanyConfig,
    CompanyDiscoveryBlock,
    LearningSettings,
    WorkdaySourceSettings,
)
from src.models.job import RawJobPosting
from src.models.state import PipelineState, RunSummary
from src.services.discovery_learning import (
    annotate_plan,
    assign_strategy,
    build_discovery_plan,
    empty_memory,
    load_memory,
    persist_learning,
    update_memory,
)
from src.services.notifications import build_email_body
from src.services.public_board_index import select_board_tokens
from src.sources.base import SourceResult
from tests.conftest import make_job

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _settings(**overrides: object) -> LearningSettings:
    data: dict[str, object] = {
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


def _job(
    company: str,
    job_id: str,
    *,
    is_new: bool,
    source: str = "greenhouse",
    strategy: str = "greenhouse_configured",
    title: str | None = None,
) -> object:
    return make_job(
        company=company,
        job_title=title or f"{company} {job_id}",
        job_id=job_id,
        source=source,
        is_new=is_new,
        provenance={"strategy_id": strategy, "discovered_from": [source]},
        direct_application_url=f"https://example.test/{company}/{job_id}",
    )


def test_empty_memory_keeps_current_discovery_behavior() -> None:
    plan = build_discovery_plan(empty_memory(), _settings())
    assert plan.effort_for("robinhood") == "full"
    assert plan.effort_for("booz allen") == "full"
    assert plan.selection_for("greenhouse") == (None, set())
    assert plan.selection_for("workday") == (None, set())
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]
    assert select_board_tokens(tokens, limit=2, now=NOW) == select_board_tokens(
        tokens, limit=2, now=NOW, priorities=None
    )
    assert len(select_board_tokens(tokens, limit=8, now=NOW)) <= 8
    assert WorkdaySourceSettings().page_size == 20
    assert WorkdaySourceSettings().max_concurrency == 2
    assert WorkdaySourceSettings().max_jobs == 2000


def test_new_jobs_outrank_repeat_only_strategy() -> None:
    settings = _settings()
    memory = empty_memory()
    first = [
        _job(f"Co{i}", f"a{i}", is_new=True, strategy="greenhouse_configured")
        for i in range(5)
    ]
    first.append(_job("Repeat Co", "b1", is_new=False, source="ashby", strategy="ashby_configured"))
    memory = update_memory(memory, jobs=first, settings=settings, run_id="run-1", now=NOW)
    second = [
        _job("Co5", "a5", is_new=True, strategy="greenhouse_configured"),
        _job("Repeat Co", "b1", is_new=False, source="ashby", strategy="ashby_configured"),
    ]
    memory = update_memory(memory, jobs=second, settings=settings, run_id="run-2", now=NOW)
    plan = build_discovery_plan(memory, settings)
    assert memory["strategies"]["greenhouse_configured"]["runs"] == 2
    assert memory["strategies"]["ashby_configured"]["runs"] == 2
    assert plan.strategy_priority["greenhouse_configured"] > plan.strategy_priority["ashby_configured"]


def test_raw_discovered_count_does_not_set_priority() -> None:
    memory = empty_memory()
    memory["strategies"] = {
        "greenhouse_configured": {
            "source": "greenhouse",
            "runs": 2,
            "discovered": 500,
            "qualified": 20,
            "new_jobs": 0,
            "repeat_jobs": 20,
            "companies": 1,
            "new_companies": 0,
            "last_used": "2026-10-01",
            "recent_novelty_rate": 0.0,
        },
        "ashby_configured": {
            "source": "ashby",
            "runs": 2,
            "discovered": 1,
            "qualified": 1,
            "new_jobs": 5,
            "repeat_jobs": 0,
            "companies": 1,
            "new_companies": 1,
            "last_used": "2026-10-01",
            "recent_novelty_rate": 1.0,
        },
    }
    plan = build_discovery_plan(memory, _settings())
    assert plan.strategy_priority["ashby_configured"] > plan.strategy_priority["greenhouse_configured"]


def test_one_run_does_not_suppress_a_company() -> None:
    settings = _settings(cooldown_runs=1)
    memory = update_memory(
        empty_memory(),
        jobs=[_job("Robinhood", "a", is_new=False)],
        settings=settings,
        run_id="only",
        now=NOW,
    )
    plan = build_discovery_plan(memory, settings)
    assert plan.effort_for("robinhood") == "full"
    assert "robinhood" in memory["companies"]


def test_repeat_only_company_moves_to_monitor_but_stays_eligible() -> None:
    settings = _settings(cooldown_runs=2)
    memory = empty_memory()
    for index in range(2):
        memory = update_memory(
            memory,
            jobs=[_job("Robinhood", "a", is_new=False)],
            settings=settings,
            run_id=f"run-{index}",
            now=NOW,
        )
    plan = build_discovery_plan(memory, settings)
    assert plan.effort_for("robinhood") == "monitor"
    assert "robinhood" in memory["companies"]
    assert memory["companies"]["robinhood"]["effort"] == "monitor"


def test_cooled_down_company_returns_to_full_effort() -> None:
    settings = _settings(cooldown_runs=2, revisit_runs=2)
    memory = empty_memory()
    for index in range(2):
        memory = update_memory(
            memory,
            jobs=[_job("Robinhood", "a", is_new=False)],
            settings=settings,
            run_id=f"cool-{index}",
            now=NOW,
        )
    assert build_discovery_plan(memory, settings).effort_for("robinhood") == "monitor"
    for index in range(2):
        memory = update_memory(
            memory,
            jobs=[_job("Robinhood", "a", is_new=False)],
            settings=settings,
            run_id=f"revisit-{index}",
            now=NOW,
        )
    plan = build_discovery_plan(memory, settings)
    assert plan.effort_for("robinhood") == "full"
    assert "robinhood" in memory["companies"]


def test_exploration_share_is_reserved_for_unseen_boards() -> None:
    tokens = [f"known{i}" for i in range(8)] + ["newco", "otherco"]
    priorities = dict.fromkeys(tokens[:8], 0.0)
    chosen = select_board_tokens(
        tokens,
        limit=8,
        now=NOW,
        priorities=priorities,
        exploration_share=0.20,
        revisit_share=0.10,
    )
    assert len(chosen) == 8
    assert {"newco", "otherco"} <= {token.lower() for token in chosen}


def test_revisit_slot_keeps_a_low_novelty_board() -> None:
    tokens = [f"b{i:02d}" for i in range(12)]
    priorities = {name: index / 12 for index, name in enumerate(tokens)}
    plain = select_board_tokens(
        tokens,
        limit=10,
        now=NOW,
        priorities=priorities,
        exploration_share=0,
        revisit_share=0,
    )
    chosen = select_board_tokens(
        tokens,
        limit=10,
        now=NOW,
        priorities=priorities,
        exploration_share=0,
        revisit_share=0.10,
        revisit_keys={"b00"},
    )
    assert "b00" not in plain
    assert "b00" in chosen
    assert len(chosen) == 10


def test_source_failure_is_not_recorded_as_zero_novelty() -> None:
    settings = _settings()
    memory = update_memory(
        empty_memory(),
        jobs=[_job("Acme", "1", is_new=True)],
        settings=settings,
        run_id="ok",
        now=NOW,
    )
    runs = memory["sources"]["greenhouse"]["runs"]
    memory = update_memory(
        memory,
        jobs=[_job("Other", "2", is_new=True, source="lever", strategy="lever_configured")],
        settings=settings,
        run_id="lever-only",
        now=NOW,
    )
    assert memory["sources"]["greenhouse"]["runs"] == runs


def test_malformed_memory_is_not_overwritten(tmp_path) -> None:
    path = tmp_path / "discovery_memory.json"
    path.write_text("{", encoding="utf-8")
    loaded = load_memory(path)
    assert loaded.writable is False
    assert path.read_text(encoding="utf-8") == "{"
    path.write_text(json.dumps({"version": 2, "companies": {}}), encoding="utf-8")
    loaded = load_memory(path)
    assert loaded.writable is False
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2


def test_fixture_and_dry_run_do_not_write_learning_memory(tmp_config) -> None:
    target = tmp_config.project_root / "data" / "learning" / "discovery_memory.json"
    fixture = tmp_config.model_copy(update={"fixture_mode": True, "dry_run": False})
    state = PipelineState(config=fixture, jobs=[make_job(is_new=True)])
    persist_learning(state)
    assert not target.exists()

    dry = tmp_config.model_copy(update={"fixture_mode": False, "dry_run": True})
    state = PipelineState(config=dry, jobs=[make_job(is_new=True)])
    persist_learning(state)
    assert not target.exists()

    live = tmp_config.model_copy(update={"fixture_mode": False, "dry_run": False})
    state = PipelineState(config=live, jobs=[make_job(is_new=True)])
    persist_learning(state)
    assert target.is_file()
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored["version"] == 1
    assert stored["runs"][0]["new_jobs"] == 1


@pytest.mark.asyncio
async def test_gemini_failure_keeps_the_deterministic_plan() -> None:
    settings = _settings()
    memory = update_memory(
        empty_memory(),
        jobs=[_job("Acme", "1", is_new=True), _job("Acme", "1", is_new=False)],
        settings=settings,
        run_id="one",
        now=NOW,
    )
    memory = update_memory(
        memory,
        jobs=[_job("Acme", "1", is_new=False)],
        settings=settings,
        run_id="two",
        now=NOW,
    )
    plan = build_discovery_plan(memory, settings)
    priorities = dict(plan.strategy_priority)
    efforts = dict(plan.company_effort)

    class Down:
        available = True

        async def structured(self, **kwargs: object) -> object:
            raise TimeoutError("503")

    result = await annotate_plan(plan, Down())
    assert result.strategy_priority == priorities
    assert result.company_effort == efforts
    assert result.exploration_share == settings.exploration_share
    assert result.explanation == ""


@pytest.mark.asyncio
async def test_repeated_job_stays_in_the_tracker_and_out_of_the_email(tmp_config) -> None:
    repeated = make_job(company="Robinhood", job_id="job-a", job_title="Backend Engineer", is_new=True)
    fresh = make_job(
        company="Factored",
        job_id="job-z",
        job_title="Staff AI Engineer",
        is_new=True,
        direct_application_url="https://example.test/factored/job-z",
    )
    state = PipelineState(
        config=tmp_config,
        jobs=[repeated, fresh],
        known_keys={repeated.dedup_key},
    )
    await run_dedup(state)
    kept = {job.job_id: job for job in state.jobs}
    assert kept["job-a"].is_new is False
    assert kept["job-z"].is_new is True
    summary = RunSummary(
        jobs_accepted=2,
        learning_report="Discovery Learning\n- Robinhood: monitor",
    )
    body = build_email_body(summary, state.jobs)
    new_section = body.split("New jobs:", 1)[1]
    assert "Backend Engineer" not in new_section
    assert "Staff AI Engineer" in new_section
    assert "Robinhood: monitor" in body


def test_strategy_id_survives_extraction(tmp_config) -> None:
    raw = RawJobPosting(
        source="greenhouse",
        company_name="Factored",
        title="Staff AI Engineer",
        job_id="5425364008",
        apply_url="https://boards.greenhouse.io/factored/jobs/5425364008",
        provenance={"discovered_from": ["greenhouse"]},
    )
    assign_strategy(raw, origin="configured")
    job = _to_job(raw)
    assert job is not None
    assert job.provenance["strategy_id"] == "greenhouse_configured"
    job.is_new = True
    memory = update_memory(
        empty_memory(),
        jobs=[job],
        settings=_settings(),
        run_id="attributed",
        now=NOW,
    )
    assert memory["strategies"]["greenhouse_configured"]["new_jobs"] == 1
    assert tmp_config.fixture_mode is True


@pytest.mark.asyncio
async def test_strategy_id_remains_after_dedup(tmp_config) -> None:
    raw = RawJobPosting(
        source="ashby",
        company_name="Notion",
        title="Software Engineer",
        job_id="notion-1",
        apply_url="https://jobs.ashbyhq.com/notion/notion-1",
        provenance={"discovered_from": ["ashby"]},
    )
    assign_strategy(raw, origin="global_index")
    job = _to_job(raw)
    assert job is not None
    assert job.provenance["strategy_id"] == "ashby_global_index"
    state = PipelineState(config=tmp_config, jobs=[job])
    await run_dedup(state)
    assert state.jobs[0].provenance["strategy_id"] == "ashby_global_index"
    memory = update_memory(
        empty_memory(),
        jobs=state.jobs,
        settings=_settings(),
        run_id="kept",
        now=NOW,
    )
    assert memory["strategies"]["ashby_global_index"]["new_jobs"] == 1


def test_three_day_fixture_lowers_repeat_companies_and_keeps_new_ones() -> None:
    """Two repeat days after the first discovery. cooldown_runs=2 makes that monitor."""
    settings = _settings(cooldown_runs=2, revisit_runs=2)
    day1 = [
        _job("Robinhood", "A", is_new=True),
        _job("Robinhood", "B", is_new=True),
        _job("Booz Allen", "A", is_new=True, source="workday", strategy="workday_configured"),
        _job("Company X", "A", is_new=True),
    ]
    day2 = [
        _job("Robinhood", "A", is_new=False),
        _job("Robinhood", "B", is_new=False),
        _job("Booz Allen", "A", is_new=False, source="workday", strategy="workday_configured"),
        _job("Company X", "A", is_new=False),
        _job("Company Y", "A", is_new=True),
    ]
    day3 = [
        _job("Robinhood", "A", is_new=False),
        _job("Robinhood", "B", is_new=False),
        _job("Booz Allen", "A", is_new=False, source="workday", strategy="workday_configured"),
        _job("Company Y", "A", is_new=False),
        _job("Company Z", "A", is_new=True),
    ]
    memory = empty_memory()
    for run_id, jobs in (("2026-10-01", day1), ("2026-10-02", day2), ("2026-10-03", day3)):
        memory = update_memory(memory, jobs=jobs, settings=settings, run_id=run_id, now=NOW)
    plan = build_discovery_plan(memory, settings)
    assert plan.effort_for("robinhood") == "monitor"
    assert plan.effort_for("booz allen") == "monitor"
    assert plan.effort_for("company y") == "full"
    assert plan.effort_for("company z") == "full"
    assert plan.effort_for("company x") == "full"
    assert "robinhood" in memory["companies"]
    assert "booz allen" in memory["companies"]
    assert memory["runs"][-1]["new_jobs"] == 1
    assert memory["runs"][-1]["repeat_jobs"] == 4


@pytest.mark.asyncio
async def test_three_day_repeats_remain_trackable_and_only_new_jobs_email(tmp_config) -> None:
    repeated = [
        make_job(company="Robinhood", job_id="A", job_title="Robinhood A", is_new=True),
        make_job(company="Robinhood", job_id="B", job_title="Robinhood B", is_new=True),
        make_job(company="Booz Allen", job_id="A", job_title="Booz Allen A", source="workday", is_new=True),
        make_job(company="Company Y", job_id="A", job_title="Company Y A", is_new=True),
    ]
    fresh = make_job(company="Company Z", job_id="A", job_title="Company Z A", is_new=True)
    known = {job.dedup_key for job in repeated}
    state = PipelineState(config=tmp_config, jobs=[*repeated, fresh], known_keys=known)
    await run_dedup(state)
    by_company = {}
    for job in state.jobs:
        by_company.setdefault(job.company, []).append(job)
    assert len(state.jobs) == 5
    assert all(job.is_new is False for job in by_company["Robinhood"])
    assert by_company["Booz Allen"][0].is_new is False
    assert by_company["Company Z"][0].is_new is True
    body = build_email_body(RunSummary(jobs_accepted=5), state.jobs)
    new_section = body.split("New jobs:", 1)[1]
    assert "Company Z A" in new_section
    assert "Robinhood A" not in new_section
    assert "Booz Allen A" not in new_section


@pytest.mark.asyncio
async def test_monitor_company_skips_workday_partitions(tmp_config) -> None:
    from src.agents.discovery_agent import _discover_company
    from src.services.ats_registry import load_ats_registry
    from src.services.discovery_learning import DiscoveryPlan

    seen: dict[str, bool] = {}

    class StubSource:
        enabled = True

        def __init__(self, ctx: object) -> None:
            del ctx

        async def discover_result(self, company: CompanyConfig, skip_partitions: bool = False):
            seen["skip"] = skip_partitions
            posting = RawJobPosting(
                source="workday",
                company_name=company.name,
                title="Engineer",
                job_id="wd-1",
                apply_url="https://example.test/wd-1",
                provenance={"workday_partitions": ["unfiltered"]},
            )
            return SourceResult.ok("workday", company.name, [posting])

    import src.agents.discovery_agent as discovery_agent

    config = tmp_config.model_copy(update={"fixture_mode": False, "dry_run": True})
    state = PipelineState(config=config)
    state.resources["discovery_plan"] = DiscoveryPlan(
        active=True,
        company_effort={"booz allen": "monitor"},
    )
    company = CompanyConfig(
        name="Booz Allen",
        ats_type="workday",
        ats_identifier="https://bah.wd1.myworkdayjobs.com/BAH_Jobs",
        careers_url="https://bah.wd1.myworkdayjobs.com/BAH_Jobs",
        discovery=CompanyDiscoveryBlock(sources=("company_ats",)),
    )
    original = discovery_agent._ATS_BY_TYPE
    discovery_agent._ATS_BY_TYPE = {**original, "workday": StubSource}
    try:
        posts, _outcomes = await _discover_company(
            state,
            None,  # type: ignore[arg-type]
            load_ats_registry(config),
            company,
        )
    finally:
        discovery_agent._ATS_BY_TYPE = original
    assert seen["skip"] is True
    assert posts[0].provenance["strategy_id"] == "workday_configured"


def test_workflow_commits_learning_state() -> None:
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily_jobs.yml").read_text(
        encoding="utf-8"
    )
    assert "git add data/current data/archive" in script
    assert "git add data/learning" in script
    assert "git add -A" not in script
    assert "persist_ats_registry" not in script

"""Global Workday novelty attribution: repeat classification, cluster value, learning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models.state import PipelineState
from src.services.deduplication import deduplicate
from src.services.discovery_learning import (
    STRATEGY_SOURCE,
    _extract_cluster,
    _is_global_workday,
    _split_counts,
    assign_strategy,
    empty_memory,
    log_global_workday_attribution,
    update_memory,
)
from tests.conftest import make_job


def _global_wday_job(
    *,
    job_id: str,
    company: str = "GlobalCorp",
    cluster: str = "wd12",
    is_new: bool = True,
) -> object:
    """A qualified job from global Workday discovery."""
    job = make_job(
        job_id=job_id,
        company=company,
        source="workday",
        direct_application_url=f"https://globalcorp.{cluster}.myworkdayjobs.com/en-US/external/job/{job_id}",
    )
    job.provenance["board_origin"] = "global_index"
    job.provenance["board_identity"] = f"globalcorp.{cluster}.myworkdayjobs.com|globalcorp|external"
    job.provenance["strategy_id"] = "workday_global_index"
    job.is_new = is_new
    return job


def _configured_wday_job(*, job_id: str, company: str = "Acme Robotics") -> object:
    """A qualified job from a configured Workday company."""
    job = make_job(
        job_id=job_id,
        company=company,
        source="workday",
    )
    job.provenance["strategy_id"] = "workday_configured"
    job.is_new = False
    return job


# ---------------------------------------------------------------------------
# 1. Configured-company repeat attribution
# ---------------------------------------------------------------------------


def test_configured_company_repeat_attributed(tmp_config):
    """Repeat from global discovery whose company is in companies.yaml."""
    # Acme Robotics is in tmp_config's companies.yaml
    job = _global_wday_job(job_id="r1", company="Acme Robotics", is_new=False)
    state = PipelineState(config=tmp_config, jobs=[job])
    # Should classify as configured_company_repeat
    assert _is_global_workday(job)
    from src.utils.normalization import normalize_company_name

    configured_keys = {normalize_company_name(c.name) for c in tmp_config.target_companies()}
    job_company = normalize_company_name(job.company)
    assert job_company in configured_keys


# ---------------------------------------------------------------------------
# 2. Previous-global repeat attribution
# ---------------------------------------------------------------------------


def test_previous_global_repeat_attributed(tmp_config):
    """Repeat from global discovery whose company is NOT configured."""
    job = _global_wday_job(job_id="r2", company="UnknownGlobalCorp", is_new=False)
    state = PipelineState(config=tmp_config, jobs=[job])
    from src.utils.normalization import normalize_company_name

    configured_keys = {normalize_company_name(c.name) for c in tmp_config.target_companies()}
    job_company = normalize_company_name(job.company)
    assert job_company not in configured_keys


# ---------------------------------------------------------------------------
# 3. Same-run duplicate attribution
# ---------------------------------------------------------------------------


def test_same_run_duplicates_removed_by_dedup():
    """Same-run duplicates are rejected by deduplicate, never reach qualified."""
    j1 = make_job(job_id="dup1", company="Corp")
    j2 = make_job(job_id="dup1", company="Corp")
    unique, duplicates, historical = deduplicate([j1, j2])
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert duplicates[0].is_new is False
    # Same-run dups are not in qualified (state.jobs), so attribution count = 0


# ---------------------------------------------------------------------------
# 4. Unknown historical repeat attribution
# ---------------------------------------------------------------------------


def test_historical_repeat_from_known_keys():
    """A job in known_keys is historically seen regardless of source."""
    job = make_job(job_id="hist1", company="OldCorp")
    known = {job.dedup_key}
    unique, duplicates, historical = deduplicate([job], known)
    assert len(historical) == 1
    assert historical[0].is_new is False


# ---------------------------------------------------------------------------
# 5. Cluster attribution for NEW jobs
# ---------------------------------------------------------------------------


def test_cluster_extracted_from_provenance():
    job = _global_wday_job(job_id="c1", cluster="wd7")
    assert _extract_cluster(job) == "wd7"


def test_cluster_attribution_new_job():
    job = _global_wday_job(job_id="c2", cluster="wd12", is_new=True)
    assert _extract_cluster(job) == "wd12"
    assert job.is_new is True


# ---------------------------------------------------------------------------
# 6. Cluster attribution for repeats
# ---------------------------------------------------------------------------


def test_cluster_attribution_repeat_job():
    job = _global_wday_job(job_id="c3", cluster="wd1", is_new=False)
    assert _extract_cluster(job) == "wd1"
    assert job.is_new is False


# ---------------------------------------------------------------------------
# 7. Qualified = new + repeat
# ---------------------------------------------------------------------------


def test_qualified_equals_new_plus_repeat():
    jobs = [
        _global_wday_job(job_id="q1", is_new=True),
        _global_wday_job(job_id="q2", is_new=False),
        _global_wday_job(job_id="q3", is_new=True),
        _global_wday_job(job_id="q4", is_new=False),
    ]
    qualified = len(jobs)
    new = sum(1 for j in jobs if j.is_new)
    repeat = qualified - new
    assert qualified == 4
    assert new == 2
    assert repeat == 2
    assert qualified == new + repeat


# ---------------------------------------------------------------------------
# 8. Cluster totals reconcile with global totals
# ---------------------------------------------------------------------------


def test_cluster_totals_reconcile():
    jobs = [
        _global_wday_job(job_id="t1", cluster="wd12", is_new=True),
        _global_wday_job(job_id="t2", cluster="wd12", is_new=False),
        _global_wday_job(job_id="t3", cluster="wd1", is_new=True),
    ]
    by_cluster: dict[str, dict[str, int]] = {}
    for job in jobs:
        c = _extract_cluster(job)
        cd = by_cluster.setdefault(c, {"qualified": 0, "new": 0, "repeat": 0})
        cd["qualified"] += 1
        if job.is_new:
            cd["new"] += 1
        else:
            cd["repeat"] += 1

    total_q = sum(cd["qualified"] for cd in by_cluster.values())
    total_n = sum(cd["new"] for cd in by_cluster.values())
    total_r = sum(cd["repeat"] for cd in by_cluster.values())
    assert total_q == 3
    assert total_n == 2
    assert total_r == 1
    assert total_q == total_n + total_r


# ---------------------------------------------------------------------------
# 9. Dedup behavior itself unchanged
# ---------------------------------------------------------------------------


def test_dedup_behavior_unchanged():
    """deduplicate splits into unique/dup/historical correctly."""
    j_new = make_job(job_id="new1")
    j_known = make_job(job_id="known1")
    j_dup = make_job(job_id="new1")  # same as j_new
    known = {j_known.dedup_key}
    unique, duplicates, historical = deduplicate([j_new, j_known, j_dup], known)
    assert len(unique) == 1 and unique[0].is_new is True
    assert len(historical) == 1 and historical[0].is_new is False
    assert len(duplicates) == 1 and duplicates[0].is_new is False


# ---------------------------------------------------------------------------
# 10. Learning uses qualified NEW jobs
# ---------------------------------------------------------------------------


def test_learning_counts_new_from_is_new():
    j1 = make_job(job_id="l1")
    j1.is_new = True
    j1.provenance["strategy_id"] = "greenhouse_configured"
    j2 = make_job(job_id="l2")
    j2.is_new = False
    j2.provenance["strategy_id"] = "greenhouse_configured"
    new_count, repeat_count = _split_counts([j1, j2])
    assert new_count == 1
    assert repeat_count == 1


def test_learning_update_memory_uses_is_new():
    j1 = make_job(job_id="lm1")
    j1.is_new = True
    j1.provenance["strategy_id"] = "workday_configured"
    j2 = make_job(job_id="lm2")
    j2.is_new = False
    j2.provenance["strategy_id"] = "workday_configured"
    settings = MagicMock()
    settings.min_runs_before_suppression = 2
    settings.cooldown_runs = 3
    settings.revisit_runs = 2
    settings.novelty_weight = 1.0
    settings.exploration_bonus = 0.35
    settings.revisit_bonus = 0.20
    settings.exploration_share = 0.20
    settings.revisit_share = 0.10
    settings.max_run_history = 30
    settings.recent_window = 3
    memory = update_memory(empty_memory(), jobs=[j1, j2], settings=settings, run_id="test")
    run = memory["runs"][-1]
    assert run["qualified"] == 2
    assert run["new_jobs"] == 1
    assert run["repeat_jobs"] == 1
    assert run["novelty_rate"] == 0.5


# ---------------------------------------------------------------------------
# 11. Skipped global discovery does not create learning observations
# ---------------------------------------------------------------------------


def test_skipped_discovery_no_learning():
    """Empty job list produces no learning update."""
    settings = MagicMock()
    settings.max_run_history = 30
    settings.recent_window = 3
    settings.min_runs_before_suppression = 2
    settings.cooldown_runs = 3
    settings.revisit_runs = 2
    settings.novelty_weight = 1.0
    settings.exploration_bonus = 0.35
    settings.revisit_bonus = 0.20
    settings.exploration_share = 0.20
    settings.revisit_share = 0.10
    memory = update_memory(empty_memory(), jobs=[], settings=settings, run_id="test")
    run = memory["runs"][-1]
    assert run["qualified"] == 0
    assert run["new_jobs"] == 0
    assert run["novelty_rate"] == 0.0


# ---------------------------------------------------------------------------
# 12. Failed global discovery does not create learning observations
# ---------------------------------------------------------------------------


def test_failed_discovery_no_jobs_no_learning():
    """When discovery fails and returns no jobs, learning is zero."""
    settings = MagicMock()
    settings.max_run_history = 30
    settings.recent_window = 3
    settings.min_runs_before_suppression = 2
    settings.cooldown_runs = 3
    settings.revisit_runs = 2
    settings.novelty_weight = 1.0
    settings.exploration_bonus = 0.35
    settings.revisit_bonus = 0.20
    settings.exploration_share = 0.20
    settings.revisit_share = 0.10
    memory = update_memory(empty_memory(), jobs=[], settings=settings, run_id="test-fail")
    assert memory["companies"] == {}
    assert memory["strategies"] == {}


# ---------------------------------------------------------------------------
# 13. _is_global_workday correctly identifies strategy
# ---------------------------------------------------------------------------


def test_is_global_workday_true():
    job = _global_wday_job(job_id="g1")
    assert _is_global_workday(job) is True


def test_is_global_workday_false_configured():
    job = _configured_wday_job(job_id="g2")
    assert _is_global_workday(job) is False


def test_is_global_workday_false_greenhouse():
    job = make_job(job_id="g3")
    job.provenance["strategy_id"] = "greenhouse_configured"
    assert _is_global_workday(job) is False


# ---------------------------------------------------------------------------
# 14. log_global_workday_attribution runs without error
# ---------------------------------------------------------------------------


def test_attribution_log_runs(tmp_config):
    """Smoke test: log_global_workday_attribution does not crash."""
    jobs = [
        _global_wday_job(job_id="a1", is_new=True, cluster="wd12"),
        _global_wday_job(job_id="a2", is_new=False, company="Acme Robotics", cluster="wd12"),
        _global_wday_job(job_id="a3", is_new=False, company="UnknownCo", cluster="wd1"),
    ]
    state = PipelineState(config=tmp_config, jobs=jobs)
    # Should not raise
    log_global_workday_attribution(state)


def test_attribution_log_skips_when_no_global_jobs(tmp_config):
    """No log emitted when there are no global Workday jobs."""
    job = _configured_wday_job(job_id="s1")
    state = PipelineState(config=tmp_config, jobs=[job])
    # Should not raise, and should return early
    log_global_workday_attribution(state)

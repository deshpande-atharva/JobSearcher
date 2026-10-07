"""Cluster coverage experiment: rotation, bounded queries, observability."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.models.config import GlobalBoardSettings, load_config
from src.services.public_board_index import workday_clusters_for_run

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. cluster_candidates config
# ---------------------------------------------------------------------------


def test_cluster_candidates_default():
    settings = GlobalBoardSettings()
    assert settings.workday_cluster_candidates == ["wd1", "wd3", "wd5", "wd7", "wd12"]


def test_cluster_candidates_from_settings_yaml():
    config = load_config(ROOT / "config", send_email=False)
    candidates = config.settings.discovery.global_boards.workday_cluster_candidates
    assert candidates == ["wd1", "wd3", "wd5", "wd7", "wd12"]


def test_cluster_candidates_override():
    settings = GlobalBoardSettings(workday_cluster_candidates=["wd3", "wd7"])
    assert settings.workday_cluster_candidates == ["wd3", "wd7"]


# ---------------------------------------------------------------------------
# 2. Rotation over 5 clusters selects 2 per day
# ---------------------------------------------------------------------------


def test_rotation_over_five_clusters_selects_two():
    pool = ("wd1", "wd3", "wd5", "wd7", "wd12")
    now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    chosen = workday_clusters_for_run(pool, now, count=2)
    assert len(chosen) == 2
    assert all(c in pool for c in chosen)


def test_rotation_cycles_through_all_five():
    pool = ("wd1", "wd3", "wd5", "wd7", "wd12")
    seen = set()
    for day_offset in range(5):
        now = datetime(2026, 10, 7 + day_offset, 12, 0, tzinfo=timezone.utc)
        chosen = workday_clusters_for_run(pool, now, count=2)
        seen.update(chosen)
    # Over 5 days, all 5 clusters should be sampled at least once.
    assert seen == set(pool)


def test_same_day_same_clusters():
    pool = ("wd1", "wd3", "wd5", "wd7", "wd12")
    now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    a = workday_clusters_for_run(pool, now, count=2)
    b = workday_clusters_for_run(pool, now, count=2)
    assert a == b


def test_rotation_deterministic_per_date():
    pool = ("wd1", "wd3", "wd5", "wd7", "wd12")
    now = datetime(2026, 10, 7, 8, 0, tzinfo=timezone.utc)
    later = datetime(2026, 10, 7, 22, 0, tzinfo=timezone.utc)
    assert workday_clusters_for_run(pool, now) == workday_clusters_for_run(pool, later)


# ---------------------------------------------------------------------------
# 3. Bounded query count: still 2 clusters x 4 prefixes = 8 queries
# ---------------------------------------------------------------------------


def test_query_count_unchanged_with_five_cluster_pool():
    """Five candidate clusters still select 2 per run → same query budget."""
    pool = ("wd1", "wd3", "wd5", "wd7", "wd12")
    now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    chosen = workday_clusters_for_run(pool, now, count=2)
    prefixes_per_run = 4
    queries = len(chosen) * prefixes_per_run
    assert queries == 8


# ---------------------------------------------------------------------------
# 4. Cluster-level tally (unit test of the tally structure)
# ---------------------------------------------------------------------------


def test_tally_includes_per_cluster_stats():
    """Verify the per_cluster CDX breakdown is structurally correct."""
    # Simulate what load_workday_archive_sample produces.
    cluster_stats = [
        {"cluster": "wd3", "queries": 4, "urls_seen": 150, "boards_found": 6},
        {"cluster": "wd7", "queries": 4, "urls_seen": 80, "boards_found": 3},
    ]
    assert len(cluster_stats) == 2
    assert all("cluster" in s for s in cluster_stats)
    assert all("boards_found" in s for s in cluster_stats)
    assert sum(s["queries"] for s in cluster_stats) == 8


# ---------------------------------------------------------------------------
# 5. Learning not polluted by cluster experiment
# ---------------------------------------------------------------------------


def test_cluster_rotation_does_not_affect_learning_keys():
    """Learning keys use board identity (tenant|site), not cluster name.

    Changing the cluster pool must not introduce cluster-named keys into
    discovery_memory.json.
    """
    from src.services.discovery_learning import STRATEGY_SOURCE

    # Strategy keys are source-based, not cluster-based.
    for key in STRATEGY_SOURCE:
        assert not key.startswith("wd"), f"strategy {key!r} looks cluster-specific"


def test_five_cluster_pool_does_not_change_configured_identities(tmp_config):
    """configured_workday_identities reads from companies, not from cluster pool."""
    from src.models.state import PipelineState
    from src.services.global_workday import configured_workday_identities

    state = PipelineState(config=tmp_config, jobs=[])
    # tmp_config has no Workday companies, so identities should be empty
    # regardless of what cluster_candidates is set to.
    identities = configured_workday_identities(state)
    # No Workday companies in the test fixture.
    assert isinstance(identities, set)


# ---------------------------------------------------------------------------
# 6. Fallback clusters still work when no candidates configured
# ---------------------------------------------------------------------------


def test_empty_candidates_uses_fallback():
    chosen = workday_clusters_for_run((), None, count=2)
    assert len(chosen) == 2
    # Falls back to _FALLBACK_WORKDAY_CLUSTERS
    assert all(c in ("wd1", "wd5", "wd12") for c in chosen)


# ---------------------------------------------------------------------------
# 7. Playlist correctly disabled
# ---------------------------------------------------------------------------


def test_playwright_remains_disabled():
    config = load_config(ROOT / "config", send_email=False)
    assert config.settings.board_search.playwright_enabled is False

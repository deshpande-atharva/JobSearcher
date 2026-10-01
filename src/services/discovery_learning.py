"""Persistent discovery learning.

Previous qualified outcomes change the next run's company effort and the
ranking of an already-collected public-board sample. They do not change job
identity, tracker retention, or ATS request limits.

An empty store leaves discovery on today's date rotation and full company
effort. Fixture mode and dry-run never write the store.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.utils.logging import get_logger
from src.utils.normalization import normalize_company_name

log = get_logger(__name__)

MEMORY_VERSION = 1

STRATEGY_SOURCE: dict[str, str] = {
    "greenhouse_configured": "greenhouse",
    "greenhouse_global_index": "greenhouse",
    "lever_configured": "lever",
    "ashby_configured": "ashby",
    "ashby_global_index": "ashby",
    "workday_configured": "workday",
    "workday_global_index": "workday",
    "workday_partition": "workday",
    "career_page_fallback": "company_career",
}


class LearningExplanation(BaseModel):
    """Optional Gemini note. Budgets are not fields on this model."""

    model_config = ConfigDict(extra="ignore")

    reason: str = ""


@dataclass
class LoadedMemory:
    memory: dict[str, Any]
    writable: bool
    path: Path


@dataclass
class DiscoveryPlan:
    """Deterministic budgets for one run. Numeric fields never come from an LLM."""

    active: bool
    company_effort: dict[str, str] = field(default_factory=dict)
    company_names: dict[str, str] = field(default_factory=dict)
    boards: dict[str, dict[str, Any]] = field(default_factory=dict)
    strategy_priority: dict[str, float] = field(default_factory=dict)
    exploration_share: float = 0.20
    revisit_share: float = 0.10
    min_runs: int = 2
    novelty_weight: float = 1.0
    exploration_bonus: float = 0.35
    revisit_bonus: float = 0.20
    reason: str = ""
    explanation: str = ""

    def effort_for(self, company_key: str) -> str:
        if not self.active:
            return "full"
        return self.company_effort.get(company_key, "full")

    def selection_for(self, kind: str) -> tuple[dict[str, float] | None, set[str]]:
        """Scores and revisit keys for one public-board source. None keeps date rotation."""
        if not self.active:
            return None, set()
        prefix = f"{kind}:"
        scores: dict[str, float] = {}
        revisit: set[str] = set()
        for key, record in self.boards.items():
            if not str(key).startswith(prefix) or not isinstance(record, dict):
                continue
            local = str(key)[len(prefix) :].lower()
            scores[local] = _board_score(record, self)
            if record.get("revisit_due"):
                revisit.add(local)
        if not scores:
            return None, set()
        return scores, revisit


def empty_memory() -> dict[str, Any]:
    return {
        "version": MEMORY_VERSION,
        "updated_at": None,
        "companies": {},
        "sources": {},
        "strategies": {},
        "boards": {},
        "runs": [],
    }


def load_memory(path: Path) -> LoadedMemory:
    """Missing file is empty and writable. A bad file is left untouched."""
    if not path.exists():
        return LoadedMemory(empty_memory(), True, path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "learning memory unreadable; leaving the file unchanged",
            error=type(exc).__name__,
        )
        return LoadedMemory(empty_memory(), False, path)
    if not isinstance(payload, dict) or payload.get("version") != MEMORY_VERSION:
        log.warning("learning memory version is not supported; leaving the file unchanged")
        return LoadedMemory(empty_memory(), False, path)
    memory = empty_memory()
    for key in ("companies", "sources", "strategies", "boards"):
        value = payload.get(key, {})
        if not isinstance(value, dict):
            log.warning("learning memory shape is invalid; leaving the file unchanged")
            return LoadedMemory(empty_memory(), False, path)
        memory[key] = value
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        log.warning("learning memory shape is invalid; leaving the file unchanged")
        return LoadedMemory(empty_memory(), False, path)
    memory["runs"] = runs
    memory["updated_at"] = payload.get("updated_at")
    return LoadedMemory(memory, True, path)


def save_memory(path: Path, memory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def score_novelty(
    recent_novelty: float,
    *,
    novelty_weight: float,
    exploration_bonus: float = 0.0,
    revisit_bonus: float = 0.0,
    unseen: bool = False,
    revisit: bool = False,
) -> float:
    """Qualified novelty only. Raw discovered volume is not an input."""
    score = novelty_weight * float(recent_novelty)
    if unseen:
        score += exploration_bonus
    if revisit:
        score += revisit_bonus
    return score


def build_discovery_plan(memory: dict[str, Any], settings: Any) -> DiscoveryPlan:
    """Plan from stored counts. One run of history cannot move a company to monitor."""
    plan = DiscoveryPlan(
        active=True,
        exploration_share=float(settings.exploration_share),
        revisit_share=float(settings.revisit_share),
        min_runs=int(settings.min_runs_before_suppression),
        novelty_weight=float(settings.novelty_weight),
        exploration_bonus=float(settings.exploration_bonus),
        revisit_bonus=float(settings.revisit_bonus),
        boards=dict(memory.get("boards") or {}),
    )
    companies = memory.get("companies") or {}
    monitors: list[str] = []
    for key, record in companies.items():
        if not isinstance(record, dict):
            continue
        display = str(record.get("display") or key)
        plan.company_names[str(key)] = display
        if _stored_effort(record, plan.min_runs) == "monitor":
            plan.company_effort[str(key)] = "monitor"
            monitors.append(display)
    strategies = memory.get("strategies") or {}
    for name, record in strategies.items():
        if not isinstance(record, dict):
            continue
        plan.strategy_priority[str(name)] = score_novelty(
            float(record.get("recent_novelty_rate") or 0.0),
            novelty_weight=plan.novelty_weight,
        )
    if monitors:
        plan.reason = "monitor " + ", ".join(sorted(monitors))
    elif not companies and not strategies and not plan.boards:
        plan.reason = "no learning history; current discovery behavior"
    else:
        plan.reason = "no company is in monitor"
    return plan


def neutral_plan(settings: Any) -> DiscoveryPlan:
    """Fixture and disabled learning. Callers must not reorder boards from this plan."""
    return DiscoveryPlan(
        active=False,
        exploration_share=float(settings.exploration_share),
        revisit_share=float(settings.revisit_share),
        min_runs=int(settings.min_runs_before_suppression),
        novelty_weight=float(settings.novelty_weight),
        exploration_bonus=float(settings.exploration_bonus),
        revisit_bonus=float(settings.revisit_bonus),
        reason="learning inactive; current discovery behavior",
    )


def update_memory(
    memory: dict[str, Any],
    *,
    jobs: list[Any],
    settings: Any,
    run_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record qualified new and repeat jobs. Companies absent from ``jobs`` are unchanged."""
    updated = deepcopy(memory) if memory else empty_memory()
    updated["version"] = MEMORY_VERSION
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    day = moment.astimezone(UTC).strftime("%Y-%m-%d")
    updated["updated_at"] = moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    grouped = _group_jobs(jobs)
    for key, bucket in grouped.items():
        _update_company(updated, key, bucket, settings=settings, day=day)
    _update_sources(updated, jobs, settings=settings, day=day)
    _update_strategies(updated, jobs, settings=settings, day=day)
    _update_boards(updated, jobs, settings=settings, day=day)

    qualified = len(jobs)
    new_jobs = sum(1 for job in jobs if bool(getattr(job, "is_new", False)))
    repeat_jobs = qualified - new_jobs
    novelty = round(new_jobs / qualified, 4) if qualified else 0.0
    runs = list(updated.get("runs") or [])
    previous = runs[-1].get("novelty_rate") if runs and isinstance(runs[-1], dict) else None
    change = None if previous is None else round(novelty - float(previous), 4)
    runs.append(
        {
            "run_id": run_id,
            "qualified": qualified,
            "new_jobs": new_jobs,
            "repeat_jobs": repeat_jobs,
            "novelty_rate": novelty,
            "previous_novelty_rate": previous,
            "novelty_change": change,
        }
    )
    keep = max(1, int(getattr(settings, "max_run_history", 30)))
    updated["runs"] = runs[-keep:]
    return updated


def assign_strategy(posting: Any, *, origin: str = "configured") -> None:
    """Stamp the strategy that actually produced this posting. Unknown sources are left alone."""
    provenance = getattr(posting, "provenance", None)
    if not isinstance(provenance, dict):
        return
    source = str(getattr(posting, "source", "") or "")
    parts = [str(part) for part in (provenance.get("workday_partitions") or [])]
    partition_only = bool(parts) and "unfiltered" not in parts
    if origin == "global_index":
        if source == "workday" and partition_only:
            strategy = "workday_partition"
        elif source == "greenhouse":
            strategy = "greenhouse_global_index"
        elif source == "ashby":
            strategy = "ashby_global_index"
        elif source == "workday":
            strategy = "workday_global_index"
        else:
            return
    elif source == "workday" and partition_only:
        strategy = "workday_partition"
    elif source in {"greenhouse", "lever", "ashby"}:
        strategy = f"{source}_configured"
    elif source == "workday":
        strategy = "workday_configured"
    elif source == "company_career":
        strategy = "career_page_fallback"
    else:
        return
    provenance["strategy_id"] = strategy


def effort_for_company(state: Any, company_name: str) -> str:
    if getattr(state.config, "fixture_mode", False):
        return "full"
    plan = state.resources.get("discovery_plan")
    if plan is None:
        return "full"
    return plan.effort_for(normalize_company_name(company_name))


def board_selection_kwargs(state: Any, kind: str) -> dict[str, Any]:
    """Empty when learning must not change the date rotation."""
    if getattr(state.config, "fixture_mode", False):
        return {}
    plan = state.resources.get("discovery_plan")
    if plan is None or not getattr(plan, "active", False):
        return {}
    priorities, revisit = plan.selection_for(kind)
    if not priorities:
        return {}
    return {
        "priorities": priorities,
        "revisit_keys": revisit,
        "exploration_share": plan.exploration_share,
        "revisit_share": plan.revisit_share,
    }


async def prepare_learning(state: Any) -> None:
    """Load memory and build the plan before discovery. Does not write."""
    settings = state.config.settings.learning
    if not settings.enabled or state.config.fixture_mode:
        state.resources["discovery_plan"] = neutral_plan(settings)
        return
    path = state.config.path(settings.path)
    loaded = load_memory(path)
    state.resources["discovery_memory_loaded"] = loaded
    if loaded.writable:
        plan = build_discovery_plan(loaded.memory, settings)
    else:
        plan = neutral_plan(settings)
        plan.reason = "learning file unreadable; using current discovery behavior"
    state.resources["discovery_plan"] = await annotate_plan(plan, state.resources.get("llm"))


async def annotate_plan(plan: DiscoveryPlan, llm: Any) -> DiscoveryPlan:
    """Attach a short explanation. Failure leaves the deterministic plan unchanged."""
    if llm is None or not getattr(llm, "available", False):
        return plan
    priorities = dict(plan.strategy_priority)
    efforts = dict(plan.company_effort)
    try:
        note = await llm.structured(
            prompt=_explanation_prompt(plan),
            response_model=LearningExplanation,
            system=(
                "You may explain this discovery plan in one or two sentences. "
                "Do not choose companies, budgets, or caps."
            ),
            purpose="discovery_learning",
        )
    except Exception as exc:
        log.warning("learning explanation unavailable", error=type(exc).__name__)
        plan.strategy_priority = priorities
        plan.company_effort = efforts
        return plan
    plan.strategy_priority = priorities
    plan.company_effort = efforts
    text = str(getattr(note, "reason", "") or "").strip()
    if text:
        plan.explanation = text[:500]
    return plan


def persist_learning(state: Any) -> None:
    """Update memory after dedup. Fixture mode and dry-run do not write."""
    settings = state.config.settings.learning
    if not settings.enabled:
        return
    jobs = list(state.jobs)
    if state.config.fixture_mode:
        state.summary.learning_report = render_learning_report(
            _snapshot_memory(jobs, settings, state),
            neutral_plan(settings),
        )
        return
    path = state.config.path(settings.path)
    loaded = state.resources.get("discovery_memory_loaded")
    if not isinstance(loaded, LoadedMemory):
        loaded = load_memory(path)
    if state.config.dry_run or not loaded.writable:
        state.summary.learning_report = render_learning_report(
            _snapshot_memory(jobs, settings, state),
            state.resources.get("discovery_plan") or neutral_plan(settings),
        )
        if not loaded.writable and not state.config.dry_run:
            state.summary.note(
                "learning memory was not updated because the file could not be read safely"
            )
        return
    run_id = _run_id(state)
    memory = update_memory(loaded.memory, jobs=jobs, settings=settings, run_id=run_id)
    save_memory(path, memory)
    plan = build_discovery_plan(memory, settings)
    state.resources["discovery_plan"] = plan
    state.resources["discovery_memory_loaded"] = LoadedMemory(memory, True, path)
    state.summary.learning_report = render_learning_report(memory, plan)


def render_learning_report(memory: dict[str, Any], plan: DiscoveryPlan) -> str:
    runs = [item for item in (memory.get("runs") or []) if isinstance(item, dict)]
    current = runs[-1] if runs else {}
    previous = current.get("previous_novelty_rate")
    change = current.get("novelty_change")
    qualified = int(current.get("qualified") or 0)
    new_jobs = int(current.get("new_jobs") or 0)
    repeat_jobs = int(current.get("repeat_jobs") or 0)
    lines = [
        "Discovery Learning",
        f"Qualified: {qualified}",
        f"New: {new_jobs}",
        f"Repeat: {repeat_jobs}",
        f"Novelty: {_percent(current.get('novelty_rate'))}",
        f"Previous novelty: {_percent(previous)}",
        f"Change: {_points(change)}",
        "Learning:",
    ]
    monitors = [
        (plan.company_names.get(key, key), effort)
        for key, effort in sorted(plan.company_effort.items())
        if effort == "monitor"
    ]
    if monitors:
        for name, effort in monitors:
            lines.append(f"- {name}: {effort}")
    else:
        lines.append("- no companies are in monitor")
    if plan.active and plan.exploration_share > 0:
        lines.append("- unexplored boards retained in exploration budget")
    ranked = sorted(plan.strategy_priority.items(), key=lambda item: (-item[1], item[0]))
    if ranked:
        high_name, high_score = ranked[0]
        low_name, low_score = ranked[-1]
        lines.append(f"Highest strategy: {high_name} ({high_score:.2f})")
        if low_name != high_name:
            lines.append(f"Lowest strategy: {low_name} ({low_score:.2f})")
    if plan.explanation:
        lines.append(f"Note: {plan.explanation}")
    elif plan.reason:
        lines.append(f"Plan: {plan.reason}")
    return "\n".join(lines)


def _snapshot_memory(jobs: list[Any], settings: Any, state: Any) -> dict[str, Any]:
    """Report counts for a run that must not touch the stored file."""
    return update_memory(
        empty_memory(),
        jobs=jobs,
        settings=settings,
        run_id=_run_id(state),
        now=getattr(state.summary, "run_started_at", None),
    )


def _run_id(state: Any) -> str:
    when = getattr(state.summary, "run_started_at", None) or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _percent(rate: Any) -> str:
    if rate is None:
        return "n/a"
    return f"{float(rate) * 100:.1f}%"


def _points(change: Any) -> str:
    if change is None:
        return "n/a"
    return f"{float(change) * 100:+.1f} percentage points"


def _stored_effort(record: dict[str, Any], min_runs: int) -> str:
    if int(record.get("runs_seen") or 0) < min_runs:
        return "full"
    if record.get("effort") == "monitor" and int(record.get("cooldown_remaining") or 0) > 0:
        return "monitor"
    return "full"


def _board_score(record: dict[str, Any], plan: DiscoveryPlan) -> float:
    return score_novelty(
        float(record.get("recent_novelty_rate") or 0.0),
        novelty_weight=plan.novelty_weight,
        revisit_bonus=plan.revisit_bonus,
        revisit=bool(record.get("revisit_due")),
    )


def _explanation_prompt(plan: DiscoveryPlan) -> str:
    ranked = sorted(plan.strategy_priority.items(), key=lambda item: (-item[1], item[0]))
    shown = ", ".join(f"{name}={score:.2f}" for name, score in ranked[:8]) or "none"
    names = (plan.company_names.get(key, key) for key in plan.company_effort)
    monitors = ", ".join(sorted(names)) or "none"
    return (
        "Discovery plan (already decided):\n"
        f"monitor companies: {monitors}\n"
        f"strategy priority: {shown}\n"
        f"exploration share: {plan.exploration_share}\n"
        f"revisit share: {plan.revisit_share}\n"
        "Explain the plan. Do not propose different budgets."
    )


def _group_jobs(jobs: list[Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for job in jobs:
        key = _company_key(job)
        if not key:
            continue
        bucket = grouped.setdefault(
            key,
            {
                "display": str(getattr(job, "company", "") or key),
                "jobs": [],
            },
        )
        bucket["jobs"].append(job)
    return grouped


def _company_key(job: Any) -> str:
    key = str(getattr(job, "company_key", "") or "")
    if key:
        return key
    return normalize_company_name(str(getattr(job, "company", "") or ""))


def _job_identity(job: Any) -> str:
    job_id = getattr(job, "job_id", None)
    if job_id:
        return str(job_id).strip().lower()
    dedup = getattr(job, "dedup_key", "")
    return str(dedup or "")


def _update_company(
    memory: dict[str, Any],
    key: str,
    bucket: dict[str, Any],
    *,
    settings: Any,
    day: str,
) -> None:
    companies = memory.setdefault("companies", {})
    record = dict(companies.get(key) or _blank_company(bucket["display"]))
    if record.get("revisit_due") and record.get("effort") != "monitor":
        record["revisit_due"] = False
    known = set(record.get("known_job_ids") or [])
    new_count = 0
    repeat_count = 0
    sources: list[str] = list(record.get("sources") or [])
    for job in bucket["jobs"]:
        source = str(getattr(job, "source", "") or "")
        if source and source not in sources:
            sources.append(source)
        identity = _job_identity(job)
        if not identity:
            continue
        if bool(getattr(job, "is_new", False)) and identity not in known:
            known.add(identity)
            record["unique_jobs"] = int(record.get("unique_jobs") or 0) + 1
            record["new_jobs"] = int(record.get("new_jobs") or 0) + 1
            new_count += 1
        else:
            if identity not in known:
                known.add(identity)
                record["unique_jobs"] = int(record.get("unique_jobs") or 0) + 1
            record["repeat_jobs"] = int(record.get("repeat_jobs") or 0) + 1
            repeat_count += 1
    record["known_job_ids"] = sorted(known)
    record["sources"] = sources
    record["display"] = bucket["display"] or record.get("display") or key
    record["runs_seen"] = int(record.get("runs_seen") or 0) + 1
    record["last_seen"] = day
    if new_count:
        record["last_new_job"] = day
        record["consecutive_zero_new"] = 0
        record["effort"] = "full"
        record["cooldown_remaining"] = 0
        record["revisit_due"] = False
    else:
        record["consecutive_zero_new"] = int(record.get("consecutive_zero_new") or 0) + 1
        _apply_cooldown(record, settings)
    _set_recent(record, new_count, new_count + repeat_count, settings)
    companies[key] = record


def _blank_company(display: str) -> dict[str, Any]:
    return {
        "display": display,
        "sources": [],
        "runs_seen": 0,
        "unique_jobs": 0,
        "new_jobs": 0,
        "repeat_jobs": 0,
        "consecutive_zero_new": 0,
        "last_seen": None,
        "last_new_job": None,
        "recent_novelty_rate": 0.0,
        "effort": "full",
        "cooldown_remaining": 0,
        "revisit_due": False,
        "known_job_ids": [],
        "recent": [],
    }


def _apply_cooldown(record: dict[str, Any], settings: Any) -> None:
    runs = int(record.get("runs_seen") or 0)
    zeros = int(record.get("consecutive_zero_new") or 0)
    min_runs = int(settings.min_runs_before_suppression)
    cooldown = int(settings.cooldown_runs)
    ready = runs >= min_runs and zeros >= cooldown
    if not ready:
        if record.get("effort") == "monitor":
            _decrement_monitor(record)
        else:
            record["effort"] = "full"
            record["cooldown_remaining"] = 0
        return
    if record.get("effort") != "monitor":
        record["effort"] = "monitor"
        record["cooldown_remaining"] = int(settings.revisit_runs)
        record["revisit_due"] = False
        return
    _decrement_monitor(record)


def _decrement_monitor(record: dict[str, Any]) -> None:
    remaining = int(record.get("cooldown_remaining") or 0) - 1
    if remaining <= 0:
        record["effort"] = "full"
        record["cooldown_remaining"] = 0
        record["revisit_due"] = True
    else:
        record["effort"] = "monitor"
        record["cooldown_remaining"] = remaining
        record["revisit_due"] = False


def _set_recent(record: dict[str, Any], new_count: int, qualified: int, settings: Any) -> None:
    recent = list(record.get("recent") or [])
    recent.append({"new": new_count, "qualified": qualified})
    window = max(1, int(getattr(settings, "recent_window", 3)))
    record["recent"] = recent[-window:]
    total = sum(int(item.get("qualified") or 0) for item in record["recent"])
    fresh = sum(int(item.get("new") or 0) for item in record["recent"])
    record["recent_novelty_rate"] = round(fresh / total, 4) if total else 0.0


def _update_sources(memory: dict[str, Any], jobs: list[Any], *, settings: Any, day: str) -> None:
    by_source: dict[str, list[Any]] = {}
    for job in jobs:
        source = str(getattr(job, "source", "") or "")
        if source:
            by_source.setdefault(source, []).append(job)
    sources = memory.setdefault("sources", {})
    for source, grouped in by_source.items():
        record = dict(sources.get(source) or _blank_source())
        new_count, repeat_count = _split_counts(grouped)
        record["runs"] = int(record.get("runs") or 0) + 1
        record["qualified"] = int(record.get("qualified") or 0) + new_count + repeat_count
        record["new_jobs"] = int(record.get("new_jobs") or 0) + new_count
        record["repeat_jobs"] = int(record.get("repeat_jobs") or 0) + repeat_count
        keys = set(record.get("company_keys") or [])
        for job in grouped:
            company_key = _company_key(job)
            if company_key and company_key not in keys:
                keys.add(company_key)
                if bool(getattr(job, "is_new", False)):
                    record["new_companies"] = int(record.get("new_companies") or 0) + 1
        record["company_keys"] = sorted(keys)
        record["unique_companies"] = len(keys)
        record["last_success"] = day
        _set_recent(record, new_count, new_count + repeat_count, settings)
        sources[source] = record


def _blank_source() -> dict[str, Any]:
    return {
        "runs": 0,
        "discovered": 0,
        "qualified": 0,
        "new_jobs": 0,
        "repeat_jobs": 0,
        "unique_companies": 0,
        "new_companies": 0,
        "last_success": None,
        "recent_novelty_rate": 0.0,
        "company_keys": [],
        "recent": [],
    }


def _update_strategies(memory: dict[str, Any], jobs: list[Any], *, settings: Any, day: str) -> None:
    by_strategy: dict[str, list[Any]] = {}
    for job in jobs:
        provenance = getattr(job, "provenance", None) or {}
        strategy = str(provenance.get("strategy_id") or "") if isinstance(provenance, dict) else ""
        if strategy in STRATEGY_SOURCE:
            by_strategy.setdefault(strategy, []).append(job)
    strategies = memory.setdefault("strategies", {})
    for strategy, grouped in by_strategy.items():
        record = dict(strategies.get(strategy) or _blank_strategy(strategy))
        new_count, repeat_count = _split_counts(grouped)
        record["runs"] = int(record.get("runs") or 0) + 1
        record["qualified"] = int(record.get("qualified") or 0) + new_count + repeat_count
        record["new_jobs"] = int(record.get("new_jobs") or 0) + new_count
        record["repeat_jobs"] = int(record.get("repeat_jobs") or 0) + repeat_count
        keys = set(record.get("company_keys") or [])
        for job in grouped:
            company_key = _company_key(job)
            if company_key and company_key not in keys:
                keys.add(company_key)
                if bool(getattr(job, "is_new", False)):
                    record["new_companies"] = int(record.get("new_companies") or 0) + 1
        record["company_keys"] = sorted(keys)
        record["companies"] = len(keys)
        record["last_used"] = day
        _set_recent(record, new_count, new_count + repeat_count, settings)
        strategies[strategy] = record


def _blank_strategy(strategy: str) -> dict[str, Any]:
    return {
        "source": STRATEGY_SOURCE.get(strategy, ""),
        "runs": 0,
        "discovered": 0,
        "qualified": 0,
        "new_jobs": 0,
        "repeat_jobs": 0,
        "companies": 0,
        "new_companies": 0,
        "last_used": None,
        "recent_novelty_rate": 0.0,
        "company_keys": [],
        "recent": [],
    }


def _update_boards(memory: dict[str, Any], jobs: list[Any], *, settings: Any, day: str) -> None:
    by_board: dict[str, list[Any]] = {}
    for job in jobs:
        key = _board_key(job)
        if key:
            by_board.setdefault(key, []).append(job)
    boards = memory.setdefault("boards", {})
    for key, grouped in by_board.items():
        record = dict(boards.get(key) or _blank_company(key))
        if record.get("revisit_due") and record.get("effort") != "monitor":
            record["revisit_due"] = False
        new_count, repeat_count = _split_counts(grouped)
        record["runs_seen"] = int(record.get("runs_seen") or 0) + 1
        record["last_seen"] = day
        record["display"] = key
        if new_count:
            record["consecutive_zero_new"] = 0
            record["effort"] = "full"
            record["cooldown_remaining"] = 0
            record["revisit_due"] = False
            record["last_new_job"] = day
        else:
            record["consecutive_zero_new"] = int(record.get("consecutive_zero_new") or 0) + 1
            _apply_cooldown(record, settings)
        _set_recent(record, new_count, new_count + repeat_count, settings)
        boards[key] = record


def _board_key(job: Any) -> str | None:
    provenance = getattr(job, "provenance", None) or {}
    if not isinstance(provenance, dict) or provenance.get("board_origin") != "global_index":
        return None
    source = str(getattr(job, "source", "") or "")
    token = provenance.get("board_token")
    if token and source in {"greenhouse", "ashby"}:
        return f"{source}:{str(token).lower()}"
    identity = provenance.get("board_identity")
    if identity and source == "workday":
        return f"workday:{str(identity).lower()}"
    return None


def _split_counts(jobs: list[Any]) -> tuple[int, int]:
    new_count = sum(1 for job in jobs if bool(getattr(job, "is_new", False)))
    return new_count, len(jobs) - new_count

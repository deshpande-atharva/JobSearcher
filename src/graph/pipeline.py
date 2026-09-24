"""LangGraph orchestration for the daily job pipeline.

Preferred order (H-1B runs *after* eligibility filters, and never filters):

    Discovery → Extraction → Role → Seniority → Location → Freshness
    → Direct URL → H-1B enrichment → Dedup → QC → XLSX / email
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict

from src.agents.dedup_agent import run_dedup
from src.agents.discovery_agent import run_discovery
from src.agents.extraction_agent import run_extraction
from src.agents.freshness_agent import run_freshness
from src.agents.h1b_sponsorship_agent import run_h1b_enrichment
from src.agents.location_agent import run_location_employment
from src.agents.output_agent import run_output
from src.agents.qc_agent import run_quality_control
from src.agents.role_agent import run_role_classification
from src.agents.seniority_agent import run_seniority
from src.agents.url_agent import run_url_verification
from src.llm.base import LLMProvider, build_llm_provider
from src.models.config import AppConfig
from src.models.state import PipelineState
from src.services.history import load_history
from src.sources.base import HttpClient
from src.sources.h1bgrader import H1BGraderClient
from src.utils.dates import utcnow
from src.utils.logging import get_logger

log = get_logger(__name__)

AgentFn = Callable[[PipelineState], Awaitable[None]]


class GraphState(TypedDict):
    pipeline: PipelineState


def _node(fn: AgentFn) -> Callable[[GraphState], Awaitable[GraphState]]:
    async def wrapped(state: GraphState) -> GraphState:
        pipeline = state["pipeline"]
        log.info("agent start", agent=fn.__name__)
        await fn(pipeline)
        return {"pipeline": pipeline}

    wrapped.__name__ = fn.__name__
    return wrapped


def build_graph() -> Any:
    """Compile the LangGraph state machine."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("discovery", _node(run_discovery))
    graph.add_node("extraction", _node(run_extraction))
    graph.add_node("role", _node(run_role_classification))
    graph.add_node("seniority", _node(run_seniority))
    graph.add_node("location", _node(run_location_employment))
    graph.add_node("freshness", _node(run_freshness))
    graph.add_node("url", _node(run_url_verification))
    graph.add_node("h1b", _node(run_h1b_enrichment))
    graph.add_node("dedup", _node(run_dedup))
    graph.add_node("qc", _node(run_quality_control))
    graph.add_node("output", _node(run_output))

    graph.add_edge(START, "discovery")
    graph.add_edge("discovery", "extraction")
    graph.add_edge("extraction", "role")
    graph.add_edge("role", "seniority")
    graph.add_edge("seniority", "location")
    graph.add_edge("location", "freshness")
    graph.add_edge("freshness", "url")
    graph.add_edge("url", "h1b")
    graph.add_edge("h1b", "dedup")
    graph.add_edge("dedup", "qc")
    graph.add_edge("qc", "output")
    graph.add_edge("output", END)
    return graph.compile()


async def run_pipeline(config: AppConfig, *, resources: dict[str, Any] | None = None) -> PipelineState:
    """Execute a full pipeline run with injected (or freshly built) resources."""
    history = load_history(config)
    state = PipelineState(
        config=config,
        known_keys=set(history.known_keys),
        preserved_tracking=dict(history.tracking),
        summary=state_summary(config),
    )
    state.resources["existing_rows"] = history.existing_rows

    owns_http = False
    owns_llm = False
    injected = resources or {}
    http = injected.get("http")
    llm = injected.get("llm")
    if http is None:
        http = HttpClient(config)
        owns_http = True
    if llm is None:
        llm = build_llm_provider(config)
        owns_llm = True

    state.resources["http"] = http
    state.resources["llm"] = llm
    state.resources["h1b"] = injected.get("h1b") or H1BGraderClient(config, http=http)
    state.summary.llm_enabled = bool(getattr(llm, "available", False))

    try:
        compiled = build_graph()
        result = await compiled.ainvoke({"pipeline": state})
        finished: PipelineState = result["pipeline"]
    finally:
        if owns_llm:
            close = getattr(llm, "aclose", None)
            if callable(close):
                await close()
        if owns_http:
            await http.aclose()

    if finished.summary.run_finished_at is None:
        finished.summary.run_finished_at = utcnow()
    if isinstance(llm, LLMProvider):
        finished.summary.llm_calls = llm.stats.calls
        finished.summary.llm_failures = llm.stats.failures
    from src.services.discovery_report import attach_discovery_reports

    finished.summary.freshness_hours_used = finished.config.freshness_hours
    if not finished.summary.jobs_processed:
        finished.summary.jobs_processed = (
            finished.summary.jobs_after_cross_source_dedup or finished.summary.jobs_discovered
        )
    attach_discovery_reports(finished)
    return finished


def state_summary(config: AppConfig):
    from src.models.state import RunSummary

    return RunSummary(
        dry_run=config.dry_run,
        fixture_mode=config.fixture_mode,
        llm_enabled=config.llm_enabled,
    )

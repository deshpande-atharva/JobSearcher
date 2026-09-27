"""CLI entry point for the daily job discovery pipeline.

Examples::

    python -m src.main
    python -m src.main --dry-run
    python -m src.main --fixture-mode
    python -m src.main --company "Amazon"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.models.config import ConfigError, load_config
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-agent",
        description="Daily AI-powered U.S. software-engineering job discovery and tracker.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run the pipeline but do not write XLSX.")
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Use local fixtures only; do not access live websites.",
    )
    parser.add_argument("--company", metavar="NAME", help="Limit discovery to one configured company.")
    parser.add_argument("--config-dir", default="config", help="Path to the config/ directory.")
    parser.add_argument(
        "--freshness-hours",
        type=float,
        default=None,
        help="Diagnostic override for the freshness window. Production default remains 24.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print discovery health and filter funnel. Does not send email.",
    )
    parser.add_argument(
        "--source-health",
        action="store_true",
        help="Probe configured discovery sources and exit without running the pipeline.",
    )
    parser.add_argument("--no-email", action="store_true", help="Skip the notification email.")
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send email even during --dry-run (default is to skip).",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR.")
    parser.add_argument("--log-format", choices=("text", "json"), default=None)
    parser.add_argument(
        "--greenhouse-smoke-test",
        action="store_true",
        help="Live Greenhouse check only. Not part of pytest or the daily pipeline.",
    )
    parser.add_argument(
        "--greenhouse-browser-smoke-test",
        action="store_true",
        help="Open a public Greenhouse career page in Playwright and discover jobs from the UI.",
    )
    parser.add_argument(
        "--greenhouse-phase3",
        action="store_true",
        help="Score browser-discovered Greenhouse jobs against the resume profile.",
    )
    parser.add_argument(
        "--resume-smoke-test",
        action="store_true",
        help="Read data/candidate/resume.pdf and write a versioned candidate profile.",
    )
    parser.add_argument(
        "--workday-browser-smoke-test",
        action="store_true",
        help="Open one public Workday career site in Playwright and discover jobs from the UI.",
    )
    parser.add_argument(
        "--workday-intelligence-smoke-test",
        action="store_true",
        help="Open a few public Workday job pages and score them with the existing intelligence agents.",
    )
    parser.add_argument(
        "--workday-production-smoke-test",
        action="store_true",
        help="Target-oriented Workday discovery through the existing qualification path. Does not write the production workbook.",
    )
    parser.add_argument(
        "--workday-live-e2e-smoke-test",
        action="store_true",
        help="Replay NVIDIA keyword search, then run surviving jobs through the existing qualification chain. Does not write the production workbook.",
    )
    parser.add_argument(
        "--multi-source-smoke-test",
        action="store_true",
        help="Run Greenhouse and Workday through one qualification path. Does not write the production workbook or enable browser discovery.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated sources for the multi-source smokes. Default: greenhouse,workday.",
    )
    parser.add_argument(
        "--lever-smoke-test",
        action="store_true",
        help="Public Lever board smoke. Dry-run. Does not write the production workbook.",
    )
    parser.add_argument(
        "--ashby-smoke-test",
        action="store_true",
        help="Public Ashby board smoke. Dry-run. Does not write the production workbook.",
    )
    parser.add_argument(
        "--jobright-smoke-test",
        action="store_true",
        help="Public Jobright discovery smoke. Dry-run. Does not write the production workbook.",
    )
    parser.add_argument(
        "--multi-source-browser-smoke-test",
        action="store_true",
        help="Greenhouse API plus NVIDIA Workday browser discovery. Does not change production config or navigation files.",
    )
    parser.add_argument(
        "--live-downstream-smoke-test",
        action="store_true",
        help="Score one real Workday job downstream. A stale job stays unqualified for production.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source for --live-downstream-smoke-test. Default: workday. Ignored by the daily pipeline.",
    )
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(".env"))
    configure_logging(level=args.log_level, fmt=args.log_format)

    overrides: dict = {}
    if args.freshness_hours is not None:
        overrides["run"] = {"freshness_hours": args.freshness_hours}

    diagnostic = bool(args.diagnostic)
    send_email = False if (args.no_email or diagnostic) else (True if args.email else not args.dry_run)

    try:
        config = load_config(
            args.config_dir,
            dry_run=args.dry_run,
            fixture_mode=args.fixture_mode,
            send_email=send_email,
            company_filter=args.company,
            diagnostic=diagnostic,
            overrides=overrides or None,
        )
    except ConfigError as exc:
        log.error("configuration error", error=str(exc))
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.source_health:
        from src.services.source_health import run_source_health

        report = await run_source_health(config)
        print()
        print(report)
        print()
        return 0

    if args.greenhouse_smoke_test:
        from src.pilot.smoke import run_greenhouse_smoke

        return await run_greenhouse_smoke(config)

    if args.greenhouse_browser_smoke_test:
        from src.pilot.browser_discovery import run_browser_smoke

        return await run_browser_smoke(config)

    if args.resume_smoke_test:
        from src.pilot.resume_smoke import run_resume_smoke

        return await run_resume_smoke(config)

    if args.greenhouse_phase3:
        from src.pilot.phase3 import run_phase3

        return await run_phase3(config)

    if args.workday_browser_smoke_test:
        from src.pilot.workday_browser import run_workday_browser_smoke

        return await run_workday_browser_smoke(config)

    if args.workday_intelligence_smoke_test:
        from src.pilot.workday_intelligence import run_workday_intelligence_smoke

        return await run_workday_intelligence_smoke(config)

    if args.workday_production_smoke_test:
        from src.pilot.workday_production import run_workday_production_smoke

        return await run_workday_production_smoke(config)

    if args.workday_live_e2e_smoke_test:
        from src.pilot.workday_production import run_workday_live_e2e

        return await run_workday_live_e2e(config)

    if args.lever_smoke_test:
        from src.services.discovery_orchestrator import run_lever_smoke

        return await run_lever_smoke(config)

    if args.ashby_smoke_test:
        from src.services.discovery_orchestrator import run_ashby_smoke

        return await run_ashby_smoke(config)

    if args.jobright_smoke_test:
        from src.services.discovery_orchestrator import run_jobright_smoke

        return await run_jobright_smoke(config)

    if args.multi_source_smoke_test:
        from src.services.discovery_orchestrator import ORCHESTRATED_SOURCES, run_multi_source_smoke

        selected = tuple(
            part.strip().lower()
            for part in (args.sources or "greenhouse,workday").split(",")
            if part.strip()
        ) or ORCHESTRATED_SOURCES
        return await run_multi_source_smoke(config, sources=selected)

    if args.multi_source_browser_smoke_test:
        from src.services.orchestration_smoke import run_multi_source_browser_smoke

        selected = tuple(
            part.strip().lower()
            for part in (args.sources or "greenhouse,workday").split(",")
            if part.strip()
        ) or ("greenhouse", "workday")
        return await run_multi_source_browser_smoke(config, sources=selected)

    if args.live_downstream_smoke_test:
        from src.services.orchestration_smoke import run_live_downstream_smoke

        return await run_live_downstream_smoke(config, source=(args.source or "workday").strip().lower())

    from src.graph.pipeline import run_pipeline

    try:
        state = await run_pipeline(config)
    except Exception:
        log.exception("pipeline crashed")
        return 1

    print()
    print(state.summary.render())
    from src.services.production_report import render_pipeline_health

    print()
    print(render_pipeline_health(state))
    if config.company_filter:
        print()
        print(_company_diagnostic(state, config.company_filter))
    elif config.diagnostic:
        from src.services.discovery_report import render_company_detail

        print()
        print("PER-COMPANY SOURCE RESULTS")
        print("==========================")
        for row in state.summary.company_discovery_rows:
            print(render_company_detail(row))
            print()
    print()
    return 0


def _company_diagnostic(state, name: str) -> str:
    from src.utils.normalization import normalize_company_name

    target = normalize_company_name(name)
    lines = [
        f"COMPANY DIAGNOSTIC: {name}",
        "==========================",
    ]
    company = state.config.universe.find(name)
    if company:
        lines.append(f"Enabled              : {company.enabled}")
        lines.append(f"Careers URL          : {company.careers_url or '-'}")
        lines.append(f"ATS (configured)     : {company.ats_type or 'auto'} / {company.ats_identifier or '-'}")
        lines.append(f"ATS discovery mode   : {company.ats_discovery_mode}")
        lines.append(f"Discovery sources    : {', '.join(company.discovery.sources)}")
    outcomes = [o for o in state.company_outcomes if normalize_company_name(o.company) == target]
    ats_used = [o.source for o in outcomes if o.source in {"greenhouse", "lever", "ashby", "smartrecruiters", "workday", "icims"}]
    if ats_used:
        lines.append(f"ATS (this run)       : {', '.join(ats_used)}")
    row = next((r for r in state.summary.company_discovery_rows if normalize_company_name(r.company) == target), None)
    if row:
        lines.append(f"ATS detected         : {row.ats} / {row.identifier}")
        lines.append(f"Detection method     : {row.method}")
        lines.append(f"Primary source       : {row.source}")
        lines.append(f"ATS source result    : {row.ats_result}")
        lines.append(f"Career page result   : {row.career_result}")
        lines.append(f"Fallback used        : {'yes' if row.fallback_used else 'no'}")
        lines.append(f"Raw / SWE / 0-2yr / Fresh / Final : {row.raw} / {row.swe} / {row.exp} / {row.fresh} / {row.final}")
        if row.failure:
            lines.append(f"Failure              : {row.failure}")
    if not outcomes:
        lines.append("No source outcomes recorded for this company.")
        return "\n".join(lines)
    for outcome in outcomes:
        status = "OK" if outcome.succeeded else "FAILED"
        extra = f" ({outcome.error})" if outcome.error else ""
        lines.append(
            f"{outcome.source:20} {status:8} jobs={outcome.jobs_found}{extra}"
        )
    accepted = [j for j in state.jobs if normalize_company_name(j.company) == target]
    rejected = [r for r in state.rejected if normalize_company_name(r.company) == target]
    lines.append(f"Jobs accepted        : {len(accepted)}")
    lines.append(f"Jobs rejected        : {len(rejected)}")
    if accepted:
        lines.append("H-1B on accepted jobs:")
        for job in accepted[:15]:
            lines.append(f"  - {job.job_title}: {job.sponsorship_display}")
    return "\n".join(lines)


def cli() -> None:
    raise SystemExit(asyncio.run(async_main()))


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

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

    from src.graph.pipeline import run_pipeline

    try:
        state = await run_pipeline(config)
    except Exception:
        log.exception("pipeline crashed")
        return 1

    print()
    print(state.summary.render())
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

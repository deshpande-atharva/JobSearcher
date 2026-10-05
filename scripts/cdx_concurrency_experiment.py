"""CDX Concurrency Experiment.

Measures Internet Archive CDX request reliability at concurrency levels 1, 2, 4, 8.

Uses the exact same CDX query construction and response parsing as production code.
Does NOT touch production code or learning semantics.

Run:
  uv run python scripts/cdx_concurrency_experiment.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from src.services.public_board_index import (
    parse_cdx_workday,
    prefixes_for_day,
    workday_archive_query_url,
    workday_clusters_for_run,
)

# ---------------------------------------------------------------------------
# Configuration — matches production settings exactly
# ---------------------------------------------------------------------------

CDX_TIMEOUT_S: float = 12.0          # same as _cdx_body timeout
PREFIXES_PER_TRIAL: int = 2           # 2 clusters × 2 prefixes = 4 queries/trial
CDX_LIMIT: int = 40                   # same as settings.max_index_urls
TRIALS_PER_LEVEL: int = 3
INTER_TRIAL_DELAY_S: float = 10.0     # gap between trials within a level
INTER_LEVEL_DELAY_S: float = 20.0     # gap between concurrency levels
CONCURRENCY_LEVELS = (1, 2, 4, 8)
# Clusters extracted from the actual companies.yaml (wd5, wd12, wd1)
CONFIGURED_CLUSTERS = ("wd5", "wd12", "wd1")

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    outcome: str          # "success" | "timeout" | "http_error" | "error"
    http_status: int | None = None
    elapsed_s: float = 0.0
    response_bytes: int = 0
    cdx_rows: int = 0     # rows that had a valid URL (regardless of board validity)
    boards: int = 0       # unique valid Workday career board URLs
    error_type: str = ""


# ---------------------------------------------------------------------------
# Single CDX request
# ---------------------------------------------------------------------------

async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.get(url, headers={"Accept": "application/json"}),
                timeout=CDX_TIMEOUT_S,
            )
            elapsed = time.monotonic() - t0
        except asyncio.TimeoutError:
            return RequestResult(
                outcome="timeout",
                elapsed_s=time.monotonic() - t0,
            )
        except OSError as exc:
            return RequestResult(
                outcome="error",
                elapsed_s=time.monotonic() - t0,
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            return RequestResult(
                outcome="error",
                elapsed_s=time.monotonic() - t0,
                error_type=type(exc).__name__,
            )

        if resp.status_code != 200:
            return RequestResult(
                outcome="http_error",
                http_status=resp.status_code,
                elapsed_s=elapsed,
                response_bytes=len(resp.text),
            )

        body = resp.text
        career_urls, rows_seen = parse_cdx_workday(body)
        return RequestResult(
            outcome="success",
            http_status=200,
            elapsed_s=elapsed,
            response_bytes=len(body),
            cdx_rows=rows_seen,
            boards=len(career_urls),
        )


# ---------------------------------------------------------------------------
# One trial: fire `len(queries)` requests at the given concurrency
# ---------------------------------------------------------------------------

async def run_trial(
    queries: list[str],
    concurrency: int,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    async with httpx.AsyncClient(
        timeout=CDX_TIMEOUT_S + 2,
        follow_redirects=True,
    ) as client:
        results = await asyncio.gather(
            *[_one_request(client, url, semaphore) for url in queries]
        )
    total_elapsed = time.monotonic() - t0
    return list(results), total_elapsed


# ---------------------------------------------------------------------------
# Build the exact same queries as production code would
# ---------------------------------------------------------------------------

def build_queries(now: datetime) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    chosen = workday_clusters_for_run(CONFIGURED_CLUSTERS, now, count=2)
    prefixes = prefixes_for_day(now, PREFIXES_PER_TRIAL)
    urls = []
    for cluster in chosen:
        for prefix in prefixes:
            urls.append(
                workday_archive_query_url(
                    domain="myworkdayjobs.com",
                    cluster=cluster,
                    prefix=prefix,
                    limit=CDX_LIMIT,
                )
            )
    return urls, chosen, prefixes


# ---------------------------------------------------------------------------
# Per-level aggregated stats
# ---------------------------------------------------------------------------

@dataclass
class LevelStats:
    concurrency: int
    attempts: int = 0
    successes: int = 0
    timeouts: int = 0
    http_errors: int = 0
    other_errors: int = 0
    boards: int = 0
    success_latencies: list[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float | None:
        if not self.success_latencies:
            return None
        return sum(self.success_latencies) / len(self.success_latencies)

    def add(self, result: RequestResult) -> None:
        self.attempts += 1
        if result.outcome == "success":
            self.successes += 1
            self.boards += result.boards
            self.success_latencies.append(result.elapsed_s)
        elif result.outcome == "timeout":
            self.timeouts += 1
        elif result.outcome == "http_error":
            self.http_errors += 1
        else:
            self.other_errors += 1


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def main() -> int:
    queries, chosen_clusters, chosen_prefixes = build_queries(NOW)

    print("=" * 70)
    print("CDX CONCURRENCY EXPERIMENT")
    print(f"  Date/time :  {NOW.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Clusters  :  {chosen_clusters}")
    print(f"  Prefixes  :  {chosen_prefixes}")
    print(f"  Queries/trial: {len(queries)}")
    print(f"  Timeout   :  {CDX_TIMEOUT_S}s (production value)")
    print(f"  Trials/level: {TRIALS_PER_LEVEL}")
    print(f"  Domain    :  myworkdayjobs.com only (myworkdaysite.com excluded)")
    print("=" * 70)
    print()

    level_stats: list[LevelStats] = []

    for lvl_idx, concurrency in enumerate(CONCURRENCY_LEVELS):
        stats = LevelStats(concurrency=concurrency)
        print(f"[ CONCURRENCY = {concurrency} ]")

        for trial in range(TRIALS_PER_LEVEL):
            if trial > 0:
                print(f"  waiting {INTER_TRIAL_DELAY_S:.0f}s before next trial...")
                await asyncio.sleep(INTER_TRIAL_DELAY_S)

            results, elapsed = await run_trial(queries, concurrency)
            for r in results:
                stats.add(r)

            outcomes = [r.outcome for r in results]
            detail_parts = []
            for r in results:
                if r.outcome == "success":
                    detail_parts.append(f"200/{r.elapsed_s:.1f}s/boards={r.boards}")
                elif r.outcome == "http_error":
                    detail_parts.append(f"HTTP{r.http_status}/{r.elapsed_s:.1f}s")
                elif r.outcome == "timeout":
                    detail_parts.append(f"TIMEOUT/{r.elapsed_s:.0f}s")
                else:
                    detail_parts.append(f"ERR({r.error_type})/{r.elapsed_s:.1f}s")

            success_count = outcomes.count("success")
            timeout_count = outcomes.count("timeout")
            http_err_count = outcomes.count("http_error")
            print(
                f"  trial {trial + 1}/{TRIALS_PER_LEVEL}: "
                f"elapsed={elapsed:.1f}s  "
                f"ok={success_count}  timeout={timeout_count}  http_err={http_err_count}"
            )
            for part in detail_parts:
                print(f"    {part}")

        level_stats.append(stats)
        print()

        if lvl_idx < len(CONCURRENCY_LEVELS) - 1:
            print(f"  waiting {INTER_LEVEL_DELAY_S:.0f}s before next concurrency level...")
            await asyncio.sleep(INTER_LEVEL_DELAY_S)
            print()

    # Print results table
    print()
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    header = (
        f"{'Conc':>4} | {'Attempts':>8} | {'Success':>7} | "
        f"{'Timeout':>7} | {'HTTP Err':>8} | {'Other':>5} | "
        f"{'Boards':>6} | {'Avg Lat':>8}"
    )
    print(header)
    print("-" * 70)
    for s in level_stats:
        lat = f"{s.avg_latency:.2f}s" if s.avg_latency is not None else "N/A"
        print(
            f"{s.concurrency:>4} | {s.attempts:>8} | {s.successes:>7} | "
            f"{s.timeouts:>7} | {s.http_errors:>8} | {s.other_errors:>5} | "
            f"{s.boards:>6} | {lat:>8}"
        )
    print("-" * 70)
    print()

    # Interpretation
    print("INTERPRETATION:")
    for s in level_stats:
        success_rate = s.successes / s.attempts if s.attempts else 0
        label = "RELIABLE" if success_rate >= 0.75 else ("DEGRADED" if success_rate >= 0.25 else "FAILED")
        print(f"  concurrency={s.concurrency}: {success_rate*100:.0f}% success  [{label}]")
    print()

    # Return 0 always — experiment result is informational
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

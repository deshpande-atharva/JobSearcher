"""Lightweight live probes for configured discovery sources.

Never fails the process because one source is down. This is a diagnostic, not
a pipeline run, and it does not bypass robots, CAPTCHAs or authentication.
"""

from __future__ import annotations

from src.models.config import AppConfig, CompanyConfig
from src.sources.base import HttpClient, classify_source_status
from src.sources.workday import CXS_PAGE_SIZE, parse_workday_site

__all__ = ["run_source_health"]

_PROBES: tuple[tuple[str, str], ...] = (
    ("greenhouse", "https://boards-api.greenhouse.io/v1/boards/robinhood/jobs"),
    ("lever", "https://api.lever.co/v0/postings/palantir?mode=json"),
    ("ashby", "https://api.ashbyhq.com/posting-api/job-board/notion"),
    ("smartrecruiters", "https://api.smartrecruiters.com/v1/companies/smartrecruiters/postings?limit=1"),
    ("jobright", "https://jobright.ai/entry-level-jobs"),
    ("h1bgrader", "https://h1bgrader.com/h1b-sponsors"),
)


async def run_source_health(config: AppConfig) -> str:
    if config.fixture_mode:
        return "SOURCE HEALTH\n=============\n(skipped: fixture-mode does not probe live sites)"

    lines = ["SOURCE HEALTH", "============="]
    async with HttpClient(config) as http:
        for name, url in _PROBES:
            status = await _probe(http, url)
            lines.append(f"{name:18} {status}")

        lines += ["", "WORKDAY CXS", "==========="]
        workday_companies = [
            company
            for company in config.universe.enabled_companies()
            if company.ats_type == "workday" or _looks_workday(company)
        ]
        if not workday_companies:
            lines.append("(no enabled Workday companies)")
        for company in workday_companies:
            lines.append(await _probe_workday(http, company))

    lines.append("")
    lines.append("OK = reachable public endpoint. EMPTY = reachable, 0 jobs.")
    lines.append("ERROR = HTTP/transport failure (not the same as empty).")
    lines.append("BLOCKED = denied/robots/CAPTCHA. UNSUPPORTED = identifier cannot be used.")
    return "\n".join(lines)


def _looks_workday(company: CompanyConfig) -> bool:
    ident = company.ats_identifier or company.careers_url or ""
    return parse_workday_site(ident) is not None


async def _probe(http: HttpClient, url: str) -> str:
    result = await http.get_text(url)
    if result.blocked_by_robots:
        return "BLOCKED (robots.txt)"
    body = (result.text or "").lower()
    if any(token in body for token in ("captcha", "verify you are human", "cf-browser-verification")):
        return "BLOCKED"
    if result.status in (401, 403):
        return "BLOCKED"
    if result.ok:
        if result.status == 200 and len(result.text) > 20:
            return "OK"
        return "EMPTY"
    if result.status:
        classified = classify_source_status(result.error, http_status=result.status)
        return f"{classified} (HTTP {result.status})"
    return f"ERROR ({result.error or 'error'})"


async def _probe_workday(http: HttpClient, company: CompanyConfig) -> str:
    identifier = company.ats_identifier or company.careers_url or ""
    parsed = parse_workday_site(identifier)
    if parsed is None:
        return f"{company.name}: UNSUPPORTED (not a public Workday site URL)"
    host, tenant, site = parsed
    url = f"https://{host}/wday/cxs/{tenant}/{site}/jobs"
    result = await http.request(
        "POST",
        url,
        json_body={"appliedFacets": {}, "limit": CXS_PAGE_SIZE, "offset": 0, "searchText": ""},
        expect_json=True,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": f"https://{host}",
            "Referer": identifier,
        },
    )
    if result.blocked_by_robots:
        return (
            f"{company.name}: BLOCKED (robots.txt) tenant={tenant} site={site} "
            f"endpoint={url}"
        )
    payload = result.json() if result.text else None
    jobs = 0
    total = None
    if isinstance(payload, dict) and isinstance(payload.get("jobPostings"), list):
        jobs = len(payload["jobPostings"])
        total = payload.get("total")
    if result.ok:
        label = "OK" if jobs else "EMPTY"
        return (
            f"{company.name}: {label} HTTP {result.status} jobs={jobs} "
            f"total={total} tenant={tenant} site={site}"
        )
    classified = classify_source_status(result.error, http_status=result.status)
    preview = result.body_preview
    return (
        f"{company.name}: {classified} HTTP {result.status} jobs=0 "
        f"tenant={tenant} site={site} reason={preview or result.error}"
    )

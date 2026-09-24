# job-agent

Daily automated **job discovery and application tracking** for U.S. software-engineering-related roles.

The system discovers new postings, decides whether they fit a new-grad / 0–2 year software-engineering profile, requires a **direct company or ATS application URL**, enriches each row with **H-1B sponsorship evidence**, deduplicates against historical results, and maintains an XLSX application tracker.

It is a **job discovery and filtering system, not an immigration-law system**.

---

## Why H-1B is informational, not a filter

H-1B sponsorship information must **never** remove an otherwise-qualified job.

If a posting matches the target role, seniority, U.S. location, accepted employment type, freshness window, and has a valid direct application URL, it stays in the workbook regardless of sponsorship status:

| Status | Meaning | Job in XLSX? |
| --- | --- | --- |
| Confirmed | Current posting (or official policy) explicitly says sponsorship is available | Yes |
| Likely | Relevant recent historical H-1B/LCA evidence, but this posting does not guarantee sponsorship | Yes |
| Unknown | Not enough reliable information | Yes |
| Not Supported | Current posting explicitly says sponsorship is unavailable | Yes |

**H1BGrader data is historical sponsorship evidence. It does not guarantee that a particular current job will sponsor an H-1B.**

Jobs are not removed solely because sponsorship is unknown or not supported. The columns exist so *you* can decide. The pipeline will not decide for you.

There is deliberately **no** `require_h1b_sponsorship` setting. Adding one to `config/settings.yaml` fails validation.

---

## Architecture

The run is a **hybrid multi-agent pipeline** orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph). Typed shared state is a Pydantic `PipelineState`.

```
                    DISCOVERY
                       │
        ┌──────────────┼────────────────┐
        │              │                │
        ▼              ▼                ▼
   Structured      Company          Aggregator
   ATS sources     career pages     (Jobright, …)
        │              │                │
        └──────────────┼────────────────┘
                       ▼
              Normalize + cross-source dedup
                       ▼
     Role → Seniority → Location → Freshness → Direct URL
                       ▼
           H-1B evidence  (never a filter)
                       ▼
           Historical dedup → XLSX tracker
```

Jobright is optional. Structured ATS boards are preferred when a public identifier is configured or auto-detected from the careers URL. One source failing does not stop the others.

Deterministic code handles the large majority of decisions (keywords, dates, URL hosts, experience ranges, explicit sponsorship phrases). Gemini is used only when semantic judgment is actually useful: ambiguous role family, unclear seniority, unclear sponsorship wording, or a posting the parsers could not read. An LLM failure never crashes the run.

### Agents

| Agent | Role |
| --- | --- |
| Discovery | Jobright, Greenhouse, Lever, Ashby, SmartRecruiters, Workday, iCIMS, company career pages |
| Extraction | Raw payload → `Job` |
| Role classification | Software-engineering-related work, not a fixed title list |
| Seniority / experience | New grad / 0–2 years; preferred years never reject |
| Location & employment | U.S. only; Full-time / Contract / Internship / Co-op |
| Freshness | Last 24 hours of *elapsed* UTC time |
| Direct URL verification | ATS / company posting only |
| H-1B evidence | Historical + current language, never a filter |
| Deduplication | `company + job_id`, else title+location+URL |
| Quality control | Schema, URL, evidence labelling |
| Output / tracking | XLSX + SMTP summary |

One company or source failing is recorded in the run summary and skipped. The rest of the run continues.

---

## Discovery architecture

```
Company
  → Manual ATS (companies.yaml)
  → Cached ATS (config/ats_registry.yaml)
  → Auto ATS discovery (careers URL / public HTML)
  → Structured ATS adapter
  → Company career-page fallback
  → Optional Jobright
```

**Structured ATS boards are tier 1.** Jobright is an optional aggregator, not the foundation. One source or company failing does not stop the run. A successful query that returns zero jobs is recorded separately from a source failure.

Automatic ATS detection (`src/services/ats_discovery.py`) reads a company's public `careers_url` (redirects, canonical links, HTML, embedded JSON, known ATS URL patterns). Identifiers are extracted from real URLs only — never guessed from the company name. Successful detections are cached in `config/ats_registry.yaml`. Failed detections are cached without an identifier so they can be retried. Manual `ats` overrides always win.

### Registry files

| File | Role |
| --- | --- |
| `config/companies.yaml` | Production company universe. Enabled entries are crawled. Manual `ats` blocks are the source of truth. |
| `config/ats_registry.yaml` | Local cache of automatically discovered ATS type/identifier. No secrets, no job listings. |

The enabled production set is a verified initial registry (Greenhouse / Lever / Ashby / Workday boards that returned real public JSON). Unverified Fortune 500 names remain in `companies.yaml` with `enabled: false` so they can be turned on later without being deleted.

```yaml
  - name: Example Company
    fortune_500: true
    careers_url: https://example.com/careers
    ats:
      type: greenhouse          # or auto
      identifier: example       # verified public board handle — never invented
      discovery: manual         # manual beats cache and auto-detection
    discovery:
      enabled: true
      sources: [company_ats, company_careers, jobright]
```

Leave `ats.type: auto` and `identifier: null` to let the pipeline detect a public board from `careers_url`. Verify a manual identifier against the public board or API before adding it.

Every run prints **DISCOVERY HEALTH**, a **SOURCE SUMMARY**, a **FILTER FUNNEL**, **DISCOVERY COVERAGE**, **FRESHNESS BY SOURCE**, and a **COMPANY DISCOVERY REPORT** so an empty spreadsheet is explainable (source down vs 0 jobs vs role vs seniority vs 24h freshness vs missing URL).

Source outcomes are classified as `OK`, `EMPTY`, `ERROR`, `BLOCKED`, or `UNSUPPORTED`. A Workday HTTP 400 is `ERROR`, never `EMPTY`. `python -m src.main --source-health` probes public ATS endpoints and each enabled Workday CXS board with the same `limit=20` payload the adapter uses.

Workday CXS tenants tested here reject `limit` greater than 20 with HTTP 400. The adapter pages at 20 and bounds Workday-only concurrency (`discovery.sources.workday.max_concurrency`, default 2) so Greenhouse/Lever/Ashby stay on the global run concurrency.

---

## Direct application URLs

Accepted as the final URL: official company career postings and legitimate ATS postings (Greenhouse, Lever, Workday, Ashby, SmartRecruiters, iCIMS, and similar).

Rejected as the final URL: LinkedIn, Indeed, Glassdoor, Jobright, ZipRecruiter, Dice, generic Google/search results, and a generic careers homepage when an exact posting exists.

Job IDs, URLs, dates, and company names are never invented.

---

## Target profile

**Roles.** Software-engineering-related work. Titles such as Software Engineer, SDE, Backend / Frontend / Full Stack, Platform, Cloud, Infrastructure, DevOps, SRE, Systems, and Product Engineer are examples, not an exhaustive list. Data / ML / AI titles are neither auto-included nor auto-excluded; the described work decides.

**Seniority.** New grad, entry level, 0–2 years. Clear 3+ / Senior / Staff / Principal / Lead / Manager / Director / Architect roles are rejected. Preferred experience above 2 years does **not** reject a job whose required qualifications still fit.

**Location.** All U.S. locations, including U.S. remote, hybrid, and on-site. International-only postings are excluded.

**Employment type.** Full-time, Contract, Internship, Co-op.

**Freshness.** Posted within the last 24 hours of real elapsed UTC time. Production and GitHub Actions stay at 24 hours. `--freshness-hours 72` is diagnostic only.

Rule: use `posted_at` when present; otherwise use `updated_at` if `freshness_use_updated_when_posted_missing` is true; otherwise `UNKNOWN` (not accepted). An updated timestamp is never rewritten as a posted date. `DISCOVERED_DATE` is not treated as fresh.

---

## H-1B evidence

Primary historical source: [H1BGrader H-1B sponsors](https://h1bgrader.com/h1b-sponsors). The integration is modular (`src/sources/h1bgrader.py`) because the site structure may change. There is no assumed unofficial API. Public pages are fetched politely, `robots.txt` is honoured, CAPTCHAs and authentication are not bypassed, and no stealth techniques are used.

Matching prefers **company + similar role + location + recency**. Company-level history is never presented as job-specific sponsorship.

Evidence hierarchy:

1. Current job-specific sponsorship statement
2. Current official company sponsorship policy
3. Recent H-1B evidence for a similar role and location
4. Recent company-level H-1B evidence
5. Older historical H-1B evidence
6. No evidence

Statuses: `CONFIRMED`, `LIKELY`, `UNKNOWN`, `NOT_SUPPORTED`.

Gemini may interpret *ambiguous wording* only. It cannot decide pipeline eligibility and cannot turn historical records into `CONFIRMED`.

Workbook display values: Confirmed / Likely / Unknown / Not Supported. Evidence text avoids legal conclusions (“this job will sponsor your H-1B”) and states what the sources actually show.

---

## XLSX tracker

| File | Purpose |
| --- | --- |
| `data/current/jobs.xlsx` | Live application tracker |
| `data/archive/YYYY-MM-DD.xlsx` | Daily snapshot |

Columns (in this order): Company, Job Title, Normalized Role, Location, Remote Type, Employment Type, Posted Date, Updated Date, Job ID, Source, Direct Application URL, Applied, Status, Found At, Visa Sponsorship, Sponsorship Evidence, and optional H-1B Last Verified.

`Applied` is a dropdown (`☐ Not Applied` / `☑ Applied`). `Status` is a dropdown (Not Started, Applied, OA, Recruiter Screen, Interview, Rejected, Offer, Withdrawn).

Manual `Applied` and `Status` values are preserved by job identity (`Company` + `Job ID`, or company + normalized title + location + direct URL when there is no job id), including a same-day rerun. Row order does not own that state. The current workbook is the latest successful snapshot: only jobs that qualified on that run. A successful run with zero matches replaces it with a header-only workbook and still writes today's archive. An older day's archive is left as it was. A crash or failed workbook write does not replace `data/current/jobs.xlsx` and does not publish a new archive. Two postings with the same title and **different Job IDs** both remain. A job that still qualifies is written again; it is not treated as a brand-new row, so a prior `☑ Applied` / `Interview` stays.

Workbooks freeze the header, enable autofilter, wrap text, add URL hyperlinks, and prefix `=`, `+`, `-`, `@` to block formula injection.

---

## Gemini

Default LLM provider is Gemini via the current [`google-genai`](https://googleapis.github.io/python-genai/) SDK (`google.genai`), not the deprecated `google-generativeai` package.

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=<secret>
```

Set `LLM_PROVIDER=none` (or leave the key empty) to run fully deterministically. Ambiguous cases then stay `UNKNOWN` / rejected by the relevant *eligibility* filter — never guessed.

---

## Local setup

The project uses a dedicated `.venv`. It is git-ignored and must never be committed. GitHub Actions creates its own fresh environment and does not use yours.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows CMD

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in secrets. `.env` is git-ignored.

Optional, for JS-heavy listing pages:

```bash
python -m playwright install chromium
```

### Commands

```bash
python -m src.main                              # daily production run (24-hour freshness)
python -m src.main --diagnostic                 # coverage + per-company report + funnel; no email
python -m src.main --source-health              # probe ATS/Jobright/H1BGrader + Workday CXS
python -m src.main --freshness-hours 72 --diagnostic   # diagnostic only; production stays 24h
python -m src.main --dry-run                    # pipeline + summary, no XLSX write, email skipped
python -m src.main --company "Airbnb"
python -m src.main --fixture-mode --dry-run
pytest
```

`--freshness-hours 72` is for debugging timestamp yield only. GitHub Actions continues to use the configured 24-hour requirement. Jobs with no posted/updated timestamp stay `UNKNOWN` and are not treated as fresh.

`pyproject.toml` is the authoritative dependency list.

---

## Configuration

| File | Purpose |
| --- | --- |
| `config/companies.yaml` | Company universe (enabled production registry + disabled Fortune 500) |
| `config/ats_registry.yaml` | Cached automatic ATS detections (no secrets, no jobs) |
| `config/settings.yaml` | Freshness, sources, URL policy, visa *interpretation*, LLM, output, email |
| `config/roles.yaml` | Role families, seniority signals, H-1B role groups |

Visa settings control how sponsorship evidence is gathered and displayed. They cannot hide a qualified job.

---

## Tests

Tests are offline. They use fixtures under `tests/fixtures/` and never depend on live websites.

```bash
pytest
```

Coverage includes core filters, 24-hour freshness, UTC normalisation, experience vs preferred years, equivalent roles, U.S. locations, employment types, job IDs, cross-source and historical dedup, direct URL validation, XLSX quality (hyperlinks, dropdowns, same-day preservation, formula injection), and the full H-1B matrix (explicit yes/no, “must not require sponsorship now or in the future”, historical same-role / unrelated-role / different-location / old vs recent, lookup failure, Gemini schema constraints, and the hard rule that `NOT_SUPPORTED` and `UNKNOWN` jobs remain).

---

## GitHub Actions

`.github/workflows/daily_jobs.yml` is the only scheduled workflow. It runs on a fresh `ubuntu-latest` runner.

Schedule: `0 12 * * *`, which is 12:00 UTC every day (about 8:00 AM US Eastern during EDT, and 7:00 AM during EST). GitHub cron starts are not guaranteed at that exact minute; the job can begin later. To run it by hand: Actions → Daily job discovery → Run workflow (`workflow_dispatch`).

```
Checkout → Python 3.12 → fresh venv → pip install -e . → Playwright Chromium
  → python -m src.main → commit data/current and data/archive if changed → push
```

The production command is `python -m src.main`. The workflow does not pass `--dry-run`, `--fixture-mode`, `--diagnostic`, or `--freshness-hours`. Freshness stays at 24 hours.

`permissions: contents: write` is the only permission. It lets the job push the tracker. The commit step uses the `github-actions[bot]` identity and runs `git add data/current data/archive`. If that staged diff is empty, it does not commit. It does not add the rest of the tree. `.venv`, `.env`, caches, logs, `*.xlsx.tmp`, and secrets stay untracked.

No secret is required for the run to succeed.

| Secret | Role |
| --- | --- |
| `GEMINI_API_KEY` | Optional. Enables Gemini for ambiguous classification. Without it the run stays deterministic. |
| `LLM_PROVIDER`, `GEMINI_MODEL` | Optional overrides. |
| `SMTP_HOST`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `NOTIFICATION_EMAIL` | Optional, and all four are needed together. Missing SMTP logs a warning and the run still succeeds. |
| `SMTP_PORT` | Optional. Defaults to 587. |
| `NOTIFICATION_FROM` | Optional. Defaults to the SMTP username. |

A failed pipeline exits non-zero, the job fails, and the commit step does not run, so the last committed workbook stays as it was. A successful run with zero qualifying jobs is valid: it writes a header-only `data/current/jobs.xlsx` and today's archive, and those files are committed when they differ. An SMTP failure after a successful write does not fail the job, and the tracker commit still happens.

---

## Email notifications

SMTP, using `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, and `NOTIFICATION_EMAIL`. If configuration is missing, the log contains `WARNING: email notifications disabled` and the job run still succeeds. A later SMTP error is also a warning: the workbook and archive are already written, and the process still exits 0. The message is a short count plus new-job titles, or a note that zero matches is a valid run. It does not include the workbook.

---

## Fixture mode and dry run

`--fixture-mode` runs extraction, classification, H-1B matching, deduplication, and XLSX generation against `tests/fixtures/` with no outbound HTTP.

`--dry-run` still executes the pipeline and prints the summary but does not write the workbook and skips email unless you pass `--email`.

---

## Scraping limitations and security

- Honour `robots.txt`. Rate-limit per host. No CAPTCHA solving, no auth bypass, no anti-bot evasion, no stealth fingerprints.
- Scraped markup is treated as untrusted data. It is never executed.
- Secrets live in the environment only. Logs redact API keys and passwords.
- XLSX values that look like formulas are stored as text.
- Site structure will change. Adapters (Jobright, H1BGrader, Workday, iCIMS) are isolated so they can be replaced.

---

## Known limitations

- Many Fortune 500 career sites do not expose a public job-board API. Those companies are crawled best-effort from `careers_url` and will often appear as isolated failures until you add a verified `ats_type` + `ats_identifier`.
- Jobright listings are JS-hydrated. Static HTML has `__NEXT_DATA__` but no job objects; Playwright Chromium is required to parse live cards. GitHub Actions installs Chromium. Locally: `python -m playwright install chromium`. Jobright remains optional.
- Some Workday site URLs are stale (for example Elevance `ELV_EXT` currently 404s). Those are recorded as structured `ERROR` and career-page fallback runs. Replacement identifiers are not guessed.
- Freshness requires a posted or updated timestamp. Postings with no date are not assumed to be new.
- Gemini is optional. Without an API key the pipeline stays deterministic.
- Historical H-1B/LCA rows describe what an employer did in past fiscal years. They are not a prediction about the current requisition.

---

## Run summary

Every run prints (and can email) companies attempted/succeeded/failed, jobs discovered/extracted/accepted, rejections by role / seniority / location / employment type / freshness / invalid URL, duplicates, new vs historical, H-1B Confirmed / Likely / Unknown / Not Supported, lookup failures, XLSX path, and email status.

The summary will never contain “jobs rejected because H-1B sponsorship was unavailable,” because that is not a filter.

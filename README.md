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

The daily run is one LangGraph pipeline. Typed shared state is a Pydantic `PipelineState`. The production order is fixed:

```
Discovery
   ↓
Extraction
   ↓
Role
   ↓
Seniority / Experience
   ↓
Location
   ↓
Employment
   ↓
Freshness
   ↓
URL Verification
   ↓
H-1B Enrichment
   ↓
Deduplication
   ↓
QC
   ↓
Job Intelligence
   ↓
Critic
   ↓
XLSX Output
```

Employment checks run after location passes, inside the location node. The critic runs after job intelligence, inside the intelligence node. Neither step reorders the gates above.

Deterministic gates run before semantic review. A title that is clearly software engineering, or clearly not, never waits on Gemini. Gemini is optional. It cannot change job id, source, official URL, posted timestamp, date source, location, or freshness. Freshness is a deterministic tier from the employer timestamp. H-1B enrichment never removes a job.

Production discovery uses configured companies plus a bounded public sample of Greenhouse, Ashby, and Workday boards. Career pages stay on the configured-company list. Jobright is disabled. Workday browser discovery is disabled. iCIMS and SmartRecruiters can be recognized on a public URL, but they are not orchestrated production collectors. One source or one board failing does not stop the others. A source that succeeds and returns zero jobs is empty, not failed. A board that cannot be retrieved is failed, not empty.

### Agents

| Agent | Role |
| --- | --- |
| Discovery | Configured companies, plus a bounded public Greenhouse/Ashby board sample. Jobright stays off. |
| Extraction | Raw payload → `Job` |
| Role classification | Deterministic accept, deterministic reject, or semantic review |
| Seniority / experience | New grad / 0–2 years; preferred years never reject |
| Location | U.S. only, including U.S. remote |
| Employment | Full-time / Contract / Internship / Co-op, after location passes |
| Freshness | Employer-timestamp tiers. Recent jobs stay eligible; stale and old jobs do not. |
| Direct URL verification | Official ATS or company posting only |
| H-1B evidence | Historical + current language, never a filter |
| Deduplication | `company + job_id`, else title+location+URL |
| Quality control | Schema, URL, evidence labelling |
| Job intelligence / critic | Optional resume comparison. Skipped when no private profile exists. Does not accept or reject. |
| Output / tracking | XLSX + optional SMTP summary |

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

Automatic ATS detection (`src/services/ats_discovery.py`) reads a company's public `careers_url` (redirects, canonical links, HTML, embedded JSON, known ATS URL patterns). Identifiers are extracted from real URLs only — never guessed from the company name. `config/ats_registry.yaml` is static repository configuration. Production sets `persist_ats_registry: false`, so the daily run does not rewrite it. Manual `ats` overrides always win.

### Registry files

| File | Role |
| --- | --- |
| `config/companies.yaml` | Production company universe. Enabled entries are crawled. Manual `ats` blocks are the source of truth. |
| `config/ats_registry.yaml` | Static ATS identifiers read during discovery. Production does not rewrite this file. No secrets, no job listings. |

The production set includes verified Greenhouse / Lever / Ashby / Workday boards and the Fortune 500 official career pages in `companies.yaml`. Companies without a verified job-board id are scraped from their `careers_url`.

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

Every run prints **DISCOVERY HEALTH**, a **SOURCE SUMMARY**, a **FILTER FUNNEL**, **DISCOVERY COVERAGE**, **FRESHNESS BY SOURCE**, global board counts, freshness tiers, and a **COMPANY DISCOVERY REPORT** so an empty spreadsheet is explainable (source down vs 0 jobs vs role vs seniority vs freshness tier vs missing URL).

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

**Freshness.** Elapsed UTC time from the employer timestamp, not from the crawl.

| Tier | Age | Daily tracker |
| --- | --- | --- |
| VERY_FRESH | under 24 hours | eligible |
| FRESH | 24 hours up to 72 hours | eligible |
| RECENT | 72 hours up to 168 hours (7 days) | eligible |
| AGING | 168 hours up to 336 hours (14 days) | eligible only when role and seniority already passed deterministically |
| STALE | 336 hours up to 720 hours (30 days) | excluded |
| OLD | 720 hours or more | excluded |
| UNKNOWN | no employer timestamp | excluded |

An age of exactly 24:00:00 is FRESH, not VERY_FRESH. Exactly 720:00:00 is OLD. `run.freshness_hours` remains 24 for previews and `--freshness-hours`. That flag does not widen the daily window.

Rule: use `posted_at` when present; otherwise use `updated_at` if `freshness_use_updated_when_posted_missing` is true; otherwise `UNKNOWN`. An updated timestamp is never rewritten as a posted date. `DISCOVERED_DATE`, `first_seen_at`, and `last_seen_at` are not posting times. A job found today can still be OLD.

`posted_at` and `updated_at` come from the employer. `first_seen_at` is the first workbook or archive day this job was seen. `last_seen_at` is the latest. Seeing a job again does not make it new, and it does not refresh `posted_at`.

### Two discovery modes

1. **Configured company discovery.** `config/companies.yaml` lists companies and, when known, their ATS board. Greenhouse, Lever, Ashby, Workday, and career-page fallback all still run for those companies.
2. **Bounded public board discovery.** After the configured crawl, the run reads a date-rotated Internet Archive CDX sample. Greenhouse and Ashby samples are path prefixes on `boards.greenhouse.io`, `job-boards.greenhouse.io`, and `jobs.ashbyhq.com`. Workday samples are public `*.wdN.myworkdayjobs.com` and `*.wdN.myworkdaysite.com` career URLs. Workday does not publish a universal jobs API, so the sample is only what that archive page returns. Each Workday URL is accepted only when the public CXS endpoint returns a job-search payload (`appliedFacets` empty, `limit` 20). Those boards then go through the same CXS collector as a configured Workday company, including the 2,000-job cap and the existing `searchText` partitions. A company does not have to be listed in `companies.yaml` to be in the sample. Boards already configured are collected once. The same job found both ways becomes one job.

This is not a Greenhouse, Ashby, or Workday directory. Common Crawl's CDX API is not used, because its robots.txt disallows that path. At most `max_boards_per_run` new boards are fetched per ATS. Coverage is partial. The report says `GLOBAL DISCOVERY: PARTIAL COVERAGE`. It does not say that every Workday job was discovered. If the archive index is down, configured companies still run and the daily job does not fail for that reason alone.

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

Set `LLM_PROVIDER=none`, or leave the key empty, to run without Gemini. Deterministic software-engineering accepts and deterministic rejects still complete.

Role outcomes stay in three internal states:

| State | Meaning |
| --- | --- |
| `DETERMINISTIC_ACCEPT` | Clear software-engineering work. Gemini is not called. |
| `DETERMINISTIC_REJECT` | Clear non-match, such as an account-manager or manager title. Recorded as `ROLE_MISMATCH`. Gemini is not called. |
| `SEMANTIC_REVIEW_REQUIRED` | Ambiguous family (AI, ML, data, analytics, QA/test, or other unclear titles). Gemini is called only when the circuit allows it. |

Semantic results are `ACCEPT`, `REJECT`, `UNCERTAIN`, or `UNAVAILABLE`. Only `ACCEPT` with confidence at least 0.6 can advance. `UNCERTAIN` does not qualify. Confidence 0.59 stays `UNCERTAIN`. `UNAVAILABLE` is not `ROLE_MISMATCH`.

`SEMANTIC_REVIEW_UNAVAILABLE` means the model was not able to review the posting. It does not mean the posting was rejected. A large unavailable count during an outage is not a large role-mismatch count. Fresh postings are counted separately, so a fresh deterministic reject is not reported as a fresh job waiting on Gemini.

The in-process circuit opens after repeated provider failures and does not retry every ambiguous posting while it is open. A later successful probe can close it. A new process starts closed. Gemini cannot modify job id, source, official URL, posted timestamp, date source, location, or freshness. Those fields are restored before a semantic accept is kept.

An unavailable Gemini call, a missing API key, or an open circuit does not fail the daily run.

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

`data/candidate/resume.pdf` and `data/candidate/profile.json` are optional private files. They are git-ignored. `python -m src.main` does not require them. Without a saved profile, job intelligence and the critic are skipped and the rest of the pipeline still runs. `--resume-smoke-test` is the command that reads the resume. Do not commit the PDF, the extracted profile, or profile history.

Optional, for JS-heavy listing pages:

```bash
python -m playwright install chromium
```

### Commands

```bash
python -m src.main                              # daily production run (tiered freshness)
python -m src.main --diagnostic                 # coverage + per-company report + funnel; no email
python -m src.main --source-health              # probe ATS/Jobright/H1BGrader + Workday CXS
python -m src.main --freshness-hours 72 --diagnostic   # diagnostic only; production stays 24h
python -m src.main --dry-run                    # pipeline + summary, no XLSX write, email skipped
python -m src.main --company "Airbnb"
python -m src.main --fixture-mode --dry-run --no-email
python -m src.main --multi-source-smoke-test --sources greenhouse,workday,lever,ashby --dry-run --no-email
python -m src.main --multi-source-browser-smoke-test --sources greenhouse,workday --dry-run --no-email
pytest
```

Smoke commands are dry-runs. They do not write `data/current/jobs.xlsx` or today's archive, and they do not change `workday.browser_enabled` in `config/settings.yaml`. The browser smoke turns the browser on for that process only.

`--freshness-hours 72` changes the observational preview window only. GitHub Actions uses the freshness tiers in `config/settings.yaml`. Jobs with no posted/updated timestamp stay `UNKNOWN` and are not treated as fresh.

`pyproject.toml` is the authoritative dependency list.

---

## Configuration

| File | Purpose |
| --- | --- |
| `config/companies.yaml` | Configured companies and their sources. Not the full Greenhouse/Ashby universe. |
| `config/ats_registry.yaml` | Static ATS identifiers shipped with the repo. Daily runs do not rewrite it. |
| `config/settings.yaml` | Freshness, sources, URL policy, visa *interpretation*, LLM, output, email |
| `config/roles.yaml` | Role families, seniority signals, H-1B role groups |

Visa settings control how sponsorship evidence is gathered and displayed. They cannot hide a qualified job.

---

## Tests

Tests are offline. They use fixtures under `tests/fixtures/` and never depend on live websites.

```bash
pytest
```

Coverage includes core filters, freshness tiers, UTC normalisation, experience vs preferred years, equivalent roles, U.S. locations, employment types, job IDs, cross-source and historical dedup, direct URL validation, XLSX quality (hyperlinks, dropdowns, same-day preservation, formula injection), and the full H-1B matrix (explicit yes/no, “must not require sponsorship now or in the future”, historical same-role / unrelated-role / different-location / old vs recent, lookup failure, Gemini schema constraints, and the hard rule that `NOT_SUPPORTED` and `UNKNOWN` jobs remain).

---

## GitHub Actions

`.github/workflows/daily_jobs.yml` is the only scheduled workflow. It runs on a fresh `ubuntu-latest` runner.

Schedule: `0 12 * * *`, which is 12:00 UTC every day (about 8:00 AM US Eastern during EDT, and 7:00 AM during EST). GitHub cron starts are not guaranteed at that exact minute; the job can begin later. To run it by hand: Actions → Daily job discovery → Run workflow (`workflow_dispatch`).

```
Checkout → Python 3.12 → fresh venv → pip install -e . → Playwright Chromium
  → python -m src.main → commit data/current and data/archive if changed → push
```

The production command is `python -m src.main`. The workflow does not pass `--dry-run`, `--fixture-mode`, `--diagnostic`, or `--freshness-hours`. Daily eligibility uses the freshness tiers in `config/settings.yaml`.

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

The workflow does not pass smoke flags, does not enable the Workday browser, and does not enable Jobright. Secrets are repository secrets mapped to environment variables. They are not written into the workflow command line. The commit adds only `data/current` and `data/archive`.

| Condition | Exit |
| --- | --- |
| Missing or malformed configuration | Nonzero (`2` for a missing config directory) |
| Current workbook write fails | Nonzero (`1`). The previous workbook is left in place. |
| Archive write fails | Nonzero. The previous current workbook and older archives are not replaced. |
| Gemini unavailable or circuit open | `0` when the rest of the pipeline finishes |
| SMTP missing or SMTP send fails | `0` |
| One source fails | `0` when other sources finish and output succeeds |
| Zero final candidates | `0` |

---

## Latest production example

Phase 15, one production run on 2026-09-27, exit 0, runtime 280.392 seconds:

* 14,695 discovered
* 14,581 deduplicated
* 3 fresh
* 0 final candidates
* 1,702 semantic reviews unavailable because the Gemini circuit was already open
* 0 fresh jobs waiting on semantic review

Those 1,702 unavailable reviews were not role mismatches. The three fresh postings were deterministic role mismatches before semantic review. Workday remained partial, including NVIDIA and Booz Allen at the public 2,000-job cap. Career-page fallback remained partial. No second production crawl is required to read this example.

---

## Email notifications

SMTP uses the existing names `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, and `NOTIFICATION_EMAIL`. `NOTIFICATION_FROM` is optional and defaults to `SMTP_USERNAME`. `SMTP_PORT` defaults to 587. Local values belong in gitignored `.env`. `.env.example` has empty placeholders only. GitHub Actions maps those same names from repository secrets. Do not commit a password, and do not put one in YAML.

All four of host, username, password, and recipient are required before a message is sent. If any are missing, the log says email notifications are disabled (`SMTP_NOT_CONFIGURED`) and the run still exits 0. If the server rejects the login or the send fails, the log records the exception type and a redacted error. The password is removed from that message. The workbook and archive are already written, and the process still exits 0.

The hosted run on 2026-09-27 reached Microsoft 365 and received `535 5.7.3 Authentication unsuccessful`. That is a rejected mailbox login, not a pipeline failure. Email delivery is not confirmed until that login is accepted. The message itself is a short count plus new-job titles, or a note that zero matches is a valid run. It does not include the workbook, job descriptions, or credentials.

---

## Fixture mode and dry run

`--fixture-mode` runs extraction, classification, H-1B matching, deduplication, and XLSX generation against `tests/fixtures/` with no outbound HTTP.

`--dry-run` still executes the pipeline and prints the summary but does not write the workbook and skips email unless you pass `--email`.

---

## Safety

The daily pipeline uses public job pages and public ATS endpoints.

- It does not bypass authentication, CAPTCHAs, robots rules, or rate limits.
- It does not spoof a browser fingerprint, rotate proxies, or add stealth behavior.
- It does not call a private Workday API and does not invent Workday facet ids. The public CXS body sends `appliedFacets: {}`, `limit: 20`, an offset, and optional `searchText`.
- It does not submit applications or collect credentials.
- It does not need a database, Redis, S3, or another cloud store.
- Scraped markup is data. It is not executed as JavaScript or shell.
- Secrets stay in the environment or in GitHub Actions secrets. Logs are not a place for API keys, SMTP passwords, or resume text.
- The workbook has the 17 public columns listed above. It does not add semantic, debug, or LLM columns, and it does not store job descriptions.

---

## Known limitations

- Workday discovery is partial in two ways. Configured boards still use the public CXS collector: page size 20, and an unfiltered board stops near 2,000 jobs. NVIDIA and Booz Allen stay incomplete at that cap even after the existing `searchText` partitions. Global discovery adds a bounded archive sample of other public career hosts. That sample is not every Workday customer. A discovered board that hits the cap stays partial. A 404 tenant is a failure for that tenant, not a reason to guess a new site id. Browser discovery stays off in production.
- Career-page fallback is partial. Some companies fail, some pages are unsupported or blocked, and many rows have no timestamp. A company that returns zero jobs without an error, such as AMD's current career result, is an empty success. iCIMS is not a production collector.
- Jobright is disabled in production. Its listing pages are JS-heavy and detail pages have returned 403. The workflow does not enable it.
- Greenhouse, Lever, and Ashby cover only companies with a verified public board id.
- Freshness requires a posted timestamp, or an updated timestamp when the posted time is missing. A discovery timestamp is never treated as the posted time. Unknown dates are not fresh.
- Gemini is optional. During an outage, ambiguous roles stay unreviewed. They are not silently accepted or silently labeled role mismatches.
- Historical H-1B/LCA rows describe what an employer did in past fiscal years. They are not a prediction about the current requisition.
- GitHub Actions cron can start later than 12:00 UTC. A local simulation of the workflow is not a hosted Actions run.

---

## Run summary

Every run prints (and can email) companies attempted/succeeded/failed, jobs discovered/extracted/accepted, rejections by role / seniority / location / employment type / freshness / invalid URL, duplicates, new vs historical, H-1B Confirmed / Likely / Unknown / Not Supported, lookup failures, XLSX path, and email status.

The summary will never contain “jobs rejected because H-1B sponsorship was unavailable,” because that is not a filter.

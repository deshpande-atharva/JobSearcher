"""H1BGrader historical sponsorship adapter.

Primary public page: ``https://h1bgrader.com/h1b-sponsors``.

H1BGrader publishes historical H-1B/LCA data derived from U.S. DOL/USCIS-related
datasets. This adapter treats that data as *historical evidence only*. It never
decides whether a job stays in the pipeline, and it never claims a current
posting will sponsor.

There is no documented official API. The adapter reads permitted public pages,
honours robots.txt (via the shared HTTP client), uses a polite request rate,
and is isolated so a site-structure change can be replaced without touching
the rest of the H-1B logic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from src.models.config import AppConfig, VisaSettings
from src.models.job import H1BLookupResult, H1BRecord
from src.sources.base import HttpClient
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import parse_html, pick
from src.utils.dates import parse_datetime, utcnow
from src.utils.logging import get_logger
from src.utils.normalization import normalize_company_name

__all__ = ["H1BGraderClient"]

log = get_logger(__name__)

_SPONSOR_PATH_RE = re.compile(
    r"""https?://(?:www\.)?h1bgrader\.com/(?:h1b-sponsors?|h1b-visa-sponsors?)/([a-z0-9\-]+)""",
    re.IGNORECASE,
)


class H1BGraderClient:
    """Look up historical H-1B/LCA records for one employer.

    Lookups are cached in memory for the run and optionally on disk so the same
    company is never requested twice in a pipeline execution.
    """

    def __init__(
        self,
        config: AppConfig,
        http: HttpClient | None = None,
        *,
        fixture_mode: bool | None = None,
    ) -> None:
        self.config = config
        self.settings: VisaSettings = config.settings.visa
        self.http = http
        self.fixture_mode = config.fixture_mode if fixture_mode is None else fixture_mode
        self._memory: dict[str, H1BLookupResult] = {}

    def _cache_key(self, company: str) -> str:
        return normalize_company_name(company) or company.strip().lower()

    async def lookup(self, company: str, aliases: tuple[str, ...] = ()) -> H1BLookupResult:
        key = self._cache_key(company)
        if key in self._memory:
            cached = self._memory[key].model_copy()
            cached.from_cache = True
            return cached

        disk = self._read_disk(key)
        if disk is not None:
            disk.from_cache = True
            self._memory[key] = disk
            return disk.model_copy()

        if self.fixture_mode:
            result = self._from_fixture(company, aliases)
        else:
            result = await self._from_network(company, aliases)

        self._memory[key] = result
        self._write_disk(key, result)
        return result.model_copy()

    def _from_fixture(self, company: str, aliases: tuple[str, ...]) -> H1BLookupResult:
        store = FixtureStore(self.config.fixture_dir)
        names = [f"{slugify(company)}.json", *[f"{slugify(alias)}.json" for alias in aliases]]
        payload = store.load_first("h1bgrader", names)
        if payload is None:
            return H1BLookupResult(
                company=company,
                found=False,
                retrieved_at=utcnow(),
                error=None,
            )
        return _payload_to_result(company, payload)

    async def _from_network(self, company: str, aliases: tuple[str, ...]) -> H1BLookupResult:
        if self.http is None:
            return H1BLookupResult(
                company=company,
                found=False,
                error="H1BGrader HTTP client is not available",
                retrieved_at=utcnow(),
            )

        queries = [company, *aliases]
        last_error: str | None = None
        for query in queries:
            try:
                result = await self._search_public_pages(query)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.info("h1bgrader lookup error", company=query, error=last_error)
                continue
            if result.found or result.error is None:
                result.company = company
                return result
            last_error = result.error

        return H1BLookupResult(
            company=company,
            found=False,
            error=last_error,
            retrieved_at=utcnow(),
        )

    async def _search_public_pages(self, query: str) -> H1BLookupResult:
        """Fetch the public sponsor index / company page and parse tables.

        URL shapes observed on the public site (subject to change -- that is
        why this method is the only place that knows them):

        * ``/h1b-sponsors`` with a query string
        * ``/h1b-visa-sponsors/{slug}`` linked from the index
        """
        assert self.http is not None
        index_url = self.settings.base_url.rstrip("/")
        search_url = f"{index_url}?q={quote_plus(query)}"
        page = await self.http.get_text(search_url)
        if page.blocked_by_robots:
            return H1BLookupResult(
                company=query,
                found=False,
                error="disallowed by robots.txt",
                retrieved_at=utcnow(),
            )
        if not page.ok:
            return H1BLookupResult(
                company=query,
                found=False,
                error=page.error or f"HTTP {page.status}",
                retrieved_at=utcnow(),
            )

        records = _parse_sponsor_html(page.text, query)
        detail_url = _first_sponsor_link(page.text)
        if detail_url and not records:
            detail = await self.http.get_text(detail_url)
            if detail.ok:
                records = _parse_sponsor_html(detail.text, query) or records

        return H1BLookupResult(
            company=query,
            found=bool(records),
            records=records,
            retrieved_at=utcnow(),
        )

    # --- disk cache ---------------------------------------------------------

    def _cache_path(self) -> Path:
        return self.config.path(self.settings.cache_path)

    def _read_disk(self, key: str) -> H1BLookupResult | None:
        path = self._cache_path()
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        entry = payload.get(key) if isinstance(payload, dict) else None
        if not isinstance(entry, dict):
            return None
        retrieved = parse_datetime(entry.get("retrieved_at"))
        ttl_hours = self.settings.cache_ttl_hours
        if retrieved is not None and ttl_hours > 0:
            age = (utcnow() - retrieved).total_seconds() / 3600.0
            if age > ttl_hours:
                return None
        try:
            return H1BLookupResult.model_validate(entry.get("result"))
        except Exception:
            return None

    def _write_disk(self, key: str, result: H1BLookupResult) -> None:
        if self.fixture_mode:
            return
        path = self._cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict[str, Any] = {}
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
                except ValueError:
                    existing = {}
            existing[key] = {
                "retrieved_at": utcnow().isoformat(),
                "result": result.model_dump(mode="json"),
            }
            path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            log.debug("could not persist h1b cache", error=str(exc))


def _payload_to_result(company: str, payload: Any) -> H1BLookupResult:
    if not isinstance(payload, dict):
        return H1BLookupResult(company=company, found=False, retrieved_at=utcnow())
    raw_records = pick(payload, "records", "jobs", default=[]) or []
    records: list[H1BRecord] = []
    if isinstance(raw_records, list):
        for item in raw_records:
            if not isinstance(item, dict):
                continue
            records.append(
                H1BRecord(
                    employer=str(pick(item, "employer", "company") or company),
                    job_title=str(pick(item, "job_title", "title") or "") or None,
                    city=str(pick(item, "city") or "") or None,
                    state=str(pick(item, "state") or "") or None,
                    fiscal_year=_as_int(pick(item, "fiscal_year", "year")),
                    approvals=_as_int(pick(item, "approvals", "certified")),
                    denials=_as_int(pick(item, "denials", "denied")),
                    record_date=parse_datetime(pick(item, "record_date", "date")),
                    provider="h1bgrader",
                )
            )
    error = pick(payload, "error")
    return H1BLookupResult(
        company=company,
        found=bool(records) or bool(pick(payload, "found")),
        records=records,
        error=str(error) if error else None,
        retrieved_at=utcnow(),
    )


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_sponsor_link(html: str) -> str | None:
    match = _SPONSOR_PATH_RE.search(html or "")
    return match.group(0) if match else None


def _parse_sponsor_html(html: str, company: str) -> list[H1BRecord]:
    """Best-effort extraction of historical rows from a public H1BGrader page.

    The site layout can change. We look for ordinary HTML tables whose headers
    mention title / city / year, and for JSON blobs that already look like
    records. Missing or unreadable markup yields an empty list, which the
    caller reports as UNKNOWN -- never as a pipeline failure.
    """
    soup = parse_html(html)
    records: list[H1BRecord] = []

    for table in soup.find_all("table"):
        headers = [
            re.sub(r"\s+", " ", th.get_text(" ", strip=True)).lower()
            for th in table.find_all("th")
        ]
        if not headers:
            continue
        index = {name: i for i, name in enumerate(headers)}

        def col(*names: str) -> int | None:
            for name in names:
                for header, idx in index.items():
                    if name in header:
                        return idx
            return None

        title_i = col("job title", "title", "occupation")
        city_i = col("city", "worksite city")
        state_i = col("state", "worksite state")
        year_i = col("year", "fiscal", "fy")
        app_i = col("approval", "certified", "lcsa")
        den_i = col("denial", "denied")
        emp_i = col("employer", "company", "sponsor")

        for row in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if not cells:
                continue
            title = cells[title_i] if title_i is not None and title_i < len(cells) else None
            if not title and title_i is None:
                continue
            records.append(
                H1BRecord(
                    employer=(
                        cells[emp_i]
                        if emp_i is not None and emp_i < len(cells) and cells[emp_i]
                        else company
                    ),
                    job_title=title or None,
                    city=cells[city_i] if city_i is not None and city_i < len(cells) else None,
                    state=cells[state_i] if state_i is not None and state_i < len(cells) else None,
                    fiscal_year=_as_int(cells[year_i] if year_i is not None and year_i < len(cells) else None),
                    approvals=_as_int(cells[app_i] if app_i is not None and app_i < len(cells) else None),
                    denials=_as_int(cells[den_i] if den_i is not None and den_i < len(cells) else None),
                )
            )

    return records

"""Ashby job board adapter.

Uses Ashby's public posting API:
``https://api.ashbyhq.com/posting-api/job-board/{jobBoardName}``

``jobBoardName`` is the handle from the public ``jobs.ashbyhq.com/{handle}``
board URL. No API key is required for the public board endpoint.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.services.board_priority import cap_board_postings
from src.sources.base import DiscoverySource, SourceError, SourceResult
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import collect_postings, pick
from src.utils.dates import parse_datetime
from src.utils.normalization import clean_text, html_to_text
from src.utils.urls import extract_job_id, host_of, is_http_url

__all__ = ["AshbySource"]

API_TEMPLATE = "https://api.ashbyhq.com/posting-api/job-board/{handle}"
_AMBIGUOUS_DATE = re.compile(r"\b(found|discovered|listed|crawled|seen|indexed)\b", re.IGNORECASE)


class AshbySource(DiscoverySource):
    name: ClassVar[str] = "ashby"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "ashby"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.cap_reached = False
        self.cap_note = ""

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        self.cap_reached = False
        self.cap_note = ""
        if company is None or not company.ats_identifier:
            return []
        handle = company.ats_identifier.strip().strip("/")

        if self.ctx.fixture_mode:
            payload = FixtureStore(self.config.fixture_dir).load_first(
                self.name, [f"{slugify(company.name)}.json", f"{slugify(handle)}.json"]
            )
            if payload is None:
                return []
        else:
            payload = await self.http.get_json(
                API_TEMPLATE.format(handle=handle),
                params={"includeCompensation": "true"},
            )

        jobs = pick(payload, "jobs", default=None)
        if not isinstance(jobs, list):
            raise SourceError(f"unexpected Ashby payload for board {handle!r}")

        postings = collect_postings(
            jobs,
            lambda entry: self._to_posting(entry, company, handle),
            log=self.log,
        )
        return self._limit(postings)

    async def discover_result(self, company: CompanyConfig | None = None) -> SourceResult:
        result = await super().discover_result(company)
        if self.cap_note:
            result.diagnostics["cap_note"] = self.cap_note
        if self.cap_reached:
            result.diagnostics["cap_reached"] = True
            result.diagnostics["source_list_cap_reached"] = True
        return result

    def _limit(self, postings: list[RawJobPosting]) -> list[RawJobPosting]:
        settings = self.config.settings.discovery.sources.ashby
        kept, capped, note = cap_board_postings(
            postings,
            self.config,
            limit=settings.max_jobs,
            prioritize=settings.prioritize_fresh_targets,
        )
        self.cap_reached = capped
        self.cap_note = note
        return kept

    def _to_posting(
        self, entry: Any, company: CompanyConfig, handle: str
    ) -> RawJobPosting | None:
        if not isinstance(entry, dict):
            return None

        # Ashby marks unpublished/internal roles; skip anything not publicly listed.
        if pick(entry, "isListed") is False:
            return None

        title = clean_text(str(pick(entry, "title") or "")) or None
        job_url = pick(entry, "jobUrl")
        apply_url = pick(entry, "applyUrl")
        if not title and not job_url:
            return None

        locations = [str(pick(entry, "location") or "")]
        for secondary in pick(entry, "secondaryLocations", default=[]) or []:
            if isinstance(secondary, dict):
                label = pick(secondary, "location", "name")
                if label:
                    locations.append(str(label))
            elif secondary:
                locations.append(str(secondary))
        location_raw = "; ".join(dict.fromkeys(loc for loc in locations if loc.strip()))

        description = html_to_text(str(pick(entry, "descriptionHtml") or "")) or clean_text(
            str(pick(entry, "descriptionPlain") or "")
        )

        published_raw = pick(entry, "publishedAt", "publishedDate")
        updated_raw = pick(entry, "updatedAt")
        posted_at, date_source = _board_stamp(published_raw, DateSource.POSTED_DATE)
        updated_at = None
        if posted_at is None:
            updated_at, updated_source = _board_stamp(updated_raw, DateSource.UPDATED_DATE)
            if updated_at is not None:
                date_source = updated_source

        remote_raw = pick(entry, "workplaceType")
        if remote_raw is None and pick(entry, "isRemote") is True:
            remote_raw = "Remote"

        official = _board_url(job_url, "ashbyhq.com") or _board_url(apply_url, "ashbyhq.com")
        raw_id = pick(entry, "id", "jobId")
        official_id = str(raw_id) if raw_id and official else (extract_job_id(official) if official else None)
        chosen = official or (str(job_url) if job_url else (str(apply_url) if apply_url else None))

        return self._posting(
            company_name=company.name,
            title=title,
            location_raw=location_raw or None,
            description=description or None,
            employment_type_raw=str(pick(entry, "employmentType") or "") or None,
            remote_type_raw=str(remote_raw) if remote_raw else None,
            job_id=official_id,
            apply_url=chosen,
            alternate_urls=[str(apply_url)] if apply_url and official and str(apply_url) != official else [],
            posted_at_raw=str(published_raw) if published_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance={
                "ats": "ashby",
                "handle": handle,
                "discovery_method": "api",
                "discovered_from": ["ashby"],
                "source_job_id": str(raw_id) if raw_id else "",
                "official_job_id": official_id or "",
                "official_url": official or "",
                "official_source": "ashby" if official else "",
            },
        )


def _board_stamp(raw: object, source: DateSource) -> tuple[object, DateSource]:
    if raw is None or (isinstance(raw, str) and _AMBIGUOUS_DATE.search(raw)):
        return None, DateSource.UNKNOWN
    parsed = parse_datetime(raw)
    if parsed is None:
        return None, DateSource.UNKNOWN
    return parsed, source


def _board_url(url: object, host_suffix: str) -> str | None:
    if not url or not is_http_url(str(url)):
        return None
    text = str(url)
    if host_suffix not in host_of(text):
        return None
    if extract_job_id(text) is None:
        return None
    return text

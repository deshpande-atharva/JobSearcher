"""Lever job board adapter.

Uses Lever's public postings API:
``https://api.lever.co/v0/postings/{company}?mode=json``

``company`` is the handle from the public ``jobs.lever.co/{handle}`` board URL.
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

__all__ = ["LeverSource"]

API_TEMPLATE = "https://api.lever.co/v0/postings/{handle}"
_AMBIGUOUS_DATE = re.compile(r"\b(found|discovered|listed|crawled|seen|indexed)\b", re.IGNORECASE)


class LeverSource(DiscoverySource):
    name: ClassVar[str] = "lever"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "lever"

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
                API_TEMPLATE.format(handle=handle), params={"mode": "json"}
            )

        # Lever returns a bare array; some proxies wrap it in {"data": [...]}.
        entries = payload if isinstance(payload, list) else pick(payload, "data", "postings")
        if not isinstance(entries, list):
            raise SourceError(f"unexpected Lever payload for handle {handle!r}")

        postings = collect_postings(
            entries,
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
        settings = self.config.settings.discovery.sources.lever
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

        title = clean_text(str(pick(entry, "text", "title") or "")) or None
        # `hostedUrl` is the canonical posting page; `applyUrl` jumps straight to
        # the form. Prefer the posting page and keep the other as a fallback.
        hosted_url = pick(entry, "hostedUrl")
        apply_url = pick(entry, "applyUrl")
        if not title and not hosted_url:
            return None

        categories = pick(entry, "categories", default={}) or {}
        location_raw = clean_text(str(pick(categories, "location", "allLocations") or ""))
        if not location_raw:
            all_locations = pick(entry, "allLocations", default=[]) or []
            if isinstance(all_locations, list):
                location_raw = "; ".join(str(item) for item in all_locations if item)
        commitment = pick(categories, "commitment")
        workplace_type = pick(entry, "workplaceType")

        description = html_to_text(str(pick(entry, "descriptionPlain", "description") or ""))
        # Requirement/benefit bullets live in `lists`; they carry the experience
        # wording the seniority agent needs.
        extra_sections = []
        for section in pick(entry, "lists", default=[]) or []:
            if not isinstance(section, dict):
                continue
            heading = clean_text(str(pick(section, "text") or ""))
            body = html_to_text(str(pick(section, "content") or ""))
            if heading or body:
                extra_sections.append(f"{heading}\n{body}".strip())
        additional = html_to_text(str(pick(entry, "additionalPlain", "additional") or ""))
        full_description = "\n\n".join(part for part in [description, *extra_sections, additional] if part)

        created_raw = pick(entry, "createdAt")
        updated_raw = pick(entry, "updatedAt")
        posted_at, date_source = _board_stamp(created_raw, DateSource.POSTED_DATE)
        updated_at = None
        if posted_at is None:
            updated_at, updated_source = _board_stamp(updated_raw, DateSource.UPDATED_DATE)
            if updated_at is not None:
                date_source = updated_source

        official = _board_url(hosted_url, "lever.co") or _board_url(apply_url, "lever.co")
        raw_id = pick(entry, "id")
        official_id = str(raw_id) if raw_id and official else (extract_job_id(official) if official else None)
        chosen = official or (str(hosted_url) if hosted_url else (str(apply_url) if apply_url else None))

        return self._posting(
            company_name=company.name,
            title=title,
            location_raw=location_raw or None,
            description=full_description or None,
            employment_type_raw=str(commitment) if commitment else None,
            remote_type_raw=str(workplace_type) if workplace_type else None,
            job_id=official_id,
            apply_url=chosen,
            alternate_urls=[str(apply_url)] if apply_url and official and str(apply_url) != official else [],
            posted_at_raw=str(created_raw) if created_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance={
                "ats": "lever",
                "handle": handle,
                "discovery_method": "api",
                "discovered_from": ["lever"],
                "source_job_id": str(raw_id) if raw_id else "",
                "official_job_id": official_id or "",
                "official_url": official or "",
                "official_source": "lever" if official else "",
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

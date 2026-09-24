"""Lever job board adapter.

Uses Lever's public postings API:
``https://api.lever.co/v0/postings/{company}?mode=json``

``company`` is the handle from the public ``jobs.lever.co/{handle}`` board URL.
"""

from __future__ import annotations

from typing import Any, ClassVar

from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.sources.base import DiscoverySource, SourceError
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import pick
from src.utils.dates import parse_datetime
from src.utils.normalization import clean_text, html_to_text

__all__ = ["LeverSource"]

API_TEMPLATE = "https://api.lever.co/v0/postings/{handle}"


class LeverSource(DiscoverySource):
    name: ClassVar[str] = "lever"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "lever"

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
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

        postings = [self._to_posting(entry, company, handle) for entry in entries]
        return [posting for posting in postings if posting is not None]

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
        posted_at = parse_datetime(created_raw)
        updated_at = parse_datetime(updated_raw)
        if posted_at is not None:
            date_source = DateSource.POSTED_DATE
        elif updated_at is not None:
            date_source = DateSource.UPDATED_DATE
        else:
            date_source = DateSource.UNKNOWN

        job_id = pick(entry, "id")

        return self._posting(
            company_name=company.name,
            title=title,
            location_raw=location_raw or None,
            description=full_description or None,
            employment_type_raw=str(commitment) if commitment else None,
            remote_type_raw=str(workplace_type) if workplace_type else None,
            job_id=str(job_id) if job_id else None,
            apply_url=str(hosted_url) if hosted_url else (str(apply_url) if apply_url else None),
            alternate_urls=[str(apply_url)] if apply_url and hosted_url else [],
            posted_at_raw=str(created_raw) if created_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance={"ats": "lever", "handle": handle},
        )

"""SmartRecruiters adapter.

Uses the public Posting API:
``https://api.smartrecruiters.com/v1/companies/{identifier}/postings``

The list endpoint returns metadata without the job ad body, so descriptions are
fetched per posting from ``/postings/{id}``. That second call is bounded by
``detail_fetch_limit`` because seniority and sponsorship analysis need the text,
but not at the cost of thousands of requests against one company.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from src.models.config import CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.sources.base import DiscoverySource, SourceError
from src.sources.fixtures import FixtureStore, slugify
from src.sources.parsing import pick
from src.utils.dates import parse_datetime
from src.utils.normalization import clean_text, html_to_text

__all__ = ["SmartRecruitersSource"]

LIST_TEMPLATE = "https://api.smartrecruiters.com/v1/companies/{identifier}/postings"
DETAIL_TEMPLATE = "https://api.smartrecruiters.com/v1/companies/{identifier}/postings/{posting_id}"
PUBLIC_TEMPLATE = "https://jobs.smartrecruiters.com/{identifier}/{posting_id}"

PAGE_SIZE = 100
MAX_PAGES = 10


class SmartRecruitersSource(DiscoverySource):
    name: ClassVar[str] = "smartrecruiters"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "smartrecruiters"

    #: Descriptions are needed for classification, but cap the fan-out.
    detail_fetch_limit: int = 60

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        if company is None or not company.ats_identifier:
            return []
        identifier = company.ats_identifier.strip().strip("/")

        entries = (
            self._fixture_entries(company, identifier)
            if self.ctx.fixture_mode
            else await self._fetch_all_pages(identifier)
        )

        postings = [self._to_posting(entry, company, identifier) for entry in entries]
        results = [posting for posting in postings if posting is not None]

        if not self.ctx.fixture_mode:
            await self._enrich_descriptions(results, identifier)
        return results

    def _fixture_entries(self, company: CompanyConfig, identifier: str) -> list[Any]:
        payload = FixtureStore(self.config.fixture_dir).load_first(
            self.name, [f"{slugify(company.name)}.json", f"{slugify(identifier)}.json"]
        )
        entries = pick(payload, "content", "postings", default=None)
        return entries if isinstance(entries, list) else []

    async def _fetch_all_pages(self, identifier: str) -> list[Any]:
        collected: list[Any] = []
        for page in range(MAX_PAGES):
            payload = await self.http.get_json(
                LIST_TEMPLATE.format(identifier=identifier),
                params={"limit": PAGE_SIZE, "offset": page * PAGE_SIZE},
            )
            entries = pick(payload, "content", default=None)
            if entries is None and page == 0:
                raise SourceError(f"unexpected SmartRecruiters payload for {identifier!r}")
            if not isinstance(entries, list) or not entries:
                break
            collected.extend(entries)
            total = pick(payload, "totalFound", default=None)
            if isinstance(total, int) and len(collected) >= total:
                break
            if len(entries) < PAGE_SIZE:
                break
        return collected

    def _to_posting(
        self, entry: Any, company: CompanyConfig, identifier: str
    ) -> RawJobPosting | None:
        if not isinstance(entry, dict):
            return None

        title = clean_text(str(pick(entry, "name", "title") or "")) or None
        posting_id = pick(entry, "id", "uuid")
        if not title and not posting_id:
            return None

        location = pick(entry, "location", default={}) or {}
        parts = [
            str(pick(location, "city") or ""),
            str(pick(location, "region") or ""),
            str(pick(location, "country") or "").upper(),
        ]
        location_raw = ", ".join(part for part in parts if part.strip())
        remote_raw = "Remote" if pick(location, "remote") is True else None

        employment = pick(entry, "typeOfEmployment", default={}) or {}
        employment_raw = (
            str(pick(employment, "label") or "")
            if isinstance(employment, dict)
            else str(employment or "")
        )

        released_raw = pick(entry, "releasedDate", "createdOn")
        updated_raw = pick(entry, "updatedOn", "lastUpdated")
        posted_at = parse_datetime(released_raw)
        updated_at = parse_datetime(updated_raw)
        if posted_at is not None:
            date_source = DateSource.POSTED_DATE
        elif updated_at is not None:
            date_source = DateSource.UPDATED_DATE
        else:
            date_source = DateSource.UNKNOWN

        # `refNumber` is the human-facing requisition id when present; the
        # posting id is always available and is what the public URL uses.
        reference = pick(entry, "refNumber")

        apply_url = PUBLIC_TEMPLATE.format(identifier=identifier, posting_id=posting_id)

        return self._posting(
            company_name=company.name,
            title=title,
            location_raw=location_raw or None,
            employment_type_raw=employment_raw or None,
            remote_type_raw=remote_raw,
            job_id=str(reference or posting_id),
            apply_url=apply_url,
            posted_at_raw=str(released_raw) if released_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance={
                "ats": "smartrecruiters",
                "identifier": identifier,
                "posting_id": str(posting_id),
            },
        )

    async def _enrich_descriptions(
        self, postings: list[RawJobPosting], identifier: str
    ) -> None:
        """Fetch job-ad bodies concurrently for the first N postings."""
        targets = postings[: self.detail_fetch_limit]
        if not targets:
            return
        if len(postings) > self.detail_fetch_limit:
            self.log.info(
                "limiting SmartRecruiters detail fetches",
                identifier=identifier,
                total=len(postings),
                fetching=self.detail_fetch_limit,
            )

        async def load(posting: RawJobPosting) -> None:
            posting_id = posting.provenance.get("posting_id")
            if not posting_id:
                return
            try:
                payload = await self.http.get_json(
                    DETAIL_TEMPLATE.format(identifier=identifier, posting_id=posting_id)
                )
            except Exception as exc:
                # A single unreadable posting is not a company failure.
                self.log.debug("smartrecruiters detail fetch failed", job_id=posting_id, error=str(exc))
                return
            posting.description = _flatten_job_ad(payload)

        await asyncio.gather(*(load(posting) for posting in targets), return_exceptions=True)


def _flatten_job_ad(payload: Any) -> str | None:
    """Concatenate the job-ad sections into one plain-text description."""
    job_ad = pick(payload, "jobAd", default={}) or {}
    sections = pick(job_ad, "sections", default={}) or {}
    if not isinstance(sections, dict):
        return None

    ordered_keys = ("companyDescription", "jobDescription", "qualifications", "additionalInformation")
    chunks: list[str] = []
    for key in (*ordered_keys, *[k for k in sections if k not in ordered_keys]):
        section = sections.get(key)
        if not isinstance(section, dict):
            continue
        title = clean_text(str(pick(section, "title") or ""))
        body = html_to_text(str(pick(section, "text") or ""))
        if body:
            chunks.append(f"{title}\n{body}".strip() if title else body)

    return "\n\n".join(chunks) or None

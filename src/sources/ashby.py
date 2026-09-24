"""Ashby job board adapter.

Uses Ashby's public posting API:
``https://api.ashbyhq.com/posting-api/job-board/{jobBoardName}``

``jobBoardName`` is the handle from the public ``jobs.ashbyhq.com/{handle}``
board URL. No API key is required for the public board endpoint.
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

__all__ = ["AshbySource"]

API_TEMPLATE = "https://api.ashbyhq.com/posting-api/job-board/{handle}"


class AshbySource(DiscoverySource):
    name: ClassVar[str] = "ashby"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "ashby"

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
                API_TEMPLATE.format(handle=handle),
                params={"includeCompensation": "true"},
            )

        jobs = pick(payload, "jobs", default=None)
        if not isinstance(jobs, list):
            raise SourceError(f"unexpected Ashby payload for board {handle!r}")

        postings = [self._to_posting(entry, company, handle) for entry in jobs]
        return [posting for posting in postings if posting is not None]

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
        posted_at = parse_datetime(published_raw)
        updated_at = parse_datetime(updated_raw)
        if posted_at is not None:
            date_source = DateSource.POSTED_DATE
        elif updated_at is not None:
            date_source = DateSource.UPDATED_DATE
        else:
            date_source = DateSource.UNKNOWN

        remote_raw = pick(entry, "workplaceType")
        if remote_raw is None and pick(entry, "isRemote") is True:
            remote_raw = "Remote"

        job_id = pick(entry, "id", "jobId")

        return self._posting(
            company_name=company.name,
            title=title,
            location_raw=location_raw or None,
            description=description or None,
            employment_type_raw=str(pick(entry, "employmentType") or "") or None,
            remote_type_raw=str(remote_raw) if remote_raw else None,
            job_id=str(job_id) if job_id else None,
            apply_url=str(job_url) if job_url else (str(apply_url) if apply_url else None),
            alternate_urls=[str(apply_url)] if apply_url and job_url else [],
            posted_at_raw=str(published_raw) if published_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance={"ats": "ashby", "handle": handle},
        )

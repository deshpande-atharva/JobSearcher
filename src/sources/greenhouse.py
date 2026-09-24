"""Greenhouse job board adapter.

Uses Greenhouse's public, documented Job Board API:
``https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true``

``token`` is the board identifier from the company's public board URL and comes
from ``ats_identifier`` in ``config/companies.yaml``. Nothing is guessed: an
unknown or wrong token yields an HTTP error that is reported as a failed company.
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

__all__ = ["GreenhouseSource"]

API_TEMPLATE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


class GreenhouseSource(DiscoverySource):
    name: ClassVar[str] = "greenhouse"
    scope: ClassVar[str] = "company"
    ats_type: ClassVar[str | None] = "greenhouse"

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        if company is None or not company.ats_identifier:
            return []
        token = company.ats_identifier.strip().strip("/")

        if self.ctx.fixture_mode:
            payload = FixtureStore(self.config.fixture_dir).load_first(
                self.name, [f"{slugify(company.name)}.json", f"{slugify(token)}.json"]
            )
            if payload is None:
                self.log.debug("no greenhouse fixture", company=company.name)
                return []
        else:
            payload = await self.http.get_json(
                API_TEMPLATE.format(token=token), params={"content": "true"}
            )

        jobs = pick(payload, "jobs", default=None)
        if not isinstance(jobs, list):
            raise SourceError(f"unexpected Greenhouse payload for board {token!r}")

        postings = [self._to_posting(entry, company, token) for entry in jobs]
        return [posting for posting in postings if posting is not None]

    def _to_posting(
        self, entry: Any, company: CompanyConfig, token: str
    ) -> RawJobPosting | None:
        if not isinstance(entry, dict):
            return None

        title = clean_text(str(pick(entry, "title") or "")) or None
        apply_url = pick(entry, "absolute_url")
        if not title and not apply_url:
            return None

        job_id = pick(entry, "id", "internal_job_id")
        location = pick(entry, "location", default={})
        location_raw = (
            clean_text(str(pick(location, "name") or ""))
            if isinstance(location, dict)
            else clean_text(str(location or ""))
        )
        if not location_raw:
            offices = pick(entry, "offices", default=[]) or []
            names = [
                str(pick(office, "name"))
                for office in offices
                if isinstance(office, dict) and pick(office, "name")
            ]
            location_raw = "; ".join(names)

        # `content` is HTML-escaped by the API; html_to_text unescapes and
        # flattens it. The markup is never rendered or executed.
        description = html_to_text(str(pick(entry, "content") or ""))

        published_raw = pick(entry, "first_published", "created_at")
        updated_raw = pick(entry, "updated_at")
        posted_at = parse_datetime(published_raw)
        updated_at = parse_datetime(updated_raw)

        if posted_at is not None:
            date_source = DateSource.POSTED_DATE
        elif updated_at is not None:
            date_source = DateSource.UPDATED_DATE
        else:
            date_source = DateSource.UNKNOWN

        # Greenhouse exposes free-form board metadata; employment type is often
        # in there, but only under an inconsistent field name.
        employment_type_raw = None
        for meta in pick(entry, "metadata", default=[]) or []:
            if not isinstance(meta, dict):
                continue
            label = str(pick(meta, "name") or "").lower()
            if "employment" in label or "job type" in label or "type" == label:
                value = pick(meta, "value")
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                if value:
                    employment_type_raw = str(value)
                    break

        return self._posting(
            company_name=company.name,
            title=title,
            location_raw=location_raw or None,
            description=description or None,
            employment_type_raw=employment_type_raw,
            job_id=str(job_id) if job_id else None,
            apply_url=str(apply_url) if apply_url else None,
            posted_at_raw=str(published_raw) if published_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance={"ats": "greenhouse", "board_token": token},
        )

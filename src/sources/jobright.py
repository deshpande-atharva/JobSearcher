"""Jobright discovery adapter.

Primary listing: ``https://jobright.ai/entry-level-jobs``.

Jobright does not publish a documented public API, so this adapter only reads
the public listing page (static HTML first, optional Playwright render second).
Jobright is a *discovery* source: aggregator URLs it produces are never treated
as the final application link.

The scraper is isolated here so a site-structure change cannot leak into the
rest of the pipeline.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from src.models.config import AppConfig, CompanyConfig
from src.models.job import DateSource, RawJobPosting
from src.sources.base import DiscoverySource, SourceError, SourceResult
from src.sources.fixtures import FixtureStore
from src.sources.parsing import (
    extract_embedded_json_blobs,
    extract_job_links,
    extract_json_ld_job_postings,
    extract_next_data,
    iter_dicts,
    json_ld_to_raw_posting,
    pick,
)
from src.utils.dates import parse_datetime
from src.utils.normalization import clean_text, normalize_company_name
from src.utils.urls import (
    UrlKind,
    classify_url,
    extract_job_id,
    host_of,
    is_generic_careers_page,
    is_http_url,
    unwrap_redirect,
)

__all__ = ["JobrightSource", "ambiguous_jobright_date", "parse_jobright_cards"]

_CARD_LINK = re.compile(
    r'<a[^>]+href="(https://jobright.ai/jobs/info/([^"?]+))[^"]*"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_RELATIVE_TIME = re.compile(r"^\d+\s+(?:minute|hour|day|week|month)s?\s+ago$", re.IGNORECASE)
_LABELED_POSTED = re.compile(r"^posted\s+(.+)$", re.IGNORECASE)

_AMBIGUOUS_DATE = re.compile(
    r"\b(found|discovered|listed|crawled|seen|indexed)\b",
    re.IGNORECASE,
)


class JobrightSource(DiscoverySource):
    name: ClassVar[str] = "jobright"
    scope: ClassVar[str] = "global"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.cap_reached = False
        self.cap_note = ""
        self.queries_attempted = 0
        self.detail_blocked = False

    @property
    def enabled(self) -> bool:
        # Offline fixtures keep exercising this source. Live runs follow the flag.
        if self.ctx.fixture_mode:
            return True
        return super().enabled

    async def discover(self, company: CompanyConfig | None = None) -> list[RawJobPosting]:
        if company is not None:
            return []

        settings = self.config.settings.discovery.sources.jobright
        if settings.max_queries < 1:
            self.cap_reached = True
            self.cap_note = "max_queries=0"
            return []
        if self.ctx.fixture_mode:
            postings = self._from_fixture()
            return self._limit(postings, settings.max_total_results, "fixture")

        self.queries_attempted = 1
        html = await self._fetch_listing(settings.entry_url)
        if _looks_blocked(html):
            raise SourceError("Jobright returned a blocked or CAPTCHA page", status="BLOCKED")
        postings = self._parse_listing(html, settings.entry_url)
        if postings and "infinite-scroll" in html:
            self.cap_reached = True
            note = "listing_page_only further_results_not_loaded"
            self.cap_note = "; ".join(item for item in (self.cap_note, note) if item)

        if not postings and settings.allow_browser_render and settings.max_navigation_attempts > 0:
            rendered = await self._render(settings.entry_url)
            if rendered:
                if _looks_blocked(rendered):
                    raise SourceError(
                        "Jobright Playwright render was blocked or showed a CAPTCHA",
                        status="BLOCKED",
                    )
                postings = self._parse_listing(rendered, settings.entry_url)
            else:
                self.log.info(
                    "jobright listing has no machine-readable jobs; "
                    "Playwright render unavailable or returned nothing"
                )

        postings = [item for item in postings if _company_selected(item, self.config)]
        postings = self._limit(postings, min(settings.max_results_per_query, settings.max_total_results), "listing")
        if settings.max_detail_pages and postings:
            await self._probe_detail(postings[0])
        return postings

    async def discover_result(self, company: CompanyConfig | None = None) -> SourceResult:
        result = await super().discover_result(company)
        if self.cap_reached:
            result.diagnostics["cap_reached"] = True
            result.diagnostics["cap_note"] = self.cap_note
            result.diagnostics["source_list_cap_reached"] = True
        result.diagnostics["queries_attempted"] = self.queries_attempted
        if self.detail_blocked:
            result.diagnostics["detail_blocked"] = True
            result.diagnostics["cap_note"] = self.cap_note
        return result

    def _limit(self, postings: list[RawJobPosting], limit: int, label: str) -> list[RawJobPosting]:
        if len(postings) > limit:
            self.cap_reached = True
            self.cap_note = f"{label}_cap={limit} discovered_before_cap={len(postings)}"
            return postings[:limit]
        return postings

    def _from_fixture(self) -> list[RawJobPosting]:
        store = FixtureStore(self.config.fixture_dir)
        payload = store.load_first(self.name, ["entry-level.json", "jobs.json"])
        if payload is None:
            self.log.debug("no jobright fixture")
            return []
        entries = payload if isinstance(payload, list) else pick(payload, "jobs", "items", default=[])
        if not isinstance(entries, list):
            raise SourceError("unexpected Jobright fixture shape")
        return [p for p in (self._entry_to_posting(e, page_url="fixture://jobright") for e in entries) if p]

    async def _fetch_listing(self, url: str) -> str:
        result = await self.http.get_text(url)
        if result.blocked_by_robots:
            raise SourceError(
                "Jobright listing is disallowed by robots.txt",
                status="BLOCKED",
                http_status=result.status,
            )
        if not result.ok:
            raise SourceError(
                f"Jobright listing fetch failed: {result.error or result.status}",
                http_status=result.status,
            )
        return result.text

    async def _probe_detail(self, posting: RawJobPosting) -> None:
        """One public detail fetch. A 403 stops the path; it is not retried or bypassed."""
        url = str((posting.provenance or {}).get("source_url") or posting.apply_url or "")
        if "jobright.ai/jobs/info/" not in url:
            return
        result = await self.http.get_text(url.split("?", 1)[0])
        blocked = result.blocked_by_robots or result.status in {401, 403} or _looks_blocked(result.text or "")
        if not blocked:
            return
        self.detail_blocked = True
        note = "detail_blocked status={}".format(result.status or "robots")
        self.cap_note = "; ".join(item for item in (self.cap_note, note) if item)

    async def _render(self, url: str) -> str | None:
        from src.sources.playwright_renderer import PlaywrightRenderer

        renderer = PlaywrightRenderer(self.config)
        if not renderer.enabled:
            return None
        self.log.info("rendering Jobright listing with Playwright")
        return await renderer.render(url)

    def _parse_listing(self, html: str, page_url: str) -> list[RawJobPosting]:
        cards = [
            posting
            for posting in (
                self._entry_to_posting(card, page_url=page_url) for card in parse_jobright_cards(html)
            )
            if posting is not None
        ]
        if cards:
            return cards
        postings: list[RawJobPosting] = []
        seen: set[str] = set()

        def add(posting: RawJobPosting | None) -> None:
            if posting is None:
                return
            posting = self._prefer_official(posting)
            key = (posting.apply_url or "") + "|" + (posting.title or "") + "|" + (posting.company_name or "")
            if key in seen:
                return
            seen.add(key)
            postings.append(posting)

        for block in extract_json_ld_job_postings(html):
            add(json_ld_to_raw_posting(block, source=self.name, page_url=page_url))

        next_data = extract_next_data(html)
        if next_data is not None:
            for node in iter_dicts(next_data):
                if _looks_like_job(node):
                    add(self._entry_to_posting(node, page_url=page_url))

        for blob in extract_embedded_json_blobs(html):
            for node in iter_dicts(blob):
                if _looks_like_job(node):
                    add(self._entry_to_posting(node, page_url=page_url))

        if not postings:
            for url, text in extract_job_links(html, page_url):
                add(
                    self._posting(
                        title=text or None,
                        apply_url=url,
                        provenance={"parser": "link-harvest", "page_url": page_url},
                    )
                )
        return postings

    def _entry_to_posting(self, entry: Any, *, page_url: str) -> RawJobPosting | None:
        if not isinstance(entry, dict):
            return None
        title = clean_text(str(pick(entry, "title", "jobTitle", "name", "position") or "")) or None
        company = clean_text(str(pick(entry, "company", "companyName", "employer") or "")) or None
        if isinstance(pick(entry, "company"), dict):
            company = clean_text(str(pick(pick(entry, "company"), "name") or "")) or company

        apply_url = _first_url(entry, "applyUrl", "applicationUrl", "url", "link", "jobUrl", "canonicalUrl")
        alternates = []
        for key in ("sourceUrl", "originalUrl", "redirectUrl", "externalUrl"):
            extra = _first_url(entry, key)
            if extra and extra != apply_url:
                alternates.append(extra)

        if not title and not apply_url:
            return None

        location = pick(entry, "location", "jobLocation", "city")
        if isinstance(location, dict):
            location = pick(location, "name", "city", "label")
        location_raw = clean_text(str(location or "")) or None

        posted_raw = pick(entry, "postedAt", "datePosted", "publishedAt", "postedDate")
        updated_raw = pick(entry, "updatedAt", "dateModified")
        posted_at, date_source = _authoritative_stamp(posted_raw, DateSource.POSTED_DATE)
        updated_at = None
        if posted_at is None:
            updated_at, updated_source = _authoritative_stamp(updated_raw, DateSource.UPDATED_DATE)
            if updated_at is not None:
                date_source = updated_source
        # An unlabeled "2 hours ago" on a search card is not an employer posted date.
        listed_ago = pick(entry, "listedAgo")
        source_job_id = pick(entry, "jobId", "id")
        official_hint = pick(entry, "requisitionId", "officialJobId", "atsJobId")
        description = clean_text(str(pick(entry, "description", "jobDescription", "snippet") or "")) or None
        official, aggregator = self._split_urls([apply_url, *alternates])
        official_job_id = extract_job_id(official) if official else None
        if not official_job_id and official_hint and not _looks_like_source_id(official_hint):
            official_job_id = str(official_hint)
        chosen = official or aggregator
        provenance = {
            "parser": "jobright",
            "page_url": page_url,
            "discovery_method": "jobright",
            "discovered_from": ["jobright"],
            "source_url": aggregator or page_url,
            "source_job_id": str(source_job_id) if source_job_id else "",
            "official_job_id": official_job_id or "",
            "official_source": _official_source(official),
            "date_evidence": str(posted_raw or updated_raw or listed_ago or ""),
        }
        return self._posting(
            company_name=company,
            title=title,
            location_raw=location_raw,
            description=description,
            employment_type_raw=str(pick(entry, "employmentType", "jobType") or "") or None,
            remote_type_raw=str(pick(entry, "workplaceType", "remoteType") or "") or None,
            job_id=official_job_id,
            apply_url=chosen,
            alternate_urls=[official] if official and aggregator and official != chosen else [],
            posted_at_raw=str(posted_raw) if posted_raw else None,
            updated_at_raw=str(updated_raw) if updated_raw else None,
            posted_at=posted_at,
            updated_at=updated_at,
            date_source=date_source,
            provenance=provenance,
        )

    def _split_urls(self, urls: list[str | None]) -> tuple[str | None, str | None]:
        policy = self.config.url_policy
        official = None
        aggregator = None
        for raw in urls:
            if not raw:
                continue
            for candidate in (unwrap_redirect(raw), raw):
                if not is_http_url(candidate):
                    continue
                host = host_of(candidate)
                verdict = classify_url(candidate, policy)
                if verdict.kind is UrlKind.AGGREGATOR or host.endswith("jobright.ai"):
                    aggregator = aggregator or candidate
                    continue
                if verdict.acceptable_as_final and not is_generic_careers_page(candidate):
                    official = official or candidate
        return official, aggregator

    def _prefer_official(self, posting: RawJobPosting) -> RawJobPosting:
        official, aggregator = self._split_urls([posting.apply_url, *posting.alternate_urls])
        provenance = dict(posting.provenance or {})
        from_url = extract_job_id(official) if official else None
        hint = str(provenance.get("official_job_id") or "")
        if _looks_like_source_id(hint):
            hint = ""
        official_id = from_url or hint or None
        provenance["discovery_method"] = "jobright"
        found = [str(item) for item in provenance.get("discovered_from") or [] if item]
        if "jobright" not in found:
            found.append("jobright")
        provenance["discovered_from"] = found
        if aggregator:
            provenance["source_url"] = aggregator
        elif not provenance.get("source_url"):
            provenance["source_url"] = str(provenance.get("page_url") or "")
        if posting.job_id and posting.job_id != official_id and not provenance.get("source_job_id"):
            provenance["source_job_id"] = str(posting.job_id)
        provenance["official_job_id"] = official_id or str(provenance.get("official_job_id") or "")
        provenance["official_source"] = _official_source(official)
        return posting.model_copy(
            update={
                "apply_url": official or aggregator or posting.apply_url,
                "alternate_urls": [],
                "job_id": official_id,
                "provenance": provenance,
            }
        )


def ambiguous_jobright_date(value: object) -> bool:
    return bool(_AMBIGUOUS_DATE.search(str(value or "")))


def _authoritative_stamp(raw: object, source: DateSource) -> tuple[object, DateSource]:
    if raw is None or ambiguous_jobright_date(raw):
        return None, DateSource.UNKNOWN
    parsed = parse_datetime(raw)
    if parsed is None:
        return None, DateSource.UNKNOWN
    return parsed, source


def _looks_like_source_id(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("jr-") or "jobright" in text


def _official_source(url: str | None) -> str:
    host = host_of(url)
    if "greenhouse.io" in host:
        return "greenhouse"
    if "myworkdayjobs.com" in host:
        return "workday"
    if "lever.co" in host:
        return "lever"
    if "ashbyhq.com" in host:
        return "ashby"
    return ""


def _company_selected(posting: RawJobPosting, config: AppConfig) -> bool:
    name = normalize_company_name(posting.company_name)
    if not name:
        return False
    allowed: set[str] = set()
    for company in config.target_companies():
        if not company.enabled:
            continue
        allowed.add(normalize_company_name(company.name))
        for alias in company.aliases:
            allowed.add(normalize_company_name(alias))
    return name in allowed


def _looks_like_job(node: dict[str, Any]) -> bool:
    keys = {str(k).lower() for k in node}
    title_keys = keys & {"title", "jobtitle", "name", "position"}
    url_keys = keys & {"applyurl", "applicationurl", "url", "joburl", "canonicalurl", "link"}
    id_keys = keys & {"jobid", "requisitionid", "id"}
    if not title_keys:
        return False
    return bool(url_keys or id_keys)


def _first_url(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = pick(entry, key)
        if isinstance(value, str) and is_http_url(value):
            return value.strip()
    return None


def parse_jobright_cards(html: str) -> list[dict[str, str]]:
    """Read public listing cards. Unlabeled recency is not a posted date."""
    cards: list[dict[str, str]] = []
    for match in _CARD_LINK.finditer(html or ""):
        href, source_id, inner = match.group(1), match.group(2), match.group(3)
        lines = _visible_lines(inner)
        company = next((line for line in lines if line not in {"·", "•", "View"}), "")
        title = ""
        for line in lines:
            if line in {company, "·", "•", "View"} or _RELATIVE_TIME.match(line) or line.startswith("$"):
                continue
            if _LABELED_POSTED.match(line):
                continue
            title = line
            break
        if not company or not title:
            continue
        location_bits = [
            line
            for line in lines
            if line not in {company, title, "·", "•", "View"}
            and not _RELATIVE_TIME.match(line)
            and not _LABELED_POSTED.match(line)
            and not line.startswith("$")
            and len(line) < 40
        ]
        description = next((line for line in lines if len(line) > 80), "")
        labeled = next((line for line in lines if _LABELED_POSTED.match(line)), "")
        unlabeled = next((line for line in lines if _RELATIVE_TIME.match(line)), "")
        card = {
            "company": company,
            "title": title,
            "location": ", ".join(location_bits[:3]),
            "jobId": source_id,
            "applyUrl": href.split("?", 1)[0],
            "description": description,
        }
        if labeled:
            card["postedAt"] = _LABELED_POSTED.match(labeled).group(1) if _LABELED_POSTED.match(labeled) else labeled
        elif unlabeled:
            card["listedAgo"] = unlabeled
        cards.append(card)
    return cards


def _visible_lines(html: str) -> list[str]:
    text = re.sub(r"<[^>]+>", "\n", html or "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if line and "css-" not in line and not line.startswith("__")]


def _looks_blocked(html: str) -> bool:
    lowered = (html or "").lower()
    return any(
        token in lowered
        for token in (
            "captcha",
            "cf-browser-verification",
            "access denied",
            "unusual traffic",
            "verify you are human",
        )
    )

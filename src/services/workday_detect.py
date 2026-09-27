"""Deterministic Workday tenant detection from a public URL or page HTML.

Signals are combined; a single CSS selector is never required. Identifiers are
taken from the URL or from Workday's own public embed, never guessed.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass

from src.services.ats_discovery import detect_ats, detect_ats_from_html
from src.sources.workday import parse_workday_site
from src.utils.urls import is_http_url

__all__ = ["WorkdayTenant", "detect_workday_tenant"]

_TENANT_JS = re.compile(r"""tenant\s*:\s*["']([A-Za-z0-9_-]+)["']""", re.I)
_SITE_JS = re.compile(r"""siteId\s*:\s*["']([A-Za-z0-9_-]+)["']""", re.I)
_CXS = re.compile(r"/wday/cxs/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", re.I)
_WD_HOST = re.compile(r"myworkday(?:jobs|site)\.com", re.I)
_WD_URL = re.compile(r"https?://[a-z0-9.-]+\.myworkday(?:jobs|site)\.com/[^\s\"'<>]+", re.I)


@dataclass(frozen=True, slots=True)
class WorkdayTenant:
    detected: bool
    company: str = ""
    careers_url: str = ""
    source_type: str = "WORKDAY"
    tenant: str = ""
    site: str = ""
    host: str = ""
    detection_method: str = "none"

    @property
    def ok(self) -> bool:
        return bool(self.detected and self.tenant and self.site and self.host)


def detect_workday_tenant(
    url: str | None,
    html: str | None = None,
    *,
    company: str = "",
) -> WorkdayTenant:
    """Identify a public Workday career site from its URL and optional markup."""
    empty = WorkdayTenant(detected=False, company=company, careers_url=url or "")
    parsed = parse_workday_site(url or "")
    if parsed is not None:
        host, tenant, site = parsed
        return WorkdayTenant(
            detected=True,
            company=company,
            careers_url=url or "",
            tenant=tenant,
            site=site,
            host=host,
            detection_method="url",
        )

    ats = detect_ats(url) if url else None
    if ats is not None and ats.ok and ats.ats_type == "workday" and ats.identifier:
        parsed = parse_workday_site(ats.identifier)
        if parsed is not None:
            host, tenant, site = parsed
            return WorkdayTenant(
                detected=True,
                company=company,
                careers_url=ats.identifier,
                tenant=tenant,
                site=site,
                host=host,
                detection_method="url",
            )

    markup = html_lib.unescape(html or "")
    if markup:
        html_hit = detect_ats_from_html(markup, page_url=url or "")
        if html_hit is not None and html_hit.ok and html_hit.ats_type == "workday" and html_hit.identifier:
            parsed = parse_workday_site(html_hit.identifier)
            if parsed is not None:
                host, tenant, site = parsed
                return WorkdayTenant(
                    detected=True,
                    company=company,
                    careers_url=html_hit.identifier,
                    tenant=tenant,
                    site=site,
                    host=host,
                    detection_method="html_embed",
                )
        cxs = _CXS.search(markup)
        if cxs:
            tenant, site = cxs.group(1), cxs.group(2)
            host = _host_from(url, markup)
            if host:
                return WorkdayTenant(
                    detected=True,
                    company=company,
                    careers_url=url or "",
                    tenant=tenant,
                    site=site,
                    host=host,
                    detection_method="html_cxs",
                )
        tenant_js = _TENANT_JS.search(markup)
        site_js = _SITE_JS.search(markup)
        if tenant_js and site_js:
            host = _host_from(url, markup)
            if host:
                return WorkdayTenant(
                    detected=True,
                    company=company,
                    careers_url=url or "",
                    tenant=tenant_js.group(1),
                    site=site_js.group(1),
                    host=host,
                    detection_method="html_script",
                )
        if _WD_HOST.search(markup) or _WD_URL.search(markup):
            return WorkdayTenant(
                detected=False,
                company=company,
                careers_url=url or "",
                detection_method="html_host_only",
            )
    return empty


def _host_from(url: str | None, html: str) -> str:
    if url and is_http_url(url):
        from urllib.parse import urlparse

        host = urlparse(url).netloc.lower()
        if _WD_HOST.search(host):
            return host
    match = _WD_URL.search(html)
    if match:
        from urllib.parse import urlparse

        host = urlparse(match.group(0)).netloc.lower()
        if host:
            return host
    return ""

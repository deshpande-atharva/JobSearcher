"""Direct application URL verification.

Aggregators are valid discovery sources and invalid final application URLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.config import AppConfig
from src.models.job import RawJobPosting
from src.sources.base import HttpClient
from src.utils.urls import (
    UrlKind,
    UrlPolicy,
    UrlVerdict,
    canonicalize_url,
    classify_url,
    extract_job_id,
    is_generic_careers_page,
    is_http_url,
    unwrap_redirect,
)

__all__ = ["UrlCheck", "pick_direct_url", "verify_url"]


@dataclass(slots=True)
class UrlCheck:
    accepted: bool
    url: str = ""
    verdict: UrlVerdict | None = None
    job_id: str | None = None
    reason: str = ""
    reachable: bool | None = None
    candidates_tried: list[str] = field(default_factory=list)


def pick_direct_url(
    posting: RawJobPosting,
    policy: UrlPolicy,
) -> UrlCheck:
    """Choose the best direct application URL from a raw posting.

    Never invents a URL. Unwraps aggregator redirect parameters when the
    destination is already present in the link.
    """
    candidates: list[str] = []
    for raw in (posting.apply_url, *posting.alternate_urls):
        if not raw:
            continue
        unwrapped = unwrap_redirect(raw)
        if unwrapped and unwrapped not in candidates:
            candidates.append(unwrapped)
        if raw not in candidates:
            candidates.append(raw)

    if not candidates:
        return UrlCheck(accepted=False, reason="no application URL provided")

    acceptable: list[tuple[str, UrlVerdict]] = []
    aggregators: list[tuple[str, UrlVerdict]] = []
    for url in candidates:
        verdict = classify_url(url, policy)
        if verdict.acceptable_as_final:
            acceptable.append((url, verdict))
        elif verdict.kind is UrlKind.AGGREGATOR:
            aggregators.append((url, verdict))

    if not acceptable:
        if aggregators:
            url, verdict = aggregators[0]
            return UrlCheck(
                accepted=False,
                url=url,
                verdict=verdict,
                reason="only aggregator/search URLs were available; they cannot be the final application link",
                candidates_tried=candidates,
            )
        return UrlCheck(
            accepted=False,
            url=candidates[0],
            reason="no valid http(s) application URL",
            candidates_tried=candidates,
        )

    # Prefer a posting URL that carries a job id over a generic careers homepage.
    acceptable.sort(
        key=lambda item: (
            0 if extract_job_id(item[0]) else 1,
            0 if item[1].kind is UrlKind.ATS else 1,
            1 if is_generic_careers_page(item[0]) else 0,
        )
    )
    url, verdict = acceptable[0]
    if is_generic_careers_page(url) and any(extract_job_id(u) for u, _ in acceptable[1:]):
        for other, other_verdict in acceptable[1:]:
            if extract_job_id(other):
                url, verdict = other, other_verdict
                break

    canonical = canonicalize_url(url, policy.strip_query_params)
    job_id = posting.job_id or extract_job_id(canonical) or extract_job_id(url)
    return UrlCheck(
        accepted=True,
        url=canonical or url,
        verdict=verdict,
        job_id=str(job_id) if job_id else None,
        reason=verdict.reason,
        candidates_tried=candidates,
    )


async def verify_url(
    check: UrlCheck,
    config: AppConfig,
    http: HttpClient | None,
) -> UrlCheck:
    """Optionally probe reachability. Transient failures do not reject ATS URLs."""
    if not check.accepted or not config.settings.urls.verify_reachability:
        return check
    if http is None or config.fixture_mode:
        check.reachable = None
        return check
    if not is_http_url(check.url):
        check.accepted = False
        check.reason = "final URL is not a valid http(s) URL"
        return check

    result = await http.head(check.url)
    if result.ok:
        check.reachable = True
        return check
    if result.blocked_by_robots:
        check.reachable = None
        return check

    check.reachable = False
    allow_unverified = config.settings.urls.allow_unverified_ats_urls
    kind = check.verdict.kind if check.verdict else UrlKind.UNKNOWN
    if allow_unverified and kind in (UrlKind.ATS, UrlKind.COMPANY_CAREER):
        check.reason = (
            f"{check.reason}; reachability probe failed ({result.error or result.status}) "
            "but the URL shape is a legitimate ATS/company posting"
        )
        return check

    check.accepted = False
    check.reason = f"application URL was not reachable: {result.error or result.status}"
    return check

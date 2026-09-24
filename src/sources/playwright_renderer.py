"""Optional Playwright renderer for public JS-heavy listing pages.

Used only as a fallback when static HTML contains no machine-readable job data.
This module never:

* spoofs a browser fingerprint
* solves or bypasses CAPTCHAs
* bypasses authentication
* evades rate limits or anti-bot controls

A 401/403, a CAPTCHA interstitial or a missing Playwright install is reported
as a failed render and the source continues without it.
"""

from __future__ import annotations

from src.models.config import AppConfig
from src.utils.logging import get_logger

__all__ = ["PlaywrightRenderer"]

log = get_logger(__name__)


class PlaywrightRenderer:
    """Headless Chromium renderer, constructed only when enabled."""

    def __init__(self, config: AppConfig) -> None:
        self._settings = config.settings.scraping.playwright
        self._enabled = self._settings.enabled and not config.fixture_mode
        self._timeout_ms = int(self._settings.timeout_seconds * 1000)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def render(self, url: str) -> str | None:
        """Return rendered HTML, or ``None`` when rendering is unavailable."""
        if not self._enabled:
            return None
        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeout
            from playwright.async_api import async_playwright
        except ImportError:
            log.warning("playwright is not installed; skipping browser render", url=url)
            return None

        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=self._settings.headless)
                try:
                    page = await browser.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
                    # Give the listing a brief moment to hydrate. We do not wait
                    # on networkidle -- that hangs on analytics-heavy sites.
                    await page.wait_for_timeout(1500)
                    return await page.content()
                finally:
                    await browser.close()
        except PlaywrightTimeout:
            log.info("playwright timed out", url=url)
            return None
        except Exception as exc:
            # Authentication walls, missing browsers, blocked launches, etc.
            safe_error = str(exc).encode("ascii", "replace").decode("ascii")[:300]
            log.info("playwright render failed", url=url, error=safe_error)
            return None

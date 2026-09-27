"""Optional Playwright renderer for public JS-heavy listing pages.

Used only as a fallback when static HTML contains no machine-readable job data.
One browser is reused for the run. This module never:

* spoofs a browser fingerprint
* solves or bypasses CAPTCHAs
* bypasses authentication
* evades rate limits or anti-bot controls

A 401/403, a CAPTCHA interstitial or a missing Playwright install is reported
as a failed render and the source continues without it.
"""

from __future__ import annotations

import asyncio
import time

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
        self._playwright: object | None = None
        self._browser: object | None = None
        self._lock = asyncio.Lock()
        self.render_count = 0
        self.render_failures = 0
        self.render_seconds = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def render(self, url: str) -> str | None:
        """Return rendered HTML, or ``None`` when rendering is unavailable."""
        if not self._enabled:
            return None
        started = time.perf_counter()
        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeout
        except ImportError:
            self.render_failures += 1
            log.warning("playwright is not installed; skipping browser render", url=url)
            return None

        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
                # Give the listing a brief moment to hydrate. We do not wait
                # on networkidle -- that hangs on analytics-heavy sites.
                await page.wait_for_timeout(1500)
                html = await page.content()
            finally:
                await page.close()
            self.render_count += 1
            from src.sources.base import note_effort

            note_effort("playwright_renders")
            return html
        except PlaywrightTimeout:
            self.render_failures += 1
            from src.sources.base import note_effort

            note_effort("playwright_renders")
            log.info("playwright timed out", url=url)
            return None
        except Exception as exc:
            self.render_failures += 1
            from src.sources.base import note_effort

            note_effort("playwright_renders")
            # Authentication walls, missing browsers, blocked launches, etc.
            safe_error = str(exc).encode("ascii", "replace").decode("ascii")[:300]
            log.info("playwright render failed", url=url, error=safe_error)
            return None
        finally:
            self.render_seconds += time.perf_counter() - started

    async def _ensure_browser(self) -> object:
        if self._browser is not None:
            return self._browser
        async with self._lock:
            if self._browser is not None:
                return self._browser
            from playwright.async_api import async_playwright

            playwright = await async_playwright().start()
            self._playwright = playwright
            self._browser = await playwright.chromium.launch(headless=self._settings.headless)
            return self._browser

    async def aclose(self) -> None:
        browser = self._browser
        playwright = self._playwright
        self._browser = None
        self._playwright = None
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                log.info("playwright browser close failed")
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                log.info("playwright stop failed")

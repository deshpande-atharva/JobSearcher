"""Playwright runtime limited to public career pages.

The model chooses a structured action. This runtime never evaluates
model-supplied JavaScript, never solves a CAPTCHA, and never signs in.
"""

from __future__ import annotations

from src.browser.actions import BrowserAction, validate_action
from src.browser.page_state import JobCard, PageElement, PageState
from src.utils.logging import get_logger

__all__ = ["BlockedPage", "PlaywrightBrowser"]

log = get_logger(__name__)

_CAPTCHA_MARKERS = ("captcha", "cf-browser-verification", "verify you are human")


class BlockedPage(RuntimeError):
    """The public page refused automated access. Do not retry around it."""


class PlaywrightBrowser:
    def __init__(self, *, headless: bool = True, timeout_ms: int = 20000) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._playwright = None
        self._browser = None
        self._page = None

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._page = await self._browser.new_page()

    async def aclose(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def open(self, url: str) -> PageState:
        if self._page is None:
            await self.start()
        assert self._page is not None
        response = await self._page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
        status = response.status if response is not None else 0
        if status in {401, 403, 429}:
            raise BlockedPage(f"BLOCKED HTTP {status} for {url}")
        return await self.get_page_state()

    async def settle(self) -> PageState:
        """Wait until public job links render. A timeout is an empty page, not a bypass."""
        assert self._page is not None
        try:
            await self._page.wait_for_selector(
                'a[href*="/positions/"], a[href*="/jobs/"], a[href*="/job/"], a[data-automation-id="jobTitle"]',
                timeout=min(self._timeout_ms, 15000),
            )
        except Exception:
            pass
        return await self.get_page_state()

    async def back(self) -> PageState:
        assert self._page is not None
        await self._page.go_back(timeout=self._timeout_ms)
        return await self.get_page_state()

    async def get_page_state(self) -> PageState:
        assert self._page is not None
        raw = await self._page.evaluate(_PAGE_STATE_JS)
        state = PageState.model_validate(raw)
        lowered = state.visible_text.lower()
        if any(marker in lowered for marker in _CAPTCHA_MARKERS):
            state.blocked = True
            state.block_reason = "captcha"
        return state

    async def execute(self, action: BrowserAction) -> PageState:
        state = await self.get_page_state()
        if state.blocked:
            raise BlockedPage(state.block_reason or "BLOCKED")
        action = validate_action(action, state)
        assert self._page is not None
        if action.action == "open":
            return await self.open(action.value or "")
        if action.action == "back":
            return await self.back()
        if action.action == "scroll":
            await self._page.mouse.wheel(0, 800)
            return await self.get_page_state()
        if action.action == "wait":
            await self._page.wait_for_timeout(500)
            return await self.get_page_state()
        selector = f'[data-agent-id="{action.target_id}"]'
        if action.action == "click":
            await self._page.locator(selector).click(timeout=self._timeout_ms)
        elif action.action == "type":
            await self._page.locator(selector).fill(action.value or "", timeout=self._timeout_ms)
        elif action.action == "submit":
            await self._page.locator(selector).press("Enter")
        elif action.action == "select":
            await self._page.locator(selector).select_option(action.value or "", timeout=self._timeout_ms)
        return await self.get_page_state()


_PAGE_STATE_JS = r"""
() => {
  const textOf = (el) => {
    const aria = (el.getAttribute('aria-label') || '').trim();
    const placeholder = (el.getAttribute('placeholder') || '').trim();
    const auto = (el.getAttribute('data-automation-id') || '').trim();
    const visible = (el.innerText || '').trim();
    return (aria || visible || placeholder || auto).slice(0, 160);
  };
  const stamp = (el, id) => { el.setAttribute('data-agent-id', id); return id; };
  const visible = (el) => {
    const box = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    return box.width > 0 && box.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  };
  const collect = (root) => {
    const found = [];
    const nodes = root.querySelectorAll('button, [role="button"], a, input, textarea, select, [data-automation-id]');
    for (const el of nodes) {
      const tag = el.tagName.toLowerCase();
      const auto = (el.getAttribute('data-automation-id') || '');
      const useful = /search|pagination|jobTitle|keyword|filter|location/i.test(auto);
      if (['button', 'a', 'input', 'textarea', 'select'].includes(tag) || useful || el.getAttribute('role') === 'button') {
        found.push(el);
      }
      if (el.shadowRoot) found.push(...collect(el.shadowRoot));
    }
    return found;
  };
  const buttons = [];
  const links = [];
  const inputs = [];
  const selects = [];
  let n = 0;
  for (const el of collect(document)) {
    if (!visible(el)) continue;
    const id = stamp(el, 'el_' + (++n));
    const text = textOf(el);
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') links.push({id, text, role: 'link', href: el.href || null, options: [], value: ''});
    else if (tag === 'input' || tag === 'textarea') inputs.push({id, text, role: el.getAttribute('type') || 'textbox', href: null, options: [], value: el.value || ''});
    else if (tag === 'select') selects.push({id, text, role: 'combobox', href: null, options: [...el.options].map(o => o.text), value: el.value || ''});
    else buttons.push({id, text, role: 'button', href: null, options: [], value: ''});
  }
  const cards = [];
  const seenJobs = new Set();
  const workdayId = (href) => {
    const req = href.match(/_((?:JR|R|REQ)-?\d{3,}(?:-\d+)?)(?:[/?#]|$)/i);
    if (req) return req[1];
    const query = href.match(/[?&](?:jobid|jobId|requisitionid)=([A-Za-z0-9._-]{3,64})/i);
    return query ? query[1] : '';
  };
  const nearbyLocation = (el) => {
    const row = el.closest('li, article, section, [role="listitem"]') || el.parentElement;
    if (!row) return '';
    const loc = row.querySelector('[data-automation-id*="ocation"], [data-automation-id*="locations"]');
    return loc ? textOf(loc) : '';
  };
  for (const el of collect(document).filter((item) => item.tagName && item.tagName.toLowerCase() === 'a' && item.href)) {
    const href = el.href || '';
    const greenhouse = href.match(/\/positions\/(\d+)|\/jobs\/(\d+)|[?&]gh_jid=(\d+)/i);
    if (greenhouse) {
      const jobId = greenhouse[1] || greenhouse[2] || greenhouse[3];
      if (!jobId || seenJobs.has(jobId)) continue;
      seenJobs.add(jobId);
      cards.push({id: el.getAttribute('data-agent-id') || '', title: textOf(el), company: '', location: nearbyLocation(el), url: href, job_id: jobId, description: ''});
      continue;
    }
    const workdayHost = /myworkday(?:jobs|site)\.com/i.test(href);
    if (workdayHost && /\/job\//i.test(href)) {
      const jobId = workdayId(href);
      const key = jobId || href;
      if (seenJobs.has(key)) continue;
      seenJobs.add(key);
      cards.push({id: el.getAttribute('data-agent-id') || '', title: textOf(el), company: '', location: nearbyLocation(el), url: href, job_id: jobId, description: ''});
    }
  }
  const body = (document.body && document.body.innerText || '').slice(0, 4000);
  return {url: location.href, title: document.title || '', visible_text: body, buttons, links, inputs, selects, job_cards: cards, blocked: false, block_reason: ''};
}
"""

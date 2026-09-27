"""In-memory browser used by tests and by the navigation loop."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from src.browser.actions import BrowserAction, validate_action
from src.browser.page_state import PageState

__all__ = ["BrowserSession", "InMemoryBrowser", "PageFixture"]


class PageFixture(BaseModel):
    id: str
    state: PageState
    transitions: dict[str, str] = Field(default_factory=dict)


class BrowserSession(Protocol):
    async def open(self, url: str) -> PageState: ...

    async def back(self) -> PageState: ...

    async def execute(self, action: BrowserAction) -> PageState: ...

    async def get_page_state(self) -> PageState: ...


class InMemoryBrowser:
    """Deterministic page graph. Clicks follow fixture transitions."""

    def __init__(self, pages: dict[str, PageFixture], start: str) -> None:
        self._pages = pages
        self._current = start
        self._history: list[str] = [start]
        self.opened: list[str] = []
        self.typed: list[tuple[str, str]] = []
        self.submitted: list[str] = []
        self.selected: list[tuple[str, str]] = []
        self.scrolls = 0

    async def open(self, url: str) -> PageState:
        self.opened.append(url)
        for page_id, page in self._pages.items():
            if page.state.url == url:
                self._current = page_id
                self._history.append(page_id)
                return await self.get_page_state()
        state = await self.get_page_state()
        return state.model_copy(update={"url": url})

    async def back(self) -> PageState:
        if len(self._history) > 1:
            self._history.pop()
            self._current = self._history[-1]
        return await self.get_page_state()

    async def get_page_state(self) -> PageState:
        return self._pages[self._current].state.model_copy(deep=True)

    async def execute(self, action: BrowserAction) -> PageState:
        page = await self.get_page_state()
        action = validate_action(action, page)
        if action.action == "open":
            return await self.open(action.value or "")
        if action.action == "back":
            return await self.back()
        if action.action == "scroll":
            self.scrolls += 1
            return page
        if action.action == "wait":
            return page
        if action.action == "type" and action.target_id and action.value is not None:
            self.typed.append((action.target_id, action.value))
            _set_value(self._pages[self._current].state, action.target_id, action.value)
            return await self.get_page_state()
        if action.action == "select" and action.target_id and action.value is not None:
            self.selected.append((action.target_id, action.value))
            _set_value(self._pages[self._current].state, action.target_id, action.value)
            return await self.get_page_state()
        if action.action == "submit" and action.target_id:
            self.submitted.append(action.target_id)
            target = self._pages[self._current].transitions.get(action.target_id)
            if target:
                self._current = target
                self._history.append(target)
            return await self.get_page_state()
        if action.action == "click" and action.target_id:
            target = self._pages[self._current].transitions.get(action.target_id)
            if target:
                self._current = target
                self._history.append(target)
            return await self.get_page_state()
        return page


def _set_value(state: PageState, element_id: str, value: str) -> None:
    element = state.find(element_id)
    if element is not None:
        element.value = value

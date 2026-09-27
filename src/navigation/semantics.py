"""Semantic labels for public career-site controls. Matching is exact-list, not free text generation."""

from __future__ import annotations

from src.browser.page_state import PageElement, PageState

__all__ = ["SEMANTIC_TARGETS", "find_by_text", "find_semantic"]

SEMANTIC_TARGETS: dict[str, tuple[str, ...]] = {
    "view_jobs": (
        "search jobs",
        "find jobs",
        "view jobs",
        "explore opportunities",
        "open positions",
        "see open roles",
        "search openings",
        "browse jobs",
        "see open jobs",
        "view open roles",
    ),
    "paginate": (
        "load more",
        "show more",
        "more jobs",
        "next",
        "next page",
        "view more",
        "go to page 2",
        "paginationnextbutton",
    ),
    "software_engineering": (
        "software engineering",
        "engineering",
        "software",
    ),
    "search_input": (
        "search",
        "search jobs",
        "search for jobs",
        "search for jobs or keywords",
        "job search",
        "keyword",
        "search openings",
        "keywordsearchinput",
        "keywordsearch",
    ),
    "search_submit": (
        "search",
        "search jobs",
        "find jobs",
        "find",
        "keywordsearchbutton",
        "searchbutton",
    ),
    "location_filter": (
        "location",
        "locations",
        "country",
        "united states",
        "locationfilter",
    ),
}


def find_by_text(page: PageState, text: str | None, *, prefer: str | None = None) -> PageElement | None:
    if not text:
        return None
    wanted = text.strip().lower()
    pools = {
        "button": [*page.buttons, *page.links],
        "input": list(page.inputs),
        "select": list(page.selects),
    }
    order = [prefer] if prefer in pools else []
    order.extend(name for name in ("button", "input", "select") if name not in order)
    for name in order:
        for element in pools[name]:
            if (element.text or "").strip().lower() == wanted:
                return element
    return None


def find_semantic(page: PageState, role: str | None) -> PageElement | None:
    phrases = SEMANTIC_TARGETS.get(role or "", ())
    if role == "search_input":
        pool = list(page.inputs)
    elif role == "location_filter":
        pool = [*page.selects, *page.inputs, *page.buttons]
    else:
        pool = [*page.buttons, *page.links]
    for element in pool:
        label = (element.text or "").strip().lower()
        if label in phrases:
            return element
        if role == "search_input" and any(phrase in label for phrase in phrases if len(phrase) >= 8):
            return element
    return None

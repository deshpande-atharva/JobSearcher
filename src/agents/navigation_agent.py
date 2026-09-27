"""Discovery agent: pick the next public navigation action from page state."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.browser.actions import BrowserAction
from src.browser.page_state import PageState
from src.llm.base import LLMProvider
from src.navigation.semantics import find_semantic

__all__ = ["NavigationChoice", "decide_action"]


class NavigationChoice(BaseModel):
    action: str = "click"
    target_id: str = ""
    semantic_role: str = ""
    text: str = ""
    reason: str = Field(default="", max_length=400)


async def decide_action(page: PageState, llm: LLMProvider | None) -> BrowserAction | None:
    """Use the semantic catalog first. Call the LLM only when that catalog misses."""
    for role in ("view_jobs", "software_engineering", "paginate"):
        element = find_semantic(page, role)
        if element is not None:
            return BrowserAction(
                action="click",
                target_id=element.id,
                semantic_role=role,
                text=element.text,
                reason=f"semantic match for {role}",
            )
    if llm is not None and getattr(llm, "available", False):
        choice = await llm.structured(
            prompt=_prompt(page),
            response_model=NavigationChoice,
            system=(
                "Choose one control that reveals job postings. "
                "Search Jobs, View Jobs, Explore Opportunities, and Open Positions are the same action. "
                "Load More, Show More, and Next Page are pagination. "
                "Never choose apply, submit, or sign-in controls."
            ),
            purpose="greenhouse_navigation",
        )
        if isinstance(choice, NavigationChoice) and choice.target_id:
            return BrowserAction(
                action="click",
                target_id=choice.target_id,
                semantic_role=choice.semantic_role or None,
                text=choice.text or None,
                reason=choice.reason,
            )
    return None


def _prompt(page: PageState) -> str:
    controls = [
        {"id": item.id, "text": item.text, "role": item.role}
        for item in [*page.buttons, *page.links]
    ]
    return (
        f"URL: {page.url}\n"
        f"Title: {page.title}\n"
        f"Controls: {controls}\n"
        "Return the control that should be activated next."
    )

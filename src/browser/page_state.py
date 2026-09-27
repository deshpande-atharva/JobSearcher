"""Normalized page state the discovery agent is allowed to see."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["JobCard", "PageElement", "PageState"]


class PageElement(BaseModel):
    id: str
    text: str = ""
    role: str = ""
    href: str | None = None
    options: list[str] = Field(default_factory=list)
    value: str = ""


class JobCard(BaseModel):
    id: str
    title: str
    company: str = ""
    location: str = ""
    url: str = ""
    job_id: str = ""
    description: str = ""


class PageState(BaseModel):
    url: str = ""
    title: str = ""
    visible_text: str = ""
    buttons: list[PageElement] = Field(default_factory=list)
    links: list[PageElement] = Field(default_factory=list)
    inputs: list[PageElement] = Field(default_factory=list)
    selects: list[PageElement] = Field(default_factory=list)
    job_cards: list[JobCard] = Field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""

    def elements(self) -> list[PageElement]:
        return [*self.buttons, *self.links, *self.inputs, *self.selects]

    def find(self, element_id: str) -> PageElement | None:
        for element in self.elements():
            if element.id == element_id:
                return element
        return None

    def fingerprint(self) -> str:
        labels = [f"{item.role}:{item.text}" for item in self.elements()]
        cards = [card.job_id or card.url or card.title for card in self.job_cards]
        return "|".join([self.url, *labels, *cards])

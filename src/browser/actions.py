"""Structured browser actions. The model never supplies code to execute."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.browser.page_state import PageState

__all__ = ["ActionError", "BrowserAction", "validate_action"]

ActionName = Literal["open", "back", "click", "type", "select", "submit", "scroll", "wait"]

_REJECTED_TEXT = (
    "submit application",
    "submit",
    "apply now",
    "create account",
    "sign in",
    "log in",
    "login",
)


class ActionError(ValueError):
    """The proposed action is not allowed on the current page."""


class BrowserAction(BaseModel):
    action: ActionName
    target_id: str | None = None
    semantic_role: str | None = None
    text: str | None = None
    value: str | None = None
    reason: str = ""
    script: str | None = Field(default=None, description="Rejected when present.")


def validate_action(action: BrowserAction, page: PageState) -> BrowserAction:
    """Return the action only when it is a safe, in-page operation."""
    if action.script:
        raise ActionError("arbitrary script execution is not allowed")
    if action.action == "open":
        url = (action.value or "").strip()
        if not url.startswith(("https://", "http://")):
            raise ActionError("open requires an http(s) URL")
        return action
    if action.action in {"back", "scroll", "wait"}:
        return action
    if action.action == "submit":
        if not action.target_id:
            raise ActionError("submit requires a target id from the page state")
        element = page.find(action.target_id)
        if element is None:
            raise ActionError(f"unknown element {action.target_id}")
        if not any(item.id == element.id for item in page.inputs):
            raise ActionError("submit is only allowed on a text input")
        label = (element.text or "").strip().lower()
        if any(banned in label for banned in ("apply", "sign in", "log in", "login")):
            raise ActionError(f"refusing to submit {element.text!r}")
        return action
    if action.action not in {"click", "type", "select"}:
        raise ActionError(f"unsupported action {action.action}")
    if not action.target_id:
        raise ActionError(f"{action.action} requires a target id from the page state")
    element = page.find(action.target_id)
    if element is None:
        raise ActionError(f"unknown element {action.target_id}")
    label = (element.text or "").strip().lower()
    if action.action == "click" and any(label == banned or banned in label for banned in _REJECTED_TEXT):
        raise ActionError(f"refusing to activate {element.text!r}")
    if action.action == "type" and not (action.value or "").strip():
        raise ActionError("type requires a value")
    if action.action == "select":
        choice = (action.value or "").strip()
        if choice not in element.options:
            raise ActionError(f"{choice!r} is not an option of {element.id}")
    return action

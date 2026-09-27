"""Future application agent. This phase does not submit applications."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from src.services.candidate_profile import CandidateProfile

__all__ = ["ApplicationRequest"]


class ApplicationRequest(BaseModel):
    official_url: str
    company: str
    title: str
    description: str = ""


class ApplicationAgent(Protocol):
    async def prepare(self, request: ApplicationRequest, profile: CandidateProfile) -> str:
        """Prepare a review packet and stop for a person. Must not submit."""

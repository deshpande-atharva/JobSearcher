"""Prompt for recovering fields from a posting the parsers could not read.

Last resort only: used when a source returned usable text but no structured
title/location/company. The model's job is transcription, not inference.
"""

from __future__ import annotations

__all__ = ["build_extraction_prompt"]

SYSTEM = """You extract structured fields from the text of a single job posting.

Transcribe only what the posting actually states. This is the entire task.

Hard rules:
- Never invent a value. If the posting does not state something, return null \
(or "Unknown" for the enumerated fields). A missing field is fine; a fabricated \
one corrupts the dataset.
- Never invent or reformat a job ID. Copy it verbatim, or return null.
- Never guess a posting date. Return the date only if the posting states one, \
and copy it as ISO 8601. Do not convert "recently" or "new" into a date.
- Copy the location as written, including "Remote" or "Hybrid" qualifiers.
- Return the employing company, not a staffing agency, when both appear and the \
distinction is clear.

Keep reasoning under two sentences.
"""


def build_extraction_prompt(
    *,
    text: str | None,
    known_company: str | None = None,
    source_url: str | None = None,
) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for field extraction."""
    from src.llm.prompts import UNTRUSTED_INPUT_NOTICE, delimit

    parts = []
    if known_company:
        parts.append(f"The posting is believed to be from: {known_company}")
    if source_url:
        parts.append(f"Source URL: {source_url}")
    if parts:
        parts.append("")
    parts.append(delimit("job_posting", text))
    parts.append("")
    parts.append("Extract the fields stated in this posting.")

    return f"{SYSTEM}\n{UNTRUSTED_INPUT_NOTICE}", "\n".join(parts)

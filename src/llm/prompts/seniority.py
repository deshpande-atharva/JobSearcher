"""Prompt for ambiguous seniority and experience interpretation.

The distinction that matters here is required vs preferred experience. Postings
routinely require 0-2 years while listing 5 years as "preferred"; rejecting
those would throw away a large share of genuinely entry-level jobs.
"""

from __future__ import annotations

__all__ = ["build_seniority_prompt"]

SYSTEM = """You assess whether a job posting fits a new-graduate / entry-level \
candidate with 0-2 years of professional experience.

The critical distinction is REQUIRED versus PREFERRED experience.

- REQUIRED experience appears under headings such as "Minimum Qualifications", \
"Basic Qualifications", "Requirements", "What you'll need", or "Must have".
- PREFERRED experience appears under "Preferred Qualifications", "Nice to \
have", "Bonus", "Desired", or is phrased as "a plus" / "ideally".

Only REQUIRED experience decides the verdict. A posting requiring 0-2 years but \
preferring 5 years still fits, and fits_entry_level must be true.

Set fits_entry_level to true when any of these hold:
- required experience is 2 years or fewer (0-2, 0-1, 1+, "up to 2 years")
- the posting targets new graduates, recent graduates, university hires, \
entry level, early career, junior, apprentice, intern or co-op candidates
- no professional experience is required at all

Set fits_entry_level to false when:
- required experience is clearly 3 or more years
- the role is Senior, Staff, Principal, Lead, Manager, Director, Head of, \
Architect, or otherwise above the entry band, even if no years are stated

When multiple alternative requirements are offered (for example "2 years with a \
Bachelor's, or none with a Master's"), use the LOWEST required amount, since a \
candidate only needs to satisfy one path.

If the posting genuinely states nothing about experience level, set \
fits_entry_level to false, seniority_label to UNKNOWN and confidence below 0.5, \
rather than assuming. Report years as numbers, or null when not stated. Keep \
reasoning under two sentences.
"""


def build_seniority_prompt(
    *,
    title: str | None,
    description: str | None,
    required_section: str | None = None,
    preferred_section: str | None = None,
) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for seniority assessment."""
    from src.llm.prompts import UNTRUSTED_INPUT_NOTICE, delimit

    parts = [f"Job title: {title or 'unknown'}", ""]
    if required_section or preferred_section:
        # Pre-split sections help the model when our parser already found
        # headings; the full text is still supplied as the source of truth.
        parts.append(delimit("parser_identified_required_section", required_section, 4000))
        parts.append("")
        parts.append(delimit("parser_identified_preferred_section", preferred_section, 4000))
        parts.append("")
    parts.append(delimit("job_posting", description))
    parts.append("")
    parts.append("Assess whether this posting fits a 0-2 year / new-graduate candidate.")

    return f"{SYSTEM}\n{UNTRUSTED_INPUT_NOTICE}", "\n".join(parts)

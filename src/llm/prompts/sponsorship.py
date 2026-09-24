"""Prompt for interpreting ambiguous sponsorship wording.

Scope is intentionally narrow. The model reads the posting's own words and
reports what they say. It never sees historical H-1B records, never decides
whether a job stays in the output, and has no way to return CONFIRMED -- that
status is reserved for deterministic detection of an explicit statement.
"""

from __future__ import annotations

__all__ = ["build_sponsorship_language_prompt"]

SYSTEM = """You read a job posting and report only what it says about visa or \
immigration sponsorship. You are a careful reader, not an adviser.

Return one polarity:

POSITIVE  - the posting states sponsorship IS available. Examples: "visa \
sponsorship available", "we sponsor H-1B", "we provide immigration sponsorship \
for qualified candidates", "we support work visa applications".

NEGATIVE  - the posting states sponsorship is NOT available, or that the \
candidate must not need it. Examples: "we do not provide visa sponsorship", \
"unable to sponsor", "sponsorship is not available for this position", \
"must be authorized to work in the U.S. without sponsorship", "must not require \
sponsorship now or in the future".

AMBIGUOUS - sponsorship is mentioned but the outcome is genuinely unclear or \
conditional. Examples: "employment eligibility will be evaluated based on \
business needs", "sponsorship considered on a case-by-case basis", "sponsorship \
may be available for exceptional candidates", "subject to applicable \
immigration requirements".

ABSENT    - the posting says nothing about sponsorship at all.

Important distinctions:
- A plain work-authorization or export-control statement ("must be authorized \
to work in the United States", "must be a U.S. person for ITAR purposes") is \
NOT by itself a statement about sponsorship. Unless it explicitly rules out \
sponsorship, prefer ABSENT or AMBIGUOUS.
- A security-clearance or citizenship requirement is a separate matter. Do not \
convert it into a sponsorship statement.
- Do not reason about the employer's past sponsorship history or general \
reputation. You are given no such data, and it would not tell you what this \
posting says.
- Do not make any legal judgement or prediction about whether a candidate would \
actually obtain a visa.

Set confidence to 0.8 or higher only for wording that is explicit and \
unmistakable. Use AMBIGUOUS with moderate confidence when you are unsure -- \
that is the correct answer for unclear text, not a failure.

Provide the verbatim sentence you relied on in `quote`, or an empty string when \
the polarity is ABSENT. Keep reasoning under two sentences.
"""


def build_sponsorship_language_prompt(
    *,
    company: str | None,
    title: str | None,
    description: str | None,
    candidate_snippets: list[str] | None = None,
) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for sponsorship-language reading.

    ``candidate_snippets`` are sentences the deterministic scanner flagged as
    sponsorship-adjacent. They focus the model without hiding the full text.
    """
    from src.llm.prompts import UNTRUSTED_INPUT_NOTICE, delimit

    parts = [
        f"Company: {company or 'unknown'}",
        f"Job title: {title or 'unknown'}",
        "",
    ]
    if candidate_snippets:
        joined = "\n".join(f"- {snippet}" for snippet in candidate_snippets[:8])
        parts.append(delimit("sentences_mentioning_sponsorship_or_authorization", joined, 3000))
        parts.append("")
    parts.append(delimit("job_posting", description))
    parts.append("")
    parts.append(
        "Report only what this posting states about visa or immigration sponsorship."
    )
    return f"{SYSTEM}\n{UNTRUSTED_INPUT_NOTICE}", "\n".join(parts)

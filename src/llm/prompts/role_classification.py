"""Prompt for semantic role classification.

Reached only when deterministic keyword matching is inconclusive -- typically a
vague title, or a Data/ML title where only the described work can settle whether
the job is really software engineering.
"""

from __future__ import annotations

__all__ = ["build_role_prompt"]

SYSTEM = """You classify job postings for a software engineering job search.

Decide whether the posting is a SOFTWARE ENGINEERING role by judging the actual \
work described, not the job title. Titles are inconsistent across companies and \
are weak evidence on their own.

Count as software engineering when the core of the job is designing, writing, \
testing, shipping, or operating software: application development, backend or \
frontend development, distributed systems, platform and infrastructure \
engineering, DevOps and SRE, systems and embedded software, mobile development, \
security engineering with a heavy build component, or data/ML work where the \
person is primarily building production software and pipelines.

Do NOT count roles whose core is something other than building software, even \
when they are technical: data analysis and reporting, research science without \
production engineering, product or program management, design, IT support and \
system administration, network or database administration, sales engineering, \
QA that is purely manual, and non-software engineering disciplines such as \
mechanical, electrical or hardware engineering.

Data, ML and AI titles are neither automatically included nor automatically \
excluded. Judge them by the work: building and operating production systems and \
pipelines is software engineering; producing analyses, dashboards or research \
papers is not.

Rules:
- Base the decision on responsibilities, qualifications and technologies.
- If the posting genuinely does not say enough to tell, set \
is_software_engineering to false with low confidence rather than guessing.
- Choose the role_family that best fits the described work. Use NOT_SOFTWARE \
when is_software_engineering is false.
- Keep reasoning under two sentences.
"""


def build_role_prompt(
    *,
    title: str | None,
    company: str | None,
    description: str | None,
    candidate_families: list[str] | None = None,
) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for role classification."""
    from src.llm.prompts import UNTRUSTED_INPUT_NOTICE, delimit

    hint = ""
    if candidate_families:
        hint = (
            "\nDeterministic keyword matching suggested these families, which may be "
            f"wrong: {', '.join(candidate_families)}\n"
        )

    user = (
        f"Company: {company or 'unknown'}\n"
        f"Job title: {title or 'unknown'}\n"
        f"{hint}\n"
        f"{delimit('job_posting', description)}\n\n"
        "Classify this posting."
    )
    return f"{SYSTEM}\n{UNTRUSTED_INPUT_NOTICE}", user

"""In-run and historical deduplication.

Primary key: company + job ID.
Fallback: company + normalized title + location + direct URL.

Two postings that share a title but carry different job IDs are different jobs.
"""

from __future__ import annotations

from src.models.job import Job

__all__ = ["DedupResult", "deduplicate"]


class DedupResult:
    __slots__ = ("unique", "duplicate_count")

    def __init__(self, unique: list[Job], duplicate_count: int) -> None:
        self.unique = unique
        self.duplicate_count = duplicate_count


def deduplicate(jobs: list[Job], known_keys: set[str] | None = None) -> tuple[list[Job], list[Job], list[Job]]:
    """Split ``jobs`` into ``(unique_new, in_run_duplicates, historically_seen)``.

    The first occurrence of a key wins. Subsequent in-run copies are duplicates.
    Keys present in ``known_keys`` are historically seen (``is_new`` is false).
    They stay available so today's snapshot can keep them with prior tracking.
    Callers decide whether to drop them; the tracker keeps qualifying ones.
    """
    seen: set[str] = set()
    unique: list[Job] = []
    duplicates: list[Job] = []
    historical: list[Job] = []
    known = known_keys or set()

    for job in jobs:
        key = job.dedup_key
        if key in known:
            job.is_new = False
            historical.append(job)
            continue
        if key in seen:
            job.is_new = False
            duplicates.append(job)
            continue
        seen.add(key)
        job.is_new = True
        unique.append(job)

    return unique, duplicates, historical

"""Persisted Greenhouse navigation strategies.

A failed replay increments the failure count. It does not replace the actions
of a validated strategy. A recovered path is stored as the next version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

__all__ = ["NavigationMemory", "StrategyStep", "StrategyVersion"]


class StrategyStep(BaseModel):
    action: str
    semantic_target: str
    text: str
    value: str = ""


class StrategyVersion(BaseModel):
    version: int
    actions: list[StrategyStep]
    status: str = "candidate"
    success_count: int = 0
    failure_count: int = 0
    created_at: str | None = None
    last_success: str | None = None
    last_failure: str | None = None
    last_verified: str | None = None


class StrategyRecord(BaseModel):
    source: str = "greenhouse"
    source_type: str = "greenhouse"
    company: str
    career_url: str
    versions: list[StrategyVersion] = Field(default_factory=list)

    @property
    def current(self) -> StrategyVersion | None:
        validated = [item for item in self.versions if item.status == "validated"]
        return validated[-1] if validated else (self.versions[-1] if self.versions else None)


class NavigationMemory:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self, company: str, source: str = "greenhouse") -> StrategyRecord | None:
        path = self._path(company, source)
        if not path.exists():
            return None
        return StrategyRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: StrategyRecord) -> None:
        source = record.source or record.source_type or "greenhouse"
        self._path(record.company, source).write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def note_success(self, record: StrategyRecord, version: StrategyVersion) -> None:
        version.success_count += 1
        version.last_success = _now()
        version.last_verified = version.last_success
        self.save(record)

    def note_failure(self, record: StrategyRecord, version: StrategyVersion) -> None:
        version.failure_count += 1
        version.last_failure = _now()
        self.save(record)

    def add_version(self, record: StrategyRecord, actions: list[StrategyStep], *, validated: bool) -> StrategyVersion:
        version = StrategyVersion(
            version=(record.versions[-1].version + 1) if record.versions else 1,
            actions=actions,
            status="validated" if validated else "candidate",
            success_count=1 if validated else 0,
            created_at=_now(),
            last_success=_now() if validated else None,
            last_verified=_now() if validated else None,
        )
        record.versions.append(version)
        self.save(record)
        return version

    def promote(self, record: StrategyRecord, version: StrategyVersion) -> None:
        """Promote a candidate only after a successful validation."""
        version.status = "validated"
        version.success_count += 1
        version.last_success = _now()
        version.last_verified = version.last_success
        self.save(record)

    def _path(self, company: str, source: str = "greenhouse") -> Path:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in company).strip("-")
        prefix = "".join(ch.lower() if ch.isalnum() else "-" for ch in (source or "greenhouse")).strip("-")
        return self.directory / f"{prefix or 'greenhouse'}-{slug}.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

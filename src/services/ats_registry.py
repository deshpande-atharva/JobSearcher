"""Repository-local cache of discovered ATS board identifiers.

Manual company configuration always wins. Failed detections are stored without
an identifier so they can be retried later. Nothing guessed is persisted.
The file stores no secrets and no job listings.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from src.models.config import AppConfig
from src.services.ats_discovery import is_valid_ats_identifier
from src.utils.dates import utcnow
from src.utils.logging import get_logger
from src.utils.normalization import normalize_company_name

__all__ = ["AtsRegistry", "AtsRegistryEntry", "load_ats_registry"]

log = get_logger(__name__)

DiscoveryMethod = Literal["manual", "automatic", "failed"]


def _normalize_method(value: str | None) -> str:
    raw = (value or "automatic").strip().lower()
    if raw == "manual":
        return "manual"
    if raw == "failed":
        return "failed"
    return "automatic"


class AtsRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ats_type: str | None = None
    ats_identifier: str | None = None
    discovered_at: datetime | None = None
    discovery_method: str = "automatic"
    careers_url: str | None = None
    confidence: float | None = None

    @field_validator("ats_type", "ats_identifier", "careers_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("discovery_method", mode="before")
    @classmethod
    def _method(cls, value: Any) -> Any:
        return _normalize_method(str(value) if value is not None else None)

    @property
    def verified(self) -> bool:
        return self.discovery_method != "failed" and is_valid_ats_identifier(
            self.ats_type, self.ats_identifier
        )


class AtsRegistry:
    def __init__(self, path: Path, entries: dict[str, AtsRegistryEntry] | None = None) -> None:
        self.path = path
        self._entries = entries or {}
        self._dirty = False

    def get(self, company_name: str) -> AtsRegistryEntry | None:
        return self._entries.get(normalize_company_name(company_name))

    def get_verified(self, company_name: str) -> AtsRegistryEntry | None:
        entry = self.get(company_name)
        if entry and entry.verified:
            return entry
        return None

    def remember(
        self,
        company_name: str,
        *,
        ats_type: str,
        ats_identifier: str,
        method: str,
        careers_url: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        """Cache a verified identifier. Invalid tokens are refused."""
        if not is_valid_ats_identifier(ats_type, ats_identifier):
            log.info(
                "refusing to cache invalid ats identifier",
                company=company_name,
                ats_type=ats_type,
                identifier=ats_identifier,
            )
            return False
        key = normalize_company_name(company_name)
        existing = self._entries.get(key)
        stored_method = _normalize_method(method)
        if (
            existing
            and existing.ats_type == ats_type
            and existing.ats_identifier == ats_identifier
            and existing.discovery_method == stored_method
        ):
            return True
        self._entries[key] = AtsRegistryEntry(
            ats_type=ats_type,
            ats_identifier=ats_identifier,
            discovered_at=utcnow(),
            discovery_method=stored_method,
            careers_url=careers_url,
            confidence=confidence,
        )
        self._dirty = True
        return True

    def remember_failure(
        self,
        company_name: str,
        *,
        careers_url: str | None = None,
    ) -> None:
        """Record that automatic detection found nothing. No identifier stored."""
        key = normalize_company_name(company_name)
        existing = self._entries.get(key)
        if existing and existing.verified:
            return
        self._entries[key] = AtsRegistryEntry(
            ats_type=None,
            ats_identifier=None,
            discovered_at=utcnow(),
            discovery_method="failed",
            careers_url=careers_url,
        )
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        payload: dict[str, Any] = {
            "companies": {
                name: entry.model_dump(mode="json")
                for name, entry in sorted(self._entries.items())
            }
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            self._dirty = False
            log.info("ats registry updated", path=str(self.path), entries=len(self._entries))
        except OSError as exc:
            log.warning("could not persist ats registry", path=str(self.path), error=str(exc))


def load_ats_registry(config: AppConfig) -> AtsRegistry:
    path = config.path(config.settings.discovery.ats_registry)
    entries: dict[str, AtsRegistryEntry] = {}
    if path.is_file():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            log.warning("invalid ats registry YAML; starting empty", error=str(exc))
            raw = {}
        companies = raw.get("companies") if isinstance(raw, dict) else {}
        if isinstance(companies, dict):
            for name, data in companies.items():
                if not isinstance(data, dict):
                    continue
                try:
                    entry = AtsRegistryEntry.model_validate(data)
                except Exception:
                    continue
                if entry.discovery_method != "failed" and not entry.verified:
                    continue
                entries[normalize_company_name(str(name))] = entry
    return AtsRegistry(path, entries)

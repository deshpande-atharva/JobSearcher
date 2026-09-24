"""Offline fixture loading.

``--fixture-mode`` runs the entire pipeline against saved payloads. Adapters read
their fixture through the same parsing code they use against the live API, so an
offline run exercises extraction, classification, H-1B matching, deduplication
and XLSX generation for real -- it is not a mock of the pipeline, only of the
network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

__all__ = ["FixtureStore", "slugify"]

log = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str | None) -> str:
    """Filesystem-safe lowercase slug, e.g. ``Home Depot`` -> ``home-depot``."""
    return _SLUG_RE.sub("-", (value or "").strip().lower()).strip("-")


class FixtureStore:
    """Reads fixture payloads from ``tests/fixtures/<source>/``.

    Every lookup is confined to the fixture root: a resolved path that escapes
    the root is refused, so a crafted company name cannot read arbitrary files.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _resolve(self, source: str, name: str) -> Path | None:
        candidate = (self.root / slugify(source) / name).resolve()
        root = self.root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            log.warning("refusing fixture path outside the fixture root", path=str(candidate))
            return None
        return candidate

    def exists(self, source: str, name: str) -> bool:
        path = self._resolve(source, name)
        return bool(path and path.is_file())

    def load_text(self, source: str, name: str) -> str | None:
        """Read a fixture as text, or ``None`` when it is absent."""
        path = self._resolve(source, name)
        if not path or not path.is_file():
            log.debug("fixture not found", source=source, name=name, path=str(path))
            return None
        return path.read_text(encoding="utf-8")

    def load_json(self, source: str, name: str) -> Any | None:
        """Read and parse a JSON fixture, or ``None`` when absent/invalid."""
        text = self.load_text(source, name)
        if text is None:
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            log.warning("fixture is not valid JSON", source=source, name=name, error=str(exc))
            return None

    def load_first(self, source: str, names: list[str]) -> Any | None:
        """Try several candidate filenames and return the first JSON payload."""
        for name in names:
            payload = self.load_json(source, name)
            if payload is not None:
                return payload
        return None

    def list_files(self, source: str, suffix: str = ".json") -> list[Path]:
        directory = self.root / slugify(source)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == suffix)

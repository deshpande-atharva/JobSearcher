"""Deterministic PDF text extraction. OCR is not used when the file has text."""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["ResumeDocument", "load_resume_pdf"]


class ResumeDocument:
    def __init__(self, path: Path, text: str, page_count: int, sha256: str) -> None:
        self.path = path
        self.text = text
        self.page_count = page_count
        self.sha256 = sha256
        self.filename = path.name

    @property
    def has_text(self) -> bool:
        return len(self.text.strip()) >= 40


def load_resume_pdf(path: Path) -> ResumeDocument:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [_page_text(page) for page in reader.pages]
    text = "\n".join(page for page in pages if page).strip()
    return ResumeDocument(path=path, text=text, page_count=len(pages), sha256=digest)


def _page_text(page: object) -> str:
    layout = _clean(page.extract_text(extraction_mode="layout") or "")  # type: ignore[attr-defined]
    if len(layout) >= 40:
        return layout
    return _clean(page.extract_text(space_width=250) or "")  # type: ignore[attr-defined]


def _clean(text: str) -> str:
    """Collapse layout gaps and join wrapped bullet lines. Does not rewrite words."""
    merged: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        piece = " ".join(raw.split())
        indented = bool(raw[:1].isspace())
        previous = merged[-1] if merged else ""
        join_wrap = indented and previous and not piece.startswith(("•", "-", "*"))
        join_wrap = join_wrap and (previous.endswith("-") or previous.lstrip().startswith(("•", "-", "*")))
        if join_wrap:
            merged[-1] = previous + piece if previous.endswith("-") else f"{previous} {piece}"
        else:
            merged.append(piece)
    return "\n".join(merged)

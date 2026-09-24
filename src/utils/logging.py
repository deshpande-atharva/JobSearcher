"""Structured logging with secret redaction.

Built on the standard library so the project has no logging dependency. Two
things matter here:

* Log records carry structured key/value context, renderable as human text or
  as JSON lines for CI.
* Anything that looks like a credential is redacted before it can reach a log
  file or the GitHub Actions console.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any

__all__ = ["JobLogger", "configure_logging", "get_logger", "redact"]


# Patterns covering the credential shapes this project actually handles.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Google AI Studio keys.
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    # Bearer tokens.
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{12,}"),
    # key=value / "key": "value" for sensitive names.
    re.compile(
        r"(?i)\b(api[_-]?key|apikey|token|secret|password|passwd|pwd|authorization|"
        r"gemini[_-]?api[_-]?key|smtp[_-]?password)\b(\s*[:=]\s*\"?)([^\s\",;]{4,})"
    ),
)

_REDACTED = "***REDACTED***"

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "gemini_api_key",
        "token",
        "secret",
        "password",
        "smtp_password",
        "authorization",
        "auth",
        "credentials",
    }
)


def redact(value: Any) -> Any:
    """Recursively strip credential-looking content from a value."""
    if isinstance(value, str):
        text = value
        text = _SECRET_PATTERNS[0].sub(_REDACTED, text)
        text = _SECRET_PATTERNS[1].sub(r"\1 " + _REDACTED, text)
        text = _SECRET_PATTERNS[2].sub(rf"\1\2{_REDACTED}", text)
        return text
    if isinstance(value, dict):
        return {
            key: (_REDACTED if str(key).lower() in _SENSITIVE_KEYS else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def _render_context(context: dict[str, Any]) -> str:
    parts = []
    for key, value in context.items():
        rendered = value if isinstance(value, str) else repr(value)
        if isinstance(value, str) and (" " in value or "=" in value):
            rendered = f'"{value}"'
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


class _TextFormatter(logging.Formatter):
    """Human-readable ``time level logger message key=value`` output."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "
            f"{record.levelname:<8} {record.name:<28} {record.getMessage()}"
        )
        context = getattr(record, "context", None)
        if context:
            base = f"{base} | {_render_context(context)}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return redact(base)


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, convenient for CI log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str)


class JobLogger:
    """Thin logging facade that accepts structured keyword context.

    ``log.info("discovered", source="jobright", count=42)`` keeps call sites
    readable while producing machine-parseable output.
    """

    __slots__ = ("_logger", "_bound")

    def __init__(self, logger: logging.Logger, bound: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._bound = bound or {}

    def bind(self, **context: Any) -> JobLogger:
        """Return a child logger carrying additional fixed context."""
        merged = {**self._bound, **context}
        return JobLogger(self._logger, merged)

    def _log(self, level: int, message: str, exc_info: bool = False, **context: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        merged = redact({**self._bound, **context})
        self._logger.log(level, message, extra={"context": merged}, exc_info=exc_info)

    def debug(self, message: str, **context: Any) -> None:
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        self._log(logging.ERROR, message, **context)

    def exception(self, message: str, **context: Any) -> None:
        self._log(logging.ERROR, message, exc_info=True, **context)


_CONFIGURED = False


def configure_logging(level: str | int | None = None, fmt: str | None = None) -> None:
    """Install the root handler. Safe to call more than once."""
    global _CONFIGURED

    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)
    resolved_fmt = (fmt or os.environ.get("LOG_FORMAT", "text")).lower()

    root = logging.getLogger()
    root.setLevel(resolved_level)

    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(resolved_level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(resolved_level)
    handler.setFormatter(_JsonFormatter() if resolved_fmt == "json" else _TextFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    # Third-party libraries are chatty at INFO and drown out the run summary.
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "google.genai", "asyncio"):
        logging.getLogger(noisy).setLevel(max(resolved_level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str, **context: Any) -> JobLogger:
    """Get a structured logger, optionally pre-bound with context."""
    return JobLogger(logging.getLogger(name), context or None)

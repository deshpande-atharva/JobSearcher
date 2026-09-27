"""LLM provider interface, budgeting and failure containment.

Design rules, all of which exist because an LLM is the least reliable component
in the pipeline:

* Every call is budgeted. Once ``max_calls_per_run`` is spent the provider stops
  calling out and returns ``None``.
* Consecutive failures trip a circuit breaker, so a provider outage costs one
  burst of retries rather than one timeout per job.
* ``structured()`` returns ``None`` on *any* problem -- transport error, bad
  JSON, schema mismatch, empty response. Callers treat ``None`` as "no opinion"
  and fall back to deterministic logic. An LLM failure must never raise into the
  pipeline.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from src.models.config import AppConfig, LLMSettings
from src.utils.logging import get_logger

__all__ = [
    "LLMProvider",
    "LLMStats",
    "NullLLMProvider",
    "build_llm_provider",
    "classify_provider_error",
    "json_schema_for",
]

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMStats(BaseModel):
    """Per-run accounting, surfaced in the run summary."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    fallbacks: int = 0
    deterministic_avoided: int = 0
    recovery_attempts: int = 0
    skipped_budget: int = 0
    skipped_circuit_open: int = 0
    by_purpose: dict[str, int] = Field(default_factory=dict)
    failures_by_category: dict[str, int] = Field(default_factory=dict)
    circuit_state: str = "CLOSED"

    def record_call(self, purpose: str) -> None:
        self.calls += 1
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1

    def record_failure(self, category: str) -> None:
        self.failures += 1
        self.failures_by_category[category] = self.failures_by_category.get(category, 0) + 1


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Produce a flat JSON Schema suitable for a structured-output request.

    Pydantic emits ``$defs``/``$ref`` for nested models and enums, which the
    Gemini structured-output subset handles inconsistently. Response models in
    :mod:`src.llm.schemas` are deliberately flat and use ``Literal`` rather than
    ``Enum``; this function inlines any remaining references and strips
    decorative keys so the payload stays inside the supported subset.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = str(node["$ref"])
                name = ref.rsplit("/", 1)[-1]
                target = defs.get(name, {})
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {
                key: resolve(value)
                for key, value in node.items()
                if key not in ("title", "default", "examples", "$schema")
            }
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(schema)
    if isinstance(resolved, dict):
        resolved.setdefault("type", "object")
    return resolved


def _strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text.strip()


def _extract_json_object(text: str) -> str:
    """Isolate the outermost JSON object in a response.

    Models occasionally prepend a sentence even under JSON mode; salvage the
    object rather than discarding an otherwise-good answer.
    """
    cleaned = _strip_code_fence(text)
    if cleaned.startswith("{"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


_RETRYABLE = frozenset({"429", "503", "timeout", "connection"})
_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|token|password|secret|authorization)\b\s*[=:]\s*\S+")


def classify_provider_error(exc: BaseException) -> str:
    """Bucket a provider exception without retaining secrets."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        code = getattr(exc, "code", None)
        status = code if isinstance(code, int) else None
    text = _SECRET_RE.sub(r"\1=<redacted>", str(exc)).lower()
    if status == 429 or "429" in text or "resource_exhausted" in text or "rate limit" in text:
        return "429"
    if status == 503 or "503" in text or "unavailable" in text:
        return "503"
    if isinstance(status, int) and 500 <= status <= 599:
        return "503"
    if any(token in text for token in ("connection", "connecterror", "network", "timed out")):
        return "connection"
    return "other"


def _permanent_configuration(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {400, 401, 403}:
        return True
    text = str(exc).lower()
    return any(token in text for token in ("unauthenticated", "permission_denied", "api key", "invalid_argument"))


class LLMProvider(ABC):
    """Base class handling budget, retries, parsing and validation.

    Subclasses implement :meth:`_generate` only.
    """

    name: str = "base"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.stats = LLMStats()
        self._consecutive_failures = 0
        self._circuit_open = False
        self._opened_at: float | None = None
        self._probing = False

    # --- subclass contract --------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether this provider can currently be called at all."""
        return True

    @abstractmethod
    async def _generate(
        self,
        *,
        prompt: str,
        system: str | None,
        schema: dict[str, Any],
    ) -> str:
        """Return the raw response text. May raise; callers handle it."""

    async def aclose(self) -> None:
        """Release provider resources. Default is a no-op."""

    # --- public API ---------------------------------------------------------

    @property
    def budget_remaining(self) -> int:
        return max(self.settings.max_calls_per_run - self.stats.calls, 0)

    def can_call(self) -> bool:
        """Cheap pre-check so callers can skip building an expensive prompt."""
        return self.available and self._circuit_allows_call() and self.budget_remaining > 0

    @property
    def circuit_state(self) -> str:
        if not self._circuit_open:
            state = "CLOSED"
        elif self._probing:
            state = "HALF_OPEN"
        else:
            state = "OPEN"
        self.stats.circuit_state = state
        return state

    def note_deterministic_avoided(self) -> None:
        self.stats.deterministic_avoided += 1

    def note_unavailable(self) -> None:
        """A caller needed the model and did not invoke it."""
        if self._circuit_open and not self._probe_due():
            self.stats.skipped_circuit_open += 1
        else:
            self.stats.fallbacks += 1

    def _circuit_allows_call(self) -> bool:
        if not self._circuit_open:
            return True
        return self._probe_due()

    def _probe_due(self) -> bool:
        if self._opened_at is None:
            return True
        return (time.monotonic() - self._opened_at) >= self.settings.circuit_reset_seconds

    def _open_circuit(self) -> None:
        # The breaker lives on this provider instance. A new process starts
        # CLOSED. There is no on-disk circuit state to go stale.
        self._circuit_open = True
        self._probing = False
        self._opened_at = time.monotonic()
        self.stats.circuit_state = "OPEN"
        log.error(
            "llm circuit breaker opened; continuing with deterministic logic only",
            provider=self.name,
            consecutive_failures=self._consecutive_failures,
        )

    def _close_circuit(self) -> None:
        self._circuit_open = False
        self._probing = False
        self._opened_at = None
        self._consecutive_failures = 0
        self.stats.circuit_state = "CLOSED"

    async def _pause(self, attempt: int) -> None:
        delay = min(self.settings.retry_backoff_seconds * attempt, 2.0)
        if delay > 0:
            await asyncio.sleep(delay)

    async def structured(
        self,
        *,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        purpose: str = "generic",
    ) -> T | None:
        """Request a JSON response and validate it against ``response_model``.

        Returns ``None`` whenever a usable answer could not be obtained, for any
        reason. Never raises.
        """
        if not self.available:
            return None
        if self._circuit_open:
            if not self._probe_due():
                self.stats.skipped_circuit_open += 1
                self.circuit_state
                return None
            self._probing = True
            self.stats.recovery_attempts += 1
            self.stats.circuit_state = "HALF_OPEN"
        if self.budget_remaining <= 0:
            self.stats.skipped_budget += 1
            log.debug("llm budget exhausted", purpose=purpose, max_calls=self.settings.max_calls_per_run)
            return None

        schema = json_schema_for(response_model)
        attempt = 0
        while True:
            if self.budget_remaining <= 0:
                self.stats.skipped_budget += 1
                break
            attempt += 1
            if attempt > 1:
                self.stats.retries += 1
            self.stats.record_call(purpose)
            category = "other"
            permanent = False
            try:
                raw = await asyncio.wait_for(
                    self._generate(prompt=prompt, system=system, schema=schema),
                    timeout=self.settings.timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                category = "timeout"
                last_error = "timeout"
            except Exception as exc:
                category = classify_provider_error(exc)
                permanent = _permanent_configuration(exc)
                last_error = _SECRET_RE.sub(r"\1=<redacted>", f"{type(exc).__name__}: {exc}")[:300]
            else:
                parsed = self._parse(raw, response_model)
                if parsed is not None:
                    self.stats.successes += 1
                    self._close_circuit()
                    return parsed
                category = "schema"
                last_error = "response did not match the requested schema"

            self.stats.record_failure(category)
            self._consecutive_failures += 1
            log.warning(
                "llm call failed",
                purpose=purpose,
                attempt=attempt,
                provider=self.name,
                category=category,
                error=last_error,
            )
            quota_exhausted = category == "429" and self._consecutive_failures >= 2
            if (
                permanent
                or quota_exhausted
                or self._consecutive_failures >= self.settings.circuit_breaker_failures
            ):
                self._open_circuit()
                break
            if category not in _RETRYABLE or permanent:
                break
            allowed = 2 if category == "429" else self.settings.retry_attempts + 1
            if attempt >= allowed:
                break
            await self._pause(attempt)

        self.stats.fallbacks += 1
        self.circuit_state
        return None

    def _parse(self, raw: str, response_model: type[T]) -> T | None:
        if not raw or not raw.strip():
            return None
        try:
            payload = json.loads(_extract_json_object(raw))
        except (json.JSONDecodeError, ValueError):
            log.debug("llm returned non-JSON payload", provider=self.name)
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            log.debug("llm payload failed validation", provider=self.name, error=str(exc)[:400])
            return None


class NullLLMProvider(LLMProvider):
    """A provider that never calls out.

    Used in fixture mode, in tests, and whenever no API key is present. Keeping
    this a real provider rather than ``None`` means no agent needs an
    ``if llm is not None`` branch; ambiguity simply stays ambiguous and is
    reported as UNKNOWN.
    """

    name = "none"

    def __init__(self, settings: LLMSettings | None = None) -> None:
        super().__init__(settings or LLMSettings(provider="none"))

    @property
    def available(self) -> bool:
        return False

    async def _generate(
        self, *, prompt: str, system: str | None, schema: dict[str, Any]
    ) -> str:  # pragma: no cover - never reached
        raise RuntimeError("NullLLMProvider does not generate")


def build_llm_provider(config: AppConfig) -> LLMProvider:
    """Construct the configured provider, degrading to :class:`NullLLMProvider`.

    Fixture mode always disables the LLM so offline runs are fully
    reproducible.
    """
    settings = config.settings.llm
    if config.fixture_mode:
        log.info("fixture mode: LLM disabled, deterministic logic only")
        return NullLLMProvider(settings)

    provider_name = (config.secrets.llm_provider or settings.provider or "none").strip().lower()
    if provider_name in ("none", "off", "disabled"):
        log.info("LLM provider disabled by configuration")
        return NullLLMProvider(settings)

    if provider_name != "gemini":
        log.warning(
            "unknown LLM provider; continuing without an LLM",
            provider=provider_name,
            supported=["gemini", "none"],
        )
        return NullLLMProvider(settings)

    if not config.secrets.gemini_api_key:
        log.warning(
            "GEMINI_API_KEY is not set; continuing with deterministic logic only. "
            "Ambiguous cases will be reported as UNKNOWN rather than guessed."
        )
        return NullLLMProvider(settings)

    from src.llm.gemini import GeminiProvider

    model = config.secrets.gemini_model or settings.model
    try:
        return GeminiProvider(
            settings=settings.model_copy(update={"model": model}),
            api_key=config.secrets.gemini_api_key,
        )
    except Exception as exc:
        log.warning("failed to initialise Gemini provider; falling back to deterministic logic", error=str(exc))
        return NullLLMProvider(settings)

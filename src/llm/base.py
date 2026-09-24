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
    "json_schema_for",
]

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMStats(BaseModel):
    """Per-run accounting, surfaced in the run summary."""

    calls: int = 0
    failures: int = 0
    skipped_budget: int = 0
    skipped_circuit_open: int = 0
    by_purpose: dict[str, int] = Field(default_factory=dict)

    def record_call(self, purpose: str) -> None:
        self.calls += 1
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1


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
        return self.available and not self._circuit_open and self.budget_remaining > 0

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
            self.stats.skipped_circuit_open += 1
            return None
        if self.budget_remaining <= 0:
            self.stats.skipped_budget += 1
            log.debug("llm budget exhausted", purpose=purpose, max_calls=self.settings.max_calls_per_run)
            return None

        schema = json_schema_for(response_model)
        attempts = self.settings.retry_attempts + 1
        last_error: str | None = None

        for attempt in range(1, attempts + 1):
            if self.budget_remaining <= 0:
                self.stats.skipped_budget += 1
                break
            self.stats.record_call(purpose)
            try:
                raw = await asyncio.wait_for(
                    self._generate(prompt=prompt, system=system, schema=schema),
                    timeout=self.settings.timeout_seconds,
                )
            except asyncio.TimeoutError:
                last_error = "timeout"
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # transport, auth, quota, anything
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                parsed = self._parse(raw, response_model)
                if parsed is not None:
                    self._consecutive_failures = 0
                    return parsed
                last_error = "response did not match the requested schema"

            self.stats.failures += 1
            self._consecutive_failures += 1
            log.warning(
                "llm call failed",
                purpose=purpose,
                attempt=attempt,
                provider=self.name,
                error=last_error,
            )
            if self._consecutive_failures >= self.settings.circuit_breaker_failures:
                self._circuit_open = True
                log.error(
                    "llm circuit breaker opened; continuing with deterministic logic only",
                    provider=self.name,
                    consecutive_failures=self._consecutive_failures,
                )
                break
            if attempt < attempts:
                await asyncio.sleep(min(0.5 * attempt, 2.0))

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

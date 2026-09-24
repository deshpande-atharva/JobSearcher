"""Gemini provider built on the current ``google-genai`` SDK.

Uses ``google.genai`` (the maintained SDK) rather than the deprecated
``google-generativeai`` package, and requests structured output via
``response_json_schema``, which is the path Google recommends going forward --
``response_schema`` receives no new features.

The SDK is imported lazily so the rest of the project (and the whole test suite)
runs without it installed.
"""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider
from src.models.config import LLMSettings
from src.utils.logging import get_logger

__all__ = ["GeminiProvider"]

log = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Structured-output Gemini client.

    One client instance is shared for the whole run. Calls go through
    :meth:`LLMProvider.structured`, which owns budgeting, retries and schema
    validation, so this class only has to perform a single request.
    """

    name = "gemini"

    def __init__(self, settings: LLMSettings, api_key: str) -> None:
        super().__init__(settings)
        if not api_key:
            raise ValueError("a Gemini API key is required")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "the google-genai package is required for the Gemini provider; "
                'install it with `pip install -e ".[dev]"`'
            ) from exc

        self._types = types
        # The key is held only here and is never logged or serialised.
        self._client = genai.Client(api_key=api_key)
        self._model = settings.model
        log.info("gemini provider ready", model=self._model)

    @property
    def available(self) -> bool:
        return True

    def _build_config(self, system: str | None, schema: dict[str, Any]) -> Any:
        return self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=schema,
        )

    async def _generate(
        self,
        *,
        prompt: str,
        system: str | None,
        schema: dict[str, Any],
    ) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._build_config(system, schema),
        )
        text = getattr(response, "text", None)
        if text:
            return text

        # Older/edge responses expose the payload only through candidate parts.
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text
        return ""

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                log.debug("error closing gemini client", error=str(exc))

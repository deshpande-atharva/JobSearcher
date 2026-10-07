"""Anthropic Claude provider.

Uses the ``anthropic`` SDK to request structured JSON output.  The SDK is
imported lazily so the rest of the project (and the whole test suite) runs
without it installed.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm.base import LLMProvider
from src.models.config import LLMSettings
from src.utils.logging import get_logger

__all__ = ["AnthropicProvider"]

log = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """Structured-output Claude client.

    One client instance is shared for the whole run.  Calls go through
    :meth:`LLMProvider.structured`, which owns budgeting, retries and schema
    validation, so this class only has to perform a single request.
    """

    name = "anthropic"

    def __init__(
        self,
        settings: LLMSettings,
        api_key: str,
        workspace_id: str | None = None,
    ) -> None:
        super().__init__(settings)
        if not api_key:
            raise ValueError("an Anthropic API key is required")

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "the anthropic package is required for the Anthropic provider; "
                'install it with `pip install anthropic`'
            ) from exc

        # Workspace-scoped keys work without extra headers.  Non-scoped keys
        # require the anthropic-workspace-id default header.
        default_headers: dict[str, str] = {}
        if workspace_id:
            default_headers["anthropic-workspace-id"] = workspace_id
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            default_headers=default_headers or None,
        )
        self._model = settings.model
        log.info(
            "anthropic provider ready",
            model=self._model,
            workspace_id_set=bool(workspace_id),
        )

    @property
    def available(self) -> bool:
        return True

    async def _generate(
        self,
        *,
        prompt: str,
        system: str | None,
        schema: dict[str, Any],
    ) -> str:
        """Send a single request to Claude and return the raw text."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.temperature,
            "messages": messages,
        }
        if system:
            # Instruct Claude to respond with valid JSON matching the schema.
            schema_text = json.dumps(schema, indent=2)
            kwargs["system"] = (
                f"{system}\n\n"
                f"Respond with a single JSON object matching this schema:\n"
                f"```json\n{schema_text}\n```\n"
                f"Return ONLY valid JSON. No markdown fences, no commentary."
            )
        response = await self._client.messages.create(**kwargs)
        # Extract text from content blocks.
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                await close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                log.debug("error closing anthropic client", error=str(exc))

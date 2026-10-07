"""Provider-agnostic LLM access.

The pipeline only ever talks to :class:`~src.llm.base.LLMProvider`. Anthropic
Claude is the default implementation; Gemini is available as an alternative;
:class:`~src.llm.base.NullLLMProvider` is a complete substitute used in tests,
fixture mode, and whenever no API key is configured.
"""

from src.llm.base import LLMProvider, LLMStats, NullLLMProvider, build_llm_provider

__all__ = [
    "LLMProvider",
    "LLMStats",
    "NullLLMProvider",
    "build_llm_provider",
]

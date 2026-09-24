"""Provider-agnostic LLM access.

The pipeline only ever talks to :class:`~src.llm.base.LLMProvider`. Gemini is
the default implementation; :class:`~src.llm.base.NullLLMProvider` is a complete
substitute used in tests, fixture mode, and whenever no API key is configured.
"""

from src.llm.base import LLMProvider, LLMStats, NullLLMProvider, build_llm_provider
from src.llm.gemini import GeminiProvider

__all__ = [
    "GeminiProvider",
    "LLMProvider",
    "LLMStats",
    "NullLLMProvider",
    "build_llm_provider",
]

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.retrieval.types import ContextSource


@dataclass(slots=True)
class LLMGenerationParams:
    """Runtime generation controls that should stay configurable per request."""

    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = False


@dataclass(slots=True)
class LLMRequest:
    """Provider-agnostic generation input assembled by the LLM layer."""

    query: str
    context_text: str = ""
    system_prompt: str | None = None
    sources: list[ContextSource] = field(default_factory=list)
    params: LLMGenerationParams = field(default_factory=LLMGenerationParams)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True)
class TokenUsage:
    """Token accounting normalized across backend response formats."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(slots=True)
class LLMResponse:
    """Normalized LLM output contract consumed by downstream components."""

    text: str
    model: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    raw: Any | None = None


class LLMBackend(ABC):
    """Stable interface for pluggable local or remote generation backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier for logs and eval artifacts."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for a normalized request payload."""

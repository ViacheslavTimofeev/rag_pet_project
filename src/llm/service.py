from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.llm.prompt_builder import GroundedPromptBuilder
from src.llm.types import LLMBackend, LLMGenerationParams, LLMRequest, LLMResponse
from src.retrieval.types import BuiltContext


class PromptBuilder(Protocol):
    """Minimal prompt-builder contract required by the service orchestration."""

    def build_request(
        self,
        query: str,
        built_context: BuiltContext,
        *,
        params: LLMGenerationParams | None = None,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> LLMRequest: ...


class LLMService:
    """Orchestrate prompt construction and generation backend invocation."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._backend = backend
        self._prompt_builder = prompt_builder or GroundedPromptBuilder()

    def generate(
        self,
        query: str,
        built_context: BuiltContext,
        *,
        params: LLMGenerationParams | None = None,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> LLMResponse:
        request = self._prompt_builder.build_request(
            query,
            built_context,
            params=params,
            metadata=metadata,
        )
        return self._backend.generate(request)

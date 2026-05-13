from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
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
        started_at = perf_counter()
        response = self._backend.generate(request)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 3)
        response_metadata = dict(response.raw) if isinstance(response.raw, dict) else {}
        response_metadata["llm_ms"] = elapsed_ms
        if response.raw is None or isinstance(response.raw, dict):
            response.raw = response_metadata
        return response

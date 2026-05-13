from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

from src.llm.postprocess import PostprocessedResponse, postprocess_response
from src.llm.service import LLMService
from src.llm.types import LLMGenerationParams
from src.retrieval.types import BuiltContext


class AnswerGenerationPipeline:
    """Run prompt generation and response normalization as one RAG step."""

    def __init__(
        self,
        *,
        service: LLMService,
        min_answer_chars: int = 10,
    ) -> None:
        if min_answer_chars <= 0:
            raise ValueError("min_answer_chars must be a positive integer.")

        self._service = service
        self._min_answer_chars = min_answer_chars

    def run(
        self,
        query: str,
        built_context: BuiltContext,
        *,
        params: LLMGenerationParams | None = None,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> PostprocessedResponse:
        """Generate a grounded answer and attach context-derived source metadata."""

        started_at = perf_counter()
        response = self._service.generate(
            query,
            built_context,
            params=params,
            metadata=metadata,
        )
        result = postprocess_response(
            response,
            sources=built_context.sources,
            context_text=built_context.text,
            metadata=metadata,
            min_answer_chars=self._min_answer_chars,
        )
        result.metadata["generation_ms"] = round((perf_counter() - started_at) * 1000, 3)
        return result

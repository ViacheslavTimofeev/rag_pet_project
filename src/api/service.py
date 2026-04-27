from __future__ import annotations

from src.api.schemas import AskRequest, GenerationParamsRequest
from src.llm.pipeline import AnswerGenerationPipeline
from src.llm.postprocess import PostprocessedResponse
from src.llm.types import LLMGenerationParams
from src.retrieval.pipeline import RetrievalPipeline


class RagAskService:
    """Application service that runs retrieval followed by grounded generation."""

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline,
        generation_pipeline: AnswerGenerationPipeline,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._generation_pipeline = generation_pipeline

    def answer(self, request: AskRequest) -> PostprocessedResponse:
        question = request.question.strip()
        if not question:
            raise ValueError("question must be a non-empty string.")

        built_context = self._retrieval_pipeline.run(question)
        return self._generation_pipeline.run(
            question,
            built_context,
            params=_build_generation_params(request.generation),
            metadata=request.metadata,
        )


def _build_generation_params(
    request: GenerationParamsRequest | None,
) -> LLMGenerationParams | None:
    if request is None:
        return None

    return LLMGenerationParams(
        temperature=0.0 if request.temperature is None else request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        stop=list(request.stop or []),
    )

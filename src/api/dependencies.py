from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from src.api.config import ApiConfig, load_api_config
from src.api.errors import service_unavailable
from src.api.schemas import AskRequest
from src.api.service import RagAskService
from src.llm.factory import build_llm_backend
from src.llm.pipeline import AnswerGenerationPipeline
from src.llm.postprocess import PostprocessedResponse
from src.llm.service import LLMService
from src.retrieval.factory import build_retrieval_pipeline


class AskService(Protocol):
    """Runtime service contract used by the HTTP layer."""

    def answer(self, request: AskRequest) -> PostprocessedResponse:
        """Return a grounded answer for a validated API request."""
        ...


def get_ask_service() -> AskService:
    """Resolve the configured RAG answer service.

    The concrete service is built lazily so importing the API does not load
    models, vector-store clients, or local LLM runtimes.
    """

    try:
        return build_ask_service()
    except Exception as exc:
        raise service_unavailable(f"RAG answer service is unavailable: {exc}") from exc


@lru_cache(maxsize=1)
def get_api_config() -> ApiConfig:
    """Load API runtime settings once per process."""

    return load_api_config()


@lru_cache(maxsize=1)
def build_ask_service() -> AskService:
    """Build the retrieval and generation runtime used by /ask."""

    config = get_api_config()
    retrieval_pipeline = build_retrieval_pipeline(
        config.runtime.retrieval_config_path,
    )
    llm_backend = build_llm_backend(config.runtime.model_config_path)
    generation_pipeline = AnswerGenerationPipeline(
        service=LLMService(backend=llm_backend),
    )

    return RagAskService(
        retrieval_pipeline=retrieval_pipeline,
        generation_pipeline=generation_pipeline,
    )

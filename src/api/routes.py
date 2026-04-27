from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import AskService, get_ask_service
from src.api.errors import raise_http_error
from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceResponse,
    UsageResponse,
)


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(
    request: AskRequest,
    service: AskService = Depends(get_ask_service),
) -> AskResponse:
    try:
        result = service.answer(request)
    except Exception as exc:
        raise_http_error(exc)

    usage = None
    if result.usage is not None:
        usage = UsageResponse(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        )

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                rank=source.rank,
                score=source.score,
                metadata=dict(source.metadata),
            )
            for source in result.sources
        ],
        warnings=list(result.warnings),
        model=result.model,
        finish_reason=result.finish_reason,
        usage=usage,
        metadata=dict(result.metadata),
    )

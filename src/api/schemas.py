from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, Field


PrimitiveMetadata: TypeAlias = dict[str, str | int | float | bool]


class HealthResponse(BaseModel):
    status: str


class GenerationParamsRequest(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: list[str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    generation: GenerationParamsRequest | None = None
    metadata: PrimitiveMetadata = Field(default_factory=dict)


class SourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    rank: int
    score: float
    metadata: PrimitiveMetadata = Field(default_factory=dict)


class UsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
    finish_reason: str | None = None
    usage: UsageResponse | None = None
    metadata: PrimitiveMetadata = Field(default_factory=dict)

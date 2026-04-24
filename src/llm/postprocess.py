from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.llm.types import LLMResponse, TokenUsage
from src.retrieval.types import ContextSource


@dataclass(slots=True)
class PostprocessedResponse:
    """Final response payload after normalization and lightweight checks."""

    answer: str
    sources: list[ContextSource] = field(default_factory=list)
    model: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    raw: Any | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert normalized payload into API-friendly primitive structures."""

        usage_dict: dict[str, int] | None = None
        if self.usage is not None:
            usage_dict = {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            }

        return {
            "answer": self.answer,
            "sources": [self._source_to_dict(source) for source in self.sources],
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": usage_dict,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "raw": self.raw,
        }

    def _source_to_dict(self, source: ContextSource) -> dict[str, object]:
        return {
            "chunk_id": source.chunk_id,
            "document_id": source.document_id,
            "rank": source.rank,
            "score": source.score,
            "metadata": dict(source.metadata),
        }


def normalize_answer_text(text: str) -> str:
    """Normalize line endings and trim noisy surrounding whitespace."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    compact = "\n".join(lines).strip()
    return compact


def collect_warnings(
    *,
    answer: str,
    finish_reason: str | None,
    context_text: str,
    min_answer_chars: int = 10,
) -> list[str]:
    """Generate lightweight operational warnings for downstream handling."""

    if min_answer_chars <= 0:
        raise ValueError("min_answer_chars must be a positive integer.")

    warnings: list[str] = []
    if not answer:
        warnings.append("empty_answer")
    elif len(answer) < min_answer_chars:
        warnings.append("short_answer")

    if not context_text.strip():
        warnings.append("missing_context")

    if finish_reason == "length":
        warnings.append("truncated_generation")

    return warnings


def postprocess_response(
    response: LLMResponse,
    *,
    sources: list[ContextSource] | None = None,
    context_text: str = "",
    metadata: Mapping[str, str | int | float | bool] | None = None,
    min_answer_chars: int = 10,
) -> PostprocessedResponse:
    """Build a normalized post-generation payload with warning signals."""

    normalized_answer = normalize_answer_text(response.text)
    warnings = collect_warnings(
        answer=normalized_answer,
        finish_reason=response.finish_reason,
        context_text=context_text,
        min_answer_chars=min_answer_chars,
    )

    return PostprocessedResponse(
        answer=normalized_answer,
        sources=list(sources or []),
        model=response.model,
        finish_reason=response.finish_reason,
        usage=response.usage,
        warnings=warnings,
        metadata=dict(metadata or {}),
        raw=response.raw,
    )

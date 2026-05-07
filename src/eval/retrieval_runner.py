from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

from src.retrieval.types import RetrievedChunk

from .retrieval_metrics import RetrievalMetricsSummary, evaluate_retrieval


class RetrievalEvalTarget(Protocol):
    """Structural retrieval contract required by eval runs."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        ...


@dataclass(frozen=True, slots=True)
class RetrievalEvalExample:
    """Single frozen retrieval eval item loaded from JSONL."""

    id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...] = ()
    reference_answer: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedChunkRecord:
    """Serializable retrieval hit captured during an eval run."""

    chunk_id: str
    document_id: str
    rank: int
    score: float
    text: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> RetrievedChunkRecord:
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            rank=chunk.rank,
            score=chunk.score,
            text=chunk.text,
            metadata=dict(chunk.metadata),
        )


@dataclass(frozen=True, slots=True)
class RetrievalEvalQueryResult:
    """Captured output and metrics for one eval query."""

    id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    reference_answer: str | None
    retrieved_chunks: tuple[RetrievedChunkRecord, ...]
    metrics: dict[str, dict[int, float]]


@dataclass(frozen=True, slots=True)
class RetrievalEvalRunResult:
    """Complete retrieval eval run artifact."""

    run_id: str
    created_at: str
    metadata: dict[str, Any]
    dataset_path: str | None
    k_values: tuple[int, ...]
    metrics: RetrievalMetricsSummary
    results: tuple[RetrievalEvalQueryResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_retrieval_eval_dataset(path: str | Path) -> list[RetrievalEvalExample]:
    """Load retrieval ground truth examples from a JSONL dataset."""

    dataset_path = Path(path)
    examples: list[RetrievalEvalExample] = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Eval row {line_number} must be a JSON object.")
            examples.append(_example_from_mapping(row, line_number=line_number))

    return examples


def run_retrieval_eval(
    retriever: RetrievalEvalTarget,
    examples: list[RetrievalEvalExample] | tuple[RetrievalEvalExample, ...],
    *,
    k_values: list[int] | tuple[int, ...],
    dataset_path: str | Path | None = None,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RetrievalEvalRunResult:
    """Run a retriever against frozen examples and compute retrieval metrics."""

    if not examples:
        raise ValueError("examples must contain at least one eval example.")

    captured_results: list[RetrievalEvalQueryResult] = []
    metric_rows: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

    for example in examples:
        chunks = tuple(retriever.retrieve(example.question))
        retrieved_ids = tuple(chunk.chunk_id for chunk in chunks)
        metric_rows[example.id] = (example.relevant_chunk_ids, retrieved_ids)
        captured_results.append(
            RetrievalEvalQueryResult(
                id=example.id,
                question=example.question,
                relevant_chunk_ids=example.relevant_chunk_ids,
                relevant_document_ids=example.relevant_document_ids,
                reference_answer=example.reference_answer,
                retrieved_chunks=tuple(
                    RetrievedChunkRecord.from_chunk(chunk) for chunk in chunks
                ),
                metrics={},
            )
        )

    metrics = evaluate_retrieval(metric_rows, k_values=k_values)
    captured_results = [
        _attach_query_metrics(result, metrics.per_query[result.id])
        for result in captured_results
    ]

    return RetrievalEvalRunResult(
        run_id=run_id or f"retrieval-{uuid4().hex}",
        created_at=datetime.now(UTC).isoformat(),
        metadata=dict(metadata or {}),
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        k_values=metrics.k_values,
        metrics=metrics,
        results=tuple(captured_results),
    )


def save_retrieval_eval_run(
    result: RetrievalEvalRunResult,
    path: str | Path,
) -> None:
    """Save a retrieval eval run as a deterministic JSON artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")


def _attach_query_metrics(
    result: RetrievalEvalQueryResult,
    metrics: Any,
) -> RetrievalEvalQueryResult:
    return RetrievalEvalQueryResult(
        id=result.id,
        question=result.question,
        relevant_chunk_ids=result.relevant_chunk_ids,
        relevant_document_ids=result.relevant_document_ids,
        reference_answer=result.reference_answer,
        retrieved_chunks=result.retrieved_chunks,
        metrics={
            "recall_at_k": dict(metrics.recall_at_k),
            "precision_at_k": dict(metrics.precision_at_k),
            "hit_rate_at_k": dict(metrics.hit_rate_at_k),
            "reciprocal_rank_at_k": dict(metrics.reciprocal_rank_at_k),
        },
    )


def _example_from_mapping(
    row: Mapping[str, Any],
    *,
    line_number: int,
) -> RetrievalEvalExample:
    return RetrievalEvalExample(
        id=_require_str(row, "id", line_number=line_number),
        question=_require_str(row, "question", line_number=line_number),
        relevant_chunk_ids=_require_str_tuple(
            row,
            "relevant_chunk_ids",
            line_number=line_number,
        ),
        relevant_document_ids=_get_str_tuple(row, "relevant_document_ids"),
        reference_answer=_get_optional_str(row, "reference_answer"),
    )


def _require_str(
    row: Mapping[str, Any],
    key: str,
    *,
    line_number: int,
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Eval row {line_number} field {key!r} must be a string.")
    return value


def _get_optional_str(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Field {key!r} must be a string or null.")
    return value


def _require_str_tuple(
    row: Mapping[str, Any],
    key: str,
    *,
    line_number: int,
) -> tuple[str, ...]:
    value = _get_str_tuple(row, key)
    if not value:
        raise ValueError(
            f"Eval row {line_number} field {key!r} must contain at least one id."
        )
    return value


def _get_str_tuple(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = row.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Field {key!r} must be a list of strings.")
    return tuple(value)

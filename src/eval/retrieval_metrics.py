from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SingleQueryRetrievalMetrics:
    """Retrieval metrics for one query against one ranked result list."""

    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank_at_k: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalMetricsSummary:
    """Macro-averaged retrieval metrics across an eval dataset."""

    query_count: int
    k_values: tuple[int, ...]
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    mean_reciprocal_rank_at_k: dict[int, float] = field(default_factory=dict)
    per_query: dict[str, SingleQueryRetrievalMetrics] = field(default_factory=dict)


def recall_at_k(
    relevant_ids: Iterable[str],
    retrieved_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return the fraction of relevant ids found in the first k retrieved ids."""

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    hits = relevant.intersection(_unique_top_k(retrieved_ids, k))
    return len(hits) / len(relevant)


def precision_at_k(
    relevant_ids: Iterable[str],
    retrieved_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return the fraction of first k retrieved ids that are relevant."""

    _validate_k(k)
    top_k = _unique_top_k(retrieved_ids, k)
    if not top_k:
        return 0.0

    relevant = set(relevant_ids)
    return sum(1 for item_id in top_k if item_id in relevant) / k


def hit_rate_at_k(
    relevant_ids: Iterable[str],
    retrieved_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return 1.0 when at least one relevant id appears in top k, else 0.0."""

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    return 1.0 if relevant.intersection(_unique_top_k(retrieved_ids, k)) else 0.0


def reciprocal_rank_at_k(
    relevant_ids: Iterable[str],
    retrieved_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant retrieved id within top k."""

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    for rank, item_id in enumerate(_unique_top_k(retrieved_ids, k), start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_at_k(
    rows: Iterable[tuple[Iterable[str], Sequence[str]]],
    *,
    k: int,
) -> float:
    """Return the mean reciprocal rank at k for multiple queries."""

    scores = [
        reciprocal_rank_at_k(relevant_ids, retrieved_ids, k=k)
        for relevant_ids, retrieved_ids in rows
    ]
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def evaluate_retrieval(
    eval_rows: Mapping[str, tuple[Iterable[str], Sequence[str]]],
    *,
    k_values: Sequence[int],
) -> RetrievalMetricsSummary:
    """Compute macro-averaged retrieval metrics for named eval rows.

    `eval_rows` maps query ids to `(relevant_ids, retrieved_ids)`.
    """

    normalized_k_values = _normalize_k_values(k_values)
    per_query: dict[str, SingleQueryRetrievalMetrics] = {}

    for query_id, (relevant_ids, retrieved_ids) in eval_rows.items():
        relevant = tuple(relevant_ids)
        retrieved = tuple(retrieved_ids)
        per_query[query_id] = SingleQueryRetrievalMetrics(
            recall_at_k={
                k: recall_at_k(relevant, retrieved, k=k) for k in normalized_k_values
            },
            precision_at_k={
                k: precision_at_k(relevant, retrieved, k=k)
                for k in normalized_k_values
            },
            hit_rate_at_k={
                k: hit_rate_at_k(relevant, retrieved, k=k) for k in normalized_k_values
            },
            reciprocal_rank_at_k={
                k: reciprocal_rank_at_k(relevant, retrieved, k=k)
                for k in normalized_k_values
            },
        )

    return RetrievalMetricsSummary(
        query_count=len(per_query),
        k_values=normalized_k_values,
        recall_at_k=_average_metric(per_query, "recall_at_k", normalized_k_values),
        precision_at_k=_average_metric(
            per_query, "precision_at_k", normalized_k_values
        ),
        hit_rate_at_k=_average_metric(
            per_query, "hit_rate_at_k", normalized_k_values
        ),
        mean_reciprocal_rank_at_k=_average_metric(
            per_query, "reciprocal_rank_at_k", normalized_k_values
        ),
        per_query=per_query,
    )


def _average_metric(
    per_query: Mapping[str, SingleQueryRetrievalMetrics],
    metric_name: str,
    k_values: Sequence[int],
) -> dict[int, float]:
    if not per_query:
        return {k: 0.0 for k in k_values}

    return {
        k: sum(getattr(metrics, metric_name)[k] for metrics in per_query.values())
        / len(per_query)
        for k in k_values
    }


def _normalize_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    if not k_values:
        raise ValueError("k_values must contain at least one value.")

    for k in k_values:
        _validate_k(k)

    return tuple(sorted(set(k_values)))


def _unique_top_k(retrieved_ids: Sequence[str], k: int) -> tuple[str, ...]:
    _validate_k(k)

    unique_ids: list[str] = []
    seen: set[str] = set()
    for item_id in retrieved_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        unique_ids.append(item_id)
        if len(unique_ids) == k:
            break

    return tuple(unique_ids)


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer.")

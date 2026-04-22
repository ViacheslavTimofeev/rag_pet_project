from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .types import Reranker, RetrievedChunk


class IdentityReranker(Reranker):
    """No-op reranker that keeps retrieval order and optionally trims top-k."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self._normalize_query(query)
        limited_chunks = self._apply_top_k(chunks, top_k=top_k)
        return self._reindex_ranks(limited_chunks)

    def _normalize_query(self, query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string.")
        return normalized_query

    def _apply_top_k(
        self, chunks: list[RetrievedChunk], *, top_k: int | None
    ) -> list[RetrievedChunk]:
        if top_k is None:
            return list(chunks)
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        return list(chunks[:top_k])

    def _reindex_ranks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=chunk.score,
                rank=rank,
                metadata=dict(chunk.metadata),
            )
            for rank, chunk in enumerate(chunks, start=1)
        ]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker that scores (query, chunk) pairs and reorders hits."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model = model if model is not None else self._load_model()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = self._normalize_query(query)
        limited_chunks = self._apply_top_k(chunks, top_k=top_k)
        if not limited_chunks:
            return []

        scores = self._score_chunks(normalized_query, limited_chunks)
        reranked = sorted(
            zip(limited_chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=score,
                rank=rank,
                metadata=dict(chunk.metadata),
            )
            for rank, (chunk, score) in enumerate(reranked, start=1)
        ]

    def _normalize_query(self, query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string.")
        return normalized_query

    def _apply_top_k(
        self, chunks: list[RetrievedChunk], *, top_k: int | None
    ) -> list[RetrievedChunk]:
        if top_k is None:
            return list(chunks)
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        return list(chunks[:top_k])
    
    def _load_model(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to use CrossEncoderReranker."
            ) from exc

        model_kwargs: dict[str, Any] = {}
        if self._device is not None:
            model_kwargs["device"] = self._device

        return CrossEncoder(self._model_name, **model_kwargs)

    def _score_chunks(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        pairs = [(query, chunk.text) for chunk in chunks]
        try:
            raw_scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            raw_scores = self._model.predict(pairs)

        scores = [float(score) for score in raw_scores]
        if len(scores) != len(chunks):
            raise ValueError(
                "Cross-encoder reranker returned a score count that does not "
                "match the number of input chunks."
            )
        return scores
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.vectordb.db import SearchResult


@dataclass(slots=True)
class RetrievedChunk:
    """Normalized retrieval hit returned to downstream context assembly."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, str | int] = field(default_factory=dict)

    @classmethod
    def from_search_result(
        cls, search_result: SearchResult, *, rank: int
    ) -> RetrievedChunk:
        """Convert a vector-store-specific hit into a stable retrieval contract."""

        return cls(
            chunk_id=search_result.chunk_id,
            document_id=search_result.document_id,
            text=search_result.text,
            score=search_result.score,
            rank=rank,
            metadata=dict(search_result.metadata),
        )


class SearchBackend(ABC):
    """Minimal search adapter required by the retrieval orchestration layer."""

    @abstractmethod
    def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[SearchResult]:
        """Return top-k vector search hits for the embedded user query."""


class Retriever(ABC):
    """Stable interface for retrieval backends used before generation."""

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve ranked chunks for a user query."""

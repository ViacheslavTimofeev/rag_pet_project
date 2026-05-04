from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.vectordb.db import SearchResult

PrimitiveMetadataValue = str | int | float | bool


@dataclass(slots=True)
class RetrievedChunk:
    """Normalized retrieval hit returned to downstream context assembly."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, PrimitiveMetadataValue] = field(default_factory=dict)

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


class Reranker(ABC):
    """Stable interface for reranking retrieved chunks before generation."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Reorder retrieved chunks by relevance and optionally trim to top-k."""


@dataclass(slots=True)
class ContextSource:
    """Source metadata included in built LLM context payloads."""

    chunk_id: str
    document_id: str
    rank: int
    score: float
    metadata: dict[str, PrimitiveMetadataValue] = field(default_factory=dict)


@dataclass(slots=True)
class BuiltContext:
    """Structured context payload assembled from retrieved chunks."""

    text: str
    sources: list[ContextSource] = field(default_factory=list)
    used_chunks: list[RetrievedChunk] = field(default_factory=list)


class ContextBuilder(ABC):
    """Stable interface for assembling generation-ready context payloads."""

    @abstractmethod
    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        """Build context text and source descriptors from ranked chunks."""

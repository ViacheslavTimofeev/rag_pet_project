from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from src.embeddings.types import EmbeddedChunk, EmbeddingModel
from src.ingest.types import Chunk


@dataclass(slots=True)
class IndexingResult:
    """Summary of a completed indexing run."""

    collection_name: str
    chunks_indexed: int
    embedding_model: str
    vector_dimension: int | None


class VectorStore(ABC):
    """Minimal vector store contract for index build orchestration."""

    @abstractmethod
    def create_or_update_collection(
        self, *, collection_name: str, vector_size: int | None
    ) -> None:
        """Ensure the target collection exists and matches the embedding shape."""

    @abstractmethod
    def upsert_embeddings(
        self, *, collection_name: str, chunks: Sequence[EmbeddedChunk]
    ) -> None:
        """Persist embedded chunks into the target collection."""


def build_vector_index(
    chunks: Sequence[Chunk],
    *,
    embedder: EmbeddingModel,
    vector_store: VectorStore,
    collection_name: str,
    batch_size: int = 128,
) -> IndexingResult:
    """Embed chunk batches and write them into the configured vector store."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    vector_store.create_or_update_collection(
        collection_name=collection_name,
        vector_size=embedder.dimension,
    )

    indexed_count = 0
    for batch in _batch_chunks(chunks, batch_size=batch_size):
        embedded_chunks = embedder.embed_chunks(batch)
        vector_store.upsert_embeddings(
            collection_name=collection_name,
            chunks=embedded_chunks,
        )
        indexed_count += len(embedded_chunks)

    return IndexingResult(
        collection_name=collection_name,
        chunks_indexed=indexed_count,
        embedding_model=embedder.model_name,
        vector_dimension=embedder.dimension,
    )


def _batch_chunks(chunks: Sequence[Chunk], *, batch_size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(chunks), batch_size):
        yield chunks[start : start + batch_size]

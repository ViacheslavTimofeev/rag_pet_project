from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.ingest.types import Chunk

EmbeddingVector = list[float]


@dataclass(slots=True)
class EmbeddedChunk:
    """Chunk enriched with an embedding vector for downstream indexing."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    vector: EmbeddingVector
    metadata: dict[str, str | int] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: EmbeddingVector) -> EmbeddedChunk:
        """Create an embedded representation from a chunk contract."""

        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            vector=vector,
            metadata=dict(chunk.metadata),
        )


class EmbeddingModel(ABC):
    """Stable interface for embedding backends used by indexing and retrieval."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable backend or model identifier."""

    @property
    def dimension(self) -> int | None:
        """Embedding size when known ahead of time."""

        return None

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed chunk texts or other batch inputs for indexing."""

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed a single user query using the same semantic space."""

        return self.embed_texts([text])[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunk contracts and preserve metadata for the vector store."""

        if not chunks:
            return []

        vectors = self.embed_texts([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding backend returned a vector count that does not match "
                "the number of input chunks."
            )

        return [
            EmbeddedChunk.from_chunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

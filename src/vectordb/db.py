from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.embeddings.types import EmbeddedChunk, EmbeddingVector

from .indexing import VectorStore


@dataclass(slots=True)
class SearchResult:
    """Single vector search hit returned from the backing store."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, str | int] = field(default_factory=dict)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store adapter for indexing and retrieval."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        prefer_grpc: bool = False,
        distance: str = "cosine",
        client: Any | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._prefer_grpc = prefer_grpc
        self._distance = distance
        self._client = client if client is not None else self._build_client()

    def create_or_update_collection(
        self, *, collection_name: str, vector_size: int | None
    ) -> None:
        if vector_size is None:
            raise ValueError(
                "vector_size must be known before creating a Qdrant collection."
            )

        if self._collection_exists(collection_name):
            return

        models = self._get_models_module()
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=self._parse_distance(models, self._distance),
            ),
        )

    def upsert_embeddings(
        self, *, collection_name: str, chunks: list[EmbeddedChunk] | tuple[EmbeddedChunk, ...]
    ) -> None:
        if not chunks:
            return

        models = self._get_models_module()
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=chunk.vector,
                payload=self._payload_from_chunk(chunk),
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        *,
        collection_name: str,
        query_vector: EmbeddingVector,
        limit: int = 5,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        points = getattr(response, "points", response)
        return [self._search_result_from_point(point) for point in points]

    def _build_client(self) -> Any:
        return QdrantClient(
            url=self._url,
            api_key=self._api_key,
            prefer_grpc=self._prefer_grpc,
        )

    def _get_models_module(self) -> Any:
        return models

    def _collection_exists(self, collection_name: str) -> bool:
        return bool(self._client.collection_exists(collection_name=collection_name))

    def _parse_distance(self, models: Any, distance: str) -> Any:
        normalized = distance.upper()
        try:
            return getattr(models.Distance, normalized)
        except AttributeError as exc:
            raise ValueError(f"Unsupported Qdrant distance: {distance!r}") from exc

    def _payload_from_chunk(self, chunk: EmbeddedChunk) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        }
        if chunk.metadata:
            payload["metadata"] = dict(chunk.metadata)
        return payload

    def _search_result_from_point(self, point: Any) -> SearchResult:
        payload = dict(getattr(point, "payload", {}) or {})
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return SearchResult(
            chunk_id=str(payload.get("chunk_id", getattr(point, "id", ""))),
            document_id=str(payload.get("document_id", "")),
            text=str(payload.get("text", "")),
            score=float(getattr(point, "score", 0.0)),
            metadata=metadata,
        )

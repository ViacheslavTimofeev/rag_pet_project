from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.embeddings.types import EmbeddingModel
from src.embeddings.types import EmbeddingVector

from .types import RetrievedChunk, Retriever, SearchBackend


class VectorIndexRetriever(Retriever):
    """Default retrieval path: embed a query and search the active vector index."""

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        search_backend: SearchBackend,
        collection_name: str,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        self._embedder = embedder
        self._search_backend = search_backend
        self._collection_name = collection_name
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        normalized_query = self._normalize_query(query)
        query_vector = self._embed_query(normalized_query)
        search_results = self._search(query_vector)
        return self._rank_results(search_results)

    def _normalize_query(self, query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string.")
        return normalized_query

    def _embed_query(self, query: str) -> EmbeddingVector:
        return self._embedder.embed_query(query)

    def _search(self, query_vector: EmbeddingVector):
        return self._search_backend.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=self._top_k,
        )

    def _rank_results(self, search_results: list) -> list[RetrievedChunk]:
        return [
            RetrievedChunk.from_search_result(search_result, rank=rank)
            for rank, search_result in enumerate(search_results, start=1)
        ]


class LlamaIndexRetriever(Retriever):
    """Adapter that normalizes LlamaIndex retriever results into project contracts."""

    def __init__(self, *, retriever: Any) -> None:
        if not hasattr(retriever, "retrieve"):
            raise ValueError("retriever must expose a callable 'retrieve' method.")

        self._retriever = retriever

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        normalized_query = self._normalize_query(query)
        nodes = self._retrieve_nodes(normalized_query)
        return self._rank_nodes(nodes)

    def _normalize_query(self, query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string.")
        return normalized_query

    def _retrieve_nodes(self, query: str) -> Sequence[Any]:
        return self._retriever.retrieve(query)

    def _rank_nodes(self, nodes: Sequence[Any]) -> list[RetrievedChunk]:
        return [
            self._node_to_retrieved_chunk(node_with_score, rank=rank)
            for rank, node_with_score in enumerate(nodes, start=1)
        ]

    def _node_to_retrieved_chunk(
        self, node_with_score: Any, *, rank: int
    ) -> RetrievedChunk:
        node = getattr(node_with_score, "node", node_with_score)
        metadata = self._extract_metadata(node)
        chunk_id = self._extract_chunk_id(node, rank=rank)
        document_id = self._extract_document_id(node, metadata)
        text = self._extract_text(node)
        score = float(getattr(node_with_score, "score", 0.0) or 0.0)

        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            score=score,
            rank=rank,
            metadata=metadata,
        )

    def _extract_metadata(self, node: Any) -> dict[str, str | int]:
        metadata = getattr(node, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return {}

        normalized_metadata: dict[str, str | int] = {}
        for key, value in metadata.items():
            if isinstance(key, str) and isinstance(value, (str, int)):
                normalized_metadata[key] = value
        return normalized_metadata

    def _extract_chunk_id(self, node: Any, *, rank: int) -> str:
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
        if node_id is None:
            return f"retrieved-{rank}"
        return str(node_id)

    def _extract_document_id(
        self, node: Any, metadata: dict[str, str | int]
    ) -> str:
        document_id = (
            getattr(node, "ref_doc_id", None)
            or metadata.get("document_id")
            or getattr(node, "doc_id", None)
        )
        if document_id is None:
            return ""
        return str(document_id)

    def _extract_text(self, node: Any) -> str:
        text = getattr(node, "text", None)
        if isinstance(text, str):
            return text

        get_content = getattr(node, "get_content", None)
        if callable(get_content):
            content = get_content()
            return content if isinstance(content, str) else str(content)

        return ""

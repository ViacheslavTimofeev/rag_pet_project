from __future__ import annotations

from .types import BuiltContext, ContextBuilder, Reranker, RetrievedChunk, Retriever


class RetrievalPipeline:
    """Compose retrieval, reranking, and context assembly in one runtime flow."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        rerank_top_k: int | None = None,
    ) -> None:
        if rerank_top_k is not None and rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be a positive integer or null.")

        self._retriever = retriever
        self._reranker = reranker
        self._context_builder = context_builder
        self._rerank_top_k = rerank_top_k

    def run(self, query: str) -> BuiltContext:
        reranked_chunks = self.retrieve(query)
        return self._context_builder.build(reranked_chunks)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Return reranked chunks before context assembly.

        This keeps retrieval evaluation focused on ranking quality instead of
        context-builder truncation or document deduplication policy.
        """

        retrieved_chunks = self._retriever.retrieve(query)
        return self._reranker.rerank(
            query,
            retrieved_chunks,
            top_k=self._rerank_top_k,
        )

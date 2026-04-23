from __future__ import annotations

import unittest

from src.retrieval import RetrievalPipeline
from src.retrieval.types import (
    BuiltContext,
    ContextBuilder,
    Reranker,
    RetrievedChunk,
    Retriever,
)


class RecordingRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return list(self._chunks)


class RecordingReranker(Reranker):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "top_k": top_k,
            }
        )
        return list(self._chunks)


class RecordingContextBuilder(ContextBuilder):
    def __init__(self, built_context: BuiltContext) -> None:
        self._built_context = built_context
        self.calls: list[list[str]] = []

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        self.calls.append([chunk.chunk_id for chunk in chunks])
        return self._built_context


def _chunk(*, chunk_id: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"text-{chunk_id}",
        score=score,
        rank=rank,
        metadata={"section": "body"},
    )


class RetrievalPipelineTests(unittest.TestCase):
    def test_run_executes_retrieve_rerank_and_context_build(self) -> None:
        retrieved_chunks = [
            _chunk(chunk_id="c1", score=0.8, rank=1),
            _chunk(chunk_id="c2", score=0.7, rank=2),
        ]
        reranked_chunks = [
            _chunk(chunk_id="c2", score=0.95, rank=1),
            _chunk(chunk_id="c1", score=0.50, rank=2),
        ]
        built_context = BuiltContext(text="final context")
        retriever = RecordingRetriever(retrieved_chunks)
        reranker = RecordingReranker(reranked_chunks)
        context_builder = RecordingContextBuilder(built_context)
        pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=reranker,
            context_builder=context_builder,
            rerank_top_k=2,
        )

        result = pipeline.run("what is alpha")

        self.assertIs(result, built_context)
        self.assertEqual(retriever.calls, ["what is alpha"])
        self.assertEqual(
            reranker.calls,
            [{"query": "what is alpha", "chunk_ids": ["c1", "c2"], "top_k": 2}],
        )
        self.assertEqual(context_builder.calls, [["c2", "c1"]])

    def test_init_rejects_invalid_rerank_top_k(self) -> None:
        retriever = RecordingRetriever([])
        reranker = RecordingReranker([])
        context_builder = RecordingContextBuilder(BuiltContext(text=""))

        with self.assertRaises(ValueError):
            RetrievalPipeline(
                retriever=retriever,
                reranker=reranker,
                context_builder=context_builder,
                rerank_top_k=0,
            )


if __name__ == "__main__":
    unittest.main()

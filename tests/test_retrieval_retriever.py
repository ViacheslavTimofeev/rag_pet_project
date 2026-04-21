from __future__ import annotations

from collections.abc import Sequence
import unittest

from src.embeddings.types import EmbeddingModel, EmbeddingVector
from src.retrieval import LlamaIndexRetriever, VectorIndexRetriever
from src.retrieval.types import SearchBackend
from src.vectordb.db import SearchResult


class DummyEmbedder(EmbeddingModel):
    @property
    def model_name(self) -> str:
        return "dummy-embedder"

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [[float(len(text)), 1.0] for text in texts]


class RecordingSearchBackend(SearchBackend):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit,
            }
        )
        return list(self._results)


class FakeNode:
    def __init__(
        self,
        *,
        text: str | None = None,
        node_id: str | None = None,
        ref_doc_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.text = text
        self.node_id = node_id
        self.ref_doc_id = ref_doc_id
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self.text or ""


class FakeNodeWithScore:
    def __init__(self, *, node: FakeNode, score: float | None = None) -> None:
        self.node = node
        self.score = score


class RecordingLlamaRetriever:
    def __init__(self, nodes: list[FakeNodeWithScore]) -> None:
        self._nodes = nodes
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[FakeNodeWithScore]:
        self.calls.append(query)
        return list(self._nodes)


class VectorIndexRetrieverTests(unittest.TestCase):
    def test_retrieve_embeds_query_searches_and_ranks_results(self) -> None:
        search_backend = RecordingSearchBackend(
            [
                SearchResult(
                    chunk_id="c1",
                    document_id="d1",
                    text="alpha",
                    score=0.91,
                    metadata={"section": "intro"},
                ),
                SearchResult(
                    chunk_id="c2",
                    document_id="d1",
                    text="beta",
                    score=0.83,
                    metadata={"section": "body"},
                ),
            ]
        )
        retriever = VectorIndexRetriever(
            embedder=DummyEmbedder(),
            search_backend=search_backend,
            collection_name="docs",
            top_k=2,
        )

        results = retriever.retrieve("hello")

        self.assertEqual(search_backend.calls, [{"collection_name": "docs", "query_vector": [5.0, 1.0], "limit": 2}])
        self.assertEqual([result.chunk_id for result in results], ["c1", "c2"])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].metadata["section"], "intro")

    def test_retrieve_rejects_blank_query(self) -> None:
        retriever = VectorIndexRetriever(
            embedder=DummyEmbedder(),
            search_backend=RecordingSearchBackend([]),
            collection_name="docs",
        )

        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

    def test_init_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(ValueError):
            VectorIndexRetriever(
                embedder=DummyEmbedder(),
                search_backend=RecordingSearchBackend([]),
                collection_name="docs",
                top_k=0,
            )


class LlamaIndexRetrieverTests(unittest.TestCase):
    def test_retrieve_normalizes_llamaindex_nodes(self) -> None:
        backend = RecordingLlamaRetriever(
            [
                FakeNodeWithScore(
                    node=FakeNode(
                        text="alpha",
                        node_id="c1",
                        ref_doc_id="d1",
                        metadata={"section": "intro", "page": 1, "ignored": 1.2},
                    ),
                    score=0.91,
                ),
                FakeNodeWithScore(
                    node=FakeNode(
                        text="beta",
                        node_id="c2",
                        metadata={"document_id": "d2", "section": "body"},
                    ),
                    score=None,
                ),
            ]
        )
        retriever = LlamaIndexRetriever(retriever=backend)

        results = retriever.retrieve("  hello  ")

        self.assertEqual(backend.calls, ["hello"])
        self.assertEqual([result.chunk_id for result in results], ["c1", "c2"])
        self.assertEqual([result.document_id for result in results], ["d1", "d2"])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].score, 0.91)
        self.assertEqual(results[1].score, 0.0)
        self.assertEqual(results[0].metadata, {"section": "intro", "page": 1})

    def test_retrieve_rejects_blank_query(self) -> None:
        retriever = LlamaIndexRetriever(retriever=RecordingLlamaRetriever([]))

        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

    def test_init_rejects_object_without_retrieve(self) -> None:
        with self.assertRaises(ValueError):
            LlamaIndexRetriever(retriever=object())


if __name__ == "__main__":
    unittest.main()

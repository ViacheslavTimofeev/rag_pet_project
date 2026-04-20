from __future__ import annotations

from collections.abc import Sequence
import unittest

from src.embeddings.types import EmbeddedChunk, EmbeddingModel, EmbeddingVector
from src.ingest.types import Chunk
from src.vectordb import VectorStore, build_vector_index


class DummyEmbedder(EmbeddingModel):
    @property
    def model_name(self) -> str:
        return "dummy-embedder"

    @property
    def dimension(self) -> int:
        return 2

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [[float(len(text)), 1.0] for text in texts]


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self.collections: list[tuple[str, int | None]] = []
        self.upserts: list[tuple[str, list[EmbeddedChunk]]] = []

    def create_or_update_collection(
        self, *, collection_name: str, vector_size: int | None
    ) -> None:
        self.collections.append((collection_name, vector_size))

    def upsert_embeddings(
        self, *, collection_name: str, chunks: Sequence[EmbeddedChunk]
    ) -> None:
        self.upserts.append((collection_name, list(chunks)))


class VectorDbIndexingTests(unittest.TestCase):
    def test_build_vector_index_embeds_and_upserts_batches(self) -> None:
        chunks = [
            Chunk(chunk_id="c1", document_id="d1", chunk_index=0, text="alpha"),
            Chunk(chunk_id="c2", document_id="d1", chunk_index=1, text="beta"),
            Chunk(chunk_id="c3", document_id="d2", chunk_index=0, text="gamma"),
        ]
        embedder = DummyEmbedder()
        vector_store = InMemoryVectorStore()

        result = build_vector_index(
            chunks,
            embedder=embedder,
            vector_store=vector_store,
            collection_name="docs",
            batch_size=2,
        )

        self.assertEqual(vector_store.collections, [("docs", 2)])
        self.assertEqual(len(vector_store.upserts), 2)
        self.assertEqual(vector_store.upserts[0][0], "docs")
        self.assertEqual([chunk.chunk_id for chunk in vector_store.upserts[0][1]], ["c1", "c2"])
        self.assertEqual(vector_store.upserts[1][0], "docs")
        self.assertEqual([chunk.chunk_id for chunk in vector_store.upserts[1][1]], ["c3"])
        self.assertEqual(result.collection_name, "docs")
        self.assertEqual(result.chunks_indexed, 3)
        self.assertEqual(result.embedding_model, "dummy-embedder")
        self.assertEqual(result.vector_dimension, 2)

    def test_build_vector_index_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            build_vector_index(
                [],
                embedder=DummyEmbedder(),
                vector_store=InMemoryVectorStore(),
                collection_name="docs",
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()

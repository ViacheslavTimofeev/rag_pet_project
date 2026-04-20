from __future__ import annotations

from dataclasses import dataclass
import unittest
from unittest.mock import patch

from src.embeddings.types import EmbeddedChunk
from src.vectordb.db import QdrantVectorStore


class FakeModels:
    class Distance:
        COSINE = "cosine"

    @dataclass
    class VectorParams:
        size: int
        distance: str

    @dataclass
    class PointStruct:
        id: str
        vector: list[float]
        payload: dict


class FakeClient:
    def __init__(self, *, collection_exists: bool = False, points=None) -> None:
        self._collection_exists = collection_exists
        self._points = points or []
        self.created_collections = []
        self.upserts = []
        self.queries = []

    def collection_exists(self, *, collection_name: str) -> bool:
        return self._collection_exists

    def create_collection(self, **kwargs) -> None:
        self.created_collections.append(kwargs)

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return type("QueryResponse", (), {"points": self._points})()


class FakePoint:
    def __init__(self, *, point_id: str, score: float, payload: dict) -> None:
        self.id = point_id
        self.score = score
        self.payload = payload


class QdrantVectorStoreTests(unittest.TestCase):
    def test_create_collection_when_missing(self) -> None:
        client = FakeClient(collection_exists=False)
        store = QdrantVectorStore(client=client)

        with patch.object(store, "_get_models_module", return_value=FakeModels):
            store.create_or_update_collection(collection_name="docs", vector_size=384)

        self.assertEqual(len(client.created_collections), 1)
        created = client.created_collections[0]
        self.assertEqual(created["collection_name"], "docs")
        self.assertEqual(created["vectors_config"].size, 384)
        self.assertEqual(created["vectors_config"].distance, "cosine")

    def test_upsert_embeddings_converts_chunks_to_points(self) -> None:
        client = FakeClient()
        store = QdrantVectorStore(client=client)
        chunks = [
            EmbeddedChunk(
                chunk_id="c1",
                document_id="d1",
                chunk_index=0,
                text="hello",
                vector=[1.0, 2.0],
                metadata={"section": "intro"},
            )
        ]

        with patch.object(store, "_get_models_module", return_value=FakeModels):
            store.upsert_embeddings(collection_name="docs", chunks=chunks)

        self.assertEqual(len(client.upserts), 1)
        point = client.upserts[0]["points"][0]
        self.assertEqual(point.id, "c1")
        self.assertEqual(point.vector, [1.0, 2.0])
        self.assertEqual(point.payload["metadata"]["section"], "intro")

    def test_search_maps_qdrant_points_to_search_results(self) -> None:
        client = FakeClient(
            points=[
                FakePoint(
                    point_id="c1",
                    score=0.93,
                    payload={
                        "chunk_id": "c1",
                        "document_id": "d1",
                        "text": "hello",
                        "metadata": {"section": "intro"},
                    },
                )
            ]
        )
        store = QdrantVectorStore(client=client)

        results = store.search(
            collection_name="docs",
            query_vector=[0.1, 0.2],
            limit=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertEqual(results[0].document_id, "d1")
        self.assertEqual(results[0].text, "hello")
        self.assertEqual(results[0].score, 0.93)
        self.assertEqual(results[0].metadata["section"], "intro")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from typing import Any

from src.retrieval import CrossEncoderReranker, IdentityReranker
from src.retrieval.types import RetrievedChunk


class RecordingCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[dict[str, Any]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"pairs": pairs, "kwargs": kwargs})
        activation_fct = kwargs.get("activation_fn") or kwargs.get("activation_fct")
        if activation_fct is None:
            return list(self._scores)
        return [activation_fct(score) for score in self._scores]


def _chunk(*, chunk_id: str, text: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=score,
        rank=rank,
        metadata={"section": "body"},
    )


class IdentityRerankerTests(unittest.TestCase):
    def test_rerank_keeps_order_applies_top_k_and_reindexes_ranks(self) -> None:
        reranker = IdentityReranker()
        chunks = [
            _chunk(chunk_id="c1", text="alpha", score=0.9, rank=8),
            _chunk(chunk_id="c2", text="beta", score=0.8, rank=9),
            _chunk(chunk_id="c3", text="gamma", score=0.7, rank=10),
        ]

        results = reranker.rerank("hello", chunks, top_k=2)

        self.assertEqual([chunk.chunk_id for chunk in results], ["c1", "c2"])
        self.assertEqual([chunk.rank for chunk in results], [1, 2])
        self.assertEqual([chunk.score for chunk in results], [0.9, 0.8])

    def test_rerank_rejects_blank_query(self) -> None:
        reranker = IdentityReranker()

        with self.assertRaises(ValueError):
            reranker.rerank("   ", [])

    def test_rerank_rejects_invalid_top_k(self) -> None:
        reranker = IdentityReranker()

        with self.assertRaises(ValueError):
            reranker.rerank("hello", [], top_k=0)


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_rerank_scores_sorts_and_reindexes(self) -> None:
        backend = RecordingCrossEncoder([0.1, 0.95, 0.4])
        reranker = CrossEncoderReranker(
            "fake-cross-encoder",
            batch_size=16,
            model=backend,
        )
        chunks = [
            _chunk(chunk_id="c1", text="alpha", score=0.9, rank=1),
            _chunk(chunk_id="c2", text="beta", score=0.8, rank=2),
            _chunk(chunk_id="c3", text="gamma", score=0.7, rank=3),
        ]

        results = reranker.rerank("  what is beta  ", chunks)

        self.assertEqual(
            backend.calls,
            [
                {
                    "pairs": [
                        ("what is beta", "alpha"),
                        ("what is beta", "beta"),
                        ("what is beta", "gamma"),
                    ],
                    "kwargs": {
                        "batch_size": 16,
                        "show_progress_bar": False,
                        "convert_to_numpy": True,
                    },
                }
            ],
        )
        self.assertEqual([chunk.chunk_id for chunk in results], ["c2", "c3", "c1"])
        self.assertEqual([chunk.rank for chunk in results], [1, 2, 3])
        self.assertEqual([chunk.score for chunk in results], [0.95, 0.4, 0.1])
        self.assertEqual(results[0].metadata["section"], "body")
        self.assertEqual(results[0].metadata["retrieval_score"], 0.8)
        self.assertEqual(results[0].metadata["reranker_score"], 0.95)
        self.assertEqual(
            results[0].metadata["reranker_model"],
            "fake-cross-encoder",
        )

    def test_init_rejects_invalid_max_length(self) -> None:
        with self.assertRaises(ValueError):
            CrossEncoderReranker(
                "fake-cross-encoder",
                max_length=0,
                model=object(),
            )

    def test_rerank_can_apply_sigmoid_to_scores(self) -> None:
        backend = RecordingCrossEncoder([0.0])
        reranker = CrossEncoderReranker(
            "fake-cross-encoder",
            use_sigmoid=True,
            model=backend,
        )
        chunks = [_chunk(chunk_id="c1", text="alpha", score=0.9, rank=1)]

        results = reranker.rerank("hello", chunks)

        self.assertEqual(results[0].score, 0.5)
        self.assertIn("activation_fn", backend.calls[0]["kwargs"])

    def test_rerank_respects_top_k_before_scoring(self) -> None:
        backend = RecordingCrossEncoder([0.2, 0.1])
        reranker = CrossEncoderReranker("fake-cross-encoder", model=backend)
        chunks = [
            _chunk(chunk_id="c1", text="alpha", score=0.9, rank=1),
            _chunk(chunk_id="c2", text="beta", score=0.8, rank=2),
            _chunk(chunk_id="c3", text="gamma", score=0.7, rank=3),
        ]

        results = reranker.rerank("hello", chunks, top_k=2)

        self.assertEqual([chunk.chunk_id for chunk in results], ["c1", "c2"])
        self.assertEqual(len(backend.calls[0]["pairs"]), 2)

    def test_rerank_rejects_invalid_score_count(self) -> None:
        backend = RecordingCrossEncoder([0.5])
        reranker = CrossEncoderReranker("fake-cross-encoder", model=backend)
        chunks = [
            _chunk(chunk_id="c1", text="alpha", score=0.9, rank=1),
            _chunk(chunk_id="c2", text="beta", score=0.8, rank=2),
        ]

        with self.assertRaises(ValueError):
            reranker.rerank("hello", chunks)

    def test_init_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            CrossEncoderReranker("fake-cross-encoder", batch_size=0, model=object())

    def test_rerank_rejects_blank_query(self) -> None:
        reranker = CrossEncoderReranker("fake-cross-encoder", model=RecordingCrossEncoder([]))

        with self.assertRaises(ValueError):
            reranker.rerank("   ", [])

    def test_rerank_rejects_invalid_top_k(self) -> None:
        reranker = CrossEncoderReranker("fake-cross-encoder", model=RecordingCrossEncoder([]))

        with self.assertRaises(ValueError):
            reranker.rerank("hello", [], top_k=0)


if __name__ == "__main__":
    unittest.main()

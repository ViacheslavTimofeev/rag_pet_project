from __future__ import annotations

import unittest

from src.retrieval import TokenBudgetContextBuilder
from src.retrieval.types import RetrievedChunk


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    text: str,
    score: float,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        rank=rank,
        metadata={"section": "body"},
    )


class TokenBudgetContextBuilderTests(unittest.TestCase):
    def test_build_returns_empty_context_for_empty_input(self) -> None:
        builder = TokenBudgetContextBuilder()

        context = builder.build([])

        self.assertEqual(context.text, "")
        self.assertEqual(context.sources, [])
        self.assertEqual(context.used_chunks, [])

    def test_build_limits_number_of_chunks(self) -> None:
        builder = TokenBudgetContextBuilder(max_chunks=2, max_chars=None)
        chunks = [
            _chunk(chunk_id="c1", document_id="d1", text="alpha", score=0.9, rank=1),
            _chunk(chunk_id="c2", document_id="d2", text="beta", score=0.8, rank=2),
            _chunk(chunk_id="c3", document_id="d3", text="gamma", score=0.7, rank=3),
        ]

        context = builder.build(chunks)

        self.assertEqual([chunk.chunk_id for chunk in context.used_chunks], ["c1", "c2"])
        self.assertEqual(context.text, "alpha\n\nbeta")

    def test_build_limits_by_character_budget(self) -> None:
        builder = TokenBudgetContextBuilder(max_chunks=None, max_chars=10)
        chunks = [
            _chunk(chunk_id="c1", document_id="d1", text="alpha", score=0.9, rank=1),
            _chunk(chunk_id="c2", document_id="d2", text="beta", score=0.8, rank=2),
            _chunk(chunk_id="c3", document_id="d3", text="zz", score=0.7, rank=3),
        ]

        context = builder.build(chunks)

        self.assertEqual([chunk.chunk_id for chunk in context.used_chunks], ["c1", "c3"])
        self.assertEqual(context.text, "alpha\n\nzz")

    def test_build_deduplicates_by_document_id(self) -> None:
        builder = TokenBudgetContextBuilder(max_chunks=None, max_chars=None)
        chunks = [
            _chunk(chunk_id="c1", document_id="d1", text="alpha", score=0.9, rank=1),
            _chunk(chunk_id="c2", document_id="d1", text="beta", score=0.8, rank=2),
            _chunk(chunk_id="c3", document_id="d2", text="gamma", score=0.7, rank=3),
        ]

        context = builder.build(chunks)

        self.assertEqual([chunk.chunk_id for chunk in context.used_chunks], ["c1", "c3"])
        self.assertEqual([source.document_id for source in context.sources], ["d1", "d2"])

    def test_build_can_disable_document_dedup(self) -> None:
        builder = TokenBudgetContextBuilder(
            max_chunks=None,
            max_chars=None,
            dedup_by_document=False,
        )
        chunks = [
            _chunk(chunk_id="c1", document_id="d1", text="alpha", score=0.9, rank=1),
            _chunk(chunk_id="c2", document_id="d1", text="beta", score=0.8, rank=2),
        ]

        context = builder.build(chunks)

        self.assertEqual([chunk.chunk_id for chunk in context.used_chunks], ["c1", "c2"])

    def test_build_preserves_source_fields(self) -> None:
        builder = TokenBudgetContextBuilder(max_chunks=1, max_chars=None)
        chunk = RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="alpha",
            score=0.91,
            rank=3,
            metadata={"section": "intro", "page": 1},
        )

        context = builder.build([chunk])

        self.assertEqual(context.sources[0].chunk_id, "c1")
        self.assertEqual(context.sources[0].document_id, "d1")
        self.assertEqual(context.sources[0].rank, 3)
        self.assertEqual(context.sources[0].score, 0.91)
        self.assertEqual(context.sources[0].metadata, {"section": "intro", "page": 1})

    def test_init_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            TokenBudgetContextBuilder(max_chunks=0)
        with self.assertRaises(ValueError):
            TokenBudgetContextBuilder(max_chars=0)
        with self.assertRaises(ValueError):
            TokenBudgetContextBuilder(chunk_separator="")


if __name__ == "__main__":
    unittest.main()

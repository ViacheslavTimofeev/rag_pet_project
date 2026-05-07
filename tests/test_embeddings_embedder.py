from __future__ import annotations

import unittest

from src.embeddings.embedder import SentenceTransformerEmbedder


class FakeNewDimensionModel:
    def get_embedding_dimension(self) -> int:
        return 384


class FakeLegacyDimensionModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 768


class SentenceTransformerEmbedderTests(unittest.TestCase):
    def test_dimension_uses_new_sentence_transformers_method(self) -> None:
        embedder = SentenceTransformerEmbedder(
            model_name="fake",
            model=FakeNewDimensionModel(),
        )

        self.assertEqual(embedder.dimension, 384)

    def test_dimension_keeps_legacy_method_fallback(self) -> None:
        embedder = SentenceTransformerEmbedder(
            model_name="fake",
            model=FakeLegacyDimensionModel(),
        )

        self.assertEqual(embedder.dimension, 768)


if __name__ == "__main__":
    unittest.main()

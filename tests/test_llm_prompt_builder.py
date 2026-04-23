from __future__ import annotations

import unittest

from src.llm.prompt_builder import GroundedPromptBuilder, PromptBuilderConfig
from src.retrieval.types import BuiltContext, ContextSource, RetrievedChunk


def _chunk(*, chunk_id: str, document_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=0.9,
        rank=rank,
        metadata={"section": "body"},
    )


class GroundedPromptBuilderTests(unittest.TestCase):
    def test_build_request_formats_context_with_source_labels(self) -> None:
        builder = GroundedPromptBuilder()
        chunks = [
            _chunk(chunk_id="c1", document_id="d1", text="alpha", rank=1),
            _chunk(chunk_id="c2", document_id="d2", text="beta", rank=2),
        ]
        context = BuiltContext(
            text="alpha\n\nbeta",
            sources=[
                ContextSource(chunk_id="c1", document_id="d1", rank=1, score=0.9),
                ContextSource(chunk_id="c2", document_id="d2", rank=2, score=0.8),
            ],
            used_chunks=chunks,
        )

        request = builder.build_request("What is this?", context)

        self.assertEqual(request.query, "What is this?")
        self.assertEqual(request.sources, context.sources)
        self.assertIn("[1] doc=d1 chunk=c1 rank=1", request.context_text)
        self.assertIn("[2] doc=d2 chunk=c2 rank=2", request.context_text)
        self.assertIn("alpha", request.context_text)
        self.assertIn("beta", request.context_text)

    def test_build_request_handles_empty_context(self) -> None:
        builder = GroundedPromptBuilder()
        request = builder.build_request("hello", BuiltContext(text=""))
        self.assertEqual(request.context_text, "")

    def test_build_request_applies_context_char_limit(self) -> None:
        builder = GroundedPromptBuilder(
            PromptBuilderConfig(include_source_labels=False, max_context_chars=20)
        )
        context = BuiltContext(text="abcdefghijklmnopqrstuvwxyz")

        request = builder.build_request("hello", context)

        self.assertTrue(request.context_text.endswith("...[truncated]"))
        self.assertLessEqual(len(request.context_text), 20)

    def test_build_request_rejects_blank_query(self) -> None:
        builder = GroundedPromptBuilder()

        with self.assertRaises(ValueError):
            builder.build_request("   ", BuiltContext(text="alpha"))

    def test_init_rejects_invalid_max_context_chars(self) -> None:
        with self.assertRaises(ValueError):
            GroundedPromptBuilder(PromptBuilderConfig(max_context_chars=0))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from typing import Any, cast

from src.llm.postprocess import (
    collect_warnings,
    normalize_answer_text,
    postprocess_response,
)
from src.llm.types import LLMResponse, TokenUsage
from src.retrieval.types import ContextSource


class LLMPostprocessTests(unittest.TestCase):
    def test_normalize_answer_text_trims_and_normalizes_lines(self) -> None:
        text = " \r\nHello   \r\nWorld\t \r\n "
        normalized = normalize_answer_text(text)
        self.assertEqual(normalized, "Hello\nWorld")

    def test_collect_warnings_flags_short_and_missing_context(self) -> None:
        warnings = collect_warnings(
            answer="short",
            finish_reason=None,
            context_text="",
            min_answer_chars=10,
        )
        self.assertEqual(warnings, ["short_answer", "missing_context"])

    def test_collect_warnings_flags_truncated_generation(self) -> None:
        warnings = collect_warnings(
            answer="This is a long enough answer.",
            finish_reason="length",
            context_text="context",
            min_answer_chars=10,
        )
        self.assertEqual(warnings, ["truncated_generation"])

    def test_postprocess_response_builds_structured_payload(self) -> None:
        response = LLMResponse(
            text="\nanswer\n",
            model="qwen",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=8, output_tokens=3, total_tokens=11),
            raw={"trace": "abc"},
        )
        source = ContextSource(
            chunk_id="c1",
            document_id="d1",
            rank=1,
            score=0.91,
            metadata={"section": "intro"},
        )

        payload = postprocess_response(
            response,
            sources=[source],
            context_text="context",
            metadata={"trace_id": "123"},
        )
        as_dict = payload.to_dict()

        self.assertEqual(payload.answer, "answer")
        self.assertEqual(payload.model, "qwen")
        self.assertEqual(payload.warnings, ["short_answer"])
        self.assertEqual(as_dict["metadata"], {"trace_id": "123"})
        self.assertEqual(as_dict["usage"], {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11})
        sources = cast(list[dict[str, Any]], as_dict["sources"])
        self.assertEqual(sources[0]["chunk_id"], "c1")

    def test_collect_warnings_rejects_invalid_threshold(self) -> None:
        with self.assertRaises(ValueError):
            collect_warnings(
                answer="ok",
                finish_reason=None,
                context_text="context",
                min_answer_chars=0,
            )


if __name__ == "__main__":
    unittest.main()

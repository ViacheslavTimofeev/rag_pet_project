from __future__ import annotations

import argparse
import json
import shutil
import unittest
from pathlib import Path

from scripts.build_markdown_gt import run


class BuildMarkdownGtScriptTest(unittest.TestCase):
    def test_builds_markdown_parent_and_chunk_labels_from_source_gt(self) -> None:
        root = Path(".tmp_test_build_markdown_gt")
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        try:
            source_gt = root / "retrieval_sources_gt.jsonl"
            chunks = root / "chunks_markdown.jsonl"
            output = root / "retrieval_markdown_gt.jsonl"

            source_gt.write_text(
                json.dumps(
                    {
                        "id": "q001",
                        "question": "How do I receive an uploaded file?",
                        "relevant_document_ids": ["request-files-md"],
                        "reference_answer": (
                            "Use UploadFile or bytes with File to receive file uploads."
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            chunk_rows = [
                {
                    "chunk_id": "request-files-md-chunk-0000",
                    "document_id": "request-files-md",
                    "text": "Section: Request Files\nUse File and UploadFile.",
                    "metadata": {
                        "parent_id": "request-files-md-parent-0000",
                        "section_path": "Request Files",
                        "symbols": "UploadFile, File",
                    },
                },
                {
                    "chunk_id": "request-files-md-chunk-0001",
                    "document_id": "request-files-md",
                    "text": "Section: Request Files\nUploadFile exposes filename.",
                    "metadata": {
                        "parent_id": "request-files-md-parent-0000",
                        "section_path": "Request Files",
                        "symbols": "UploadFile",
                    },
                },
                {
                    "chunk_id": "body-md-chunk-0000",
                    "document_id": "body-md",
                    "text": "Section: Body\nDeclare Pydantic models.",
                    "metadata": {
                        "parent_id": "body-md-parent-0000",
                        "section_path": "Body",
                        "symbols": "BaseModel",
                    },
                },
            ]
            chunks.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in chunk_rows
                ),
                encoding="utf-8",
            )

            exit_code = run(
                argparse.Namespace(
                    source_gt=source_gt,
                    chunks=chunks,
                    output=output,
                    max_parents_per_document=2,
                    max_chunks_per_parent=2,
                    relative_threshold=0.65,
                )
            )

            self.assertEqual(exit_code, 0)
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relevant_document_ids"], ["request-files-md"])
            self.assertEqual(
                rows[0]["relevant_parent_ids"],
                ["request-files-md-parent-0000"],
            )
            self.assertTrue(rows[0]["relevant_chunk_ids"])
            self.assertTrue(
                all(
                    chunk_id.startswith("request-files-md-chunk-")
                    for chunk_id in rows[0]["relevant_chunk_ids"]
                )
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

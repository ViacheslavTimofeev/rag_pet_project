from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
import uuid

from scripts import ingest


class IngestScriptTests(unittest.TestCase):
    def test_default_character_strategy_writes_chunks(self) -> None:
        tmp_root = Path(".tmp_test") / f"ingest-script-character-{uuid.uuid4().hex}"
        try:
            source_dir = tmp_root / "raw"
            output_dir = tmp_root / "processed"
            source_dir.mkdir(parents=True)
            (source_dir / "intro.md").write_text(
                "# Intro\n\nFastAPI is a framework.\n\nIt uses type hints.",
                encoding="utf-8",
            )
            args = ingest.build_parser().parse_args(
                [
                    str(source_dir),
                    str(output_dir),
                    "--chunk-size",
                    "40",
                    "--chunk-overlap",
                    "5",
                ]
            )

            exit_code = ingest.run(args)

            chunks = _read_jsonl(output_dir / "chunks.jsonl")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("char_start", chunks[0]["metadata"])

    def test_markdown_parent_child_strategy_writes_parent_metadata(self) -> None:
        tmp_root = Path(".tmp_test") / f"ingest-script-markdown-{uuid.uuid4().hex}"
        try:
            source_dir = tmp_root / "raw"
            output_dir = tmp_root / "processed"
            source_dir.mkdir(parents=True)
            (source_dir / "body.md").write_text(
                (
                    "# Body\n\n"
                    "Intro.\n\n"
                    "## Request body\n\n"
                    "Use `BaseModel` with request bodies in FastAPI."
                ),
                encoding="utf-8",
            )
            args = ingest.build_parser().parse_args(
                [
                    str(source_dir),
                    str(output_dir),
                    "--chunk-strategy",
                    "markdown-parent-child",
                    "--child-chunk-size",
                    "120",
                    "--child-chunk-overlap",
                    "20",
                ]
            )

            exit_code = ingest.run(args)

            chunks = _read_jsonl(output_dir / "chunks.jsonl")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(chunks), 1)
        metadata = chunks[0]["metadata"]
        self.assertIn("parent_id", metadata)
        self.assertIn("section_path", metadata)
        self.assertIn("Section:", chunks[0]["text"])


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()

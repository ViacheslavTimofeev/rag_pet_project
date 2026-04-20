from pathlib import Path
import shutil
from typing import cast
import unittest
import uuid

from src.ingest import CharacterTextChunker
from src.ingest.raw_creation import FastAPITutorialIngestor
from src.ingest.types import RawDocument


class FastAPITutorialIngestorTests(unittest.TestCase):
    def test_loads_supported_files_and_normalizes_content(self) -> None:
        tmp_root = Path(".tmp_test") / f"ingest-{uuid.uuid4().hex}"
        try:
            root = tmp_root
            docs_dir = root / "tutorial" / "first-steps"
            docs_dir.mkdir(parents=True)

            (docs_dir / "index.html").write_text(
                "<html><body><h1>Hello</h1><p>FastAPI tutorial</p><script>bad()</script></body></html>",
                encoding="utf-8",
            )
            (docs_dir / "notes.md").write_text("# Notes\n\nUse dependency injection.", encoding="utf-8")
            (docs_dir / "ignore.json").write_text('{"skip": true}', encoding="utf-8")

            ingestor = FastAPITutorialIngestor(source_dir=root)
            documents = ingestor.load()
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(len(documents), 2)

        first_doc = documents[0]
        self.assertEqual(first_doc.document_id, "tutorial-first-steps-index-html")
        self.assertEqual(first_doc.metadata["dataset"], "fastapi_tutorial")
        self.assertEqual(first_doc.metadata["section"], "tutorial/first-steps")
        self.assertEqual(first_doc.metadata["source"], "tutorial/first-steps/index.html")
        self.assertEqual(first_doc.text, "Hello FastAPI tutorial")

        second_doc = documents[1]
        self.assertIn("Use dependency injection.", second_doc.text)
        self.assertEqual(second_doc.metadata["title"], "notes")


class CharacterTextChunkerTests(unittest.TestCase):
    def test_chunks_document_and_preserves_document_metadata(self) -> None:
        chunker = CharacterTextChunker(chunk_size=60, chunk_overlap=10)
        document = RawDocument(
            document_id="fastapi-overview",
            source_path=Path("docs/overview.md"),
            text=(
                "FastAPI is a modern web framework.\n\n"
                "It is built for speed and clean type hints.\n\n"
                "Dependency injection is one of its core patterns."
            ),
            metadata={"dataset": "fastapi_tutorial", "section": "tutorial"},
        )

        chunks = chunker.chunk_single_document(document)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].document_id, document.document_id)
        self.assertEqual(chunks[0].metadata["dataset"], "fastapi_tutorial")
        self.assertEqual(chunks[0].metadata["section"], "tutorial")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].metadata["chunk_count"], len(chunks))
        self.assertEqual(chunks[0].metadata["prev_chunk_id"], "")
        self.assertEqual(chunks[0].metadata["next_chunk_id"], chunks[1].chunk_id)
        self.assertEqual(chunks[-1].metadata["next_chunk_id"], "")
        self.assertTrue(chunks[0].chunk_id.startswith("fastapi-overview-chunk-"))

    def test_chunk_many_documents_flattens_multiple_documents(self) -> None:
        chunker = CharacterTextChunker(chunk_size=20, chunk_overlap=5)
        documents = [
            RawDocument(document_id="doc-one", source_path=Path("one.txt"), text="A" * 30),
            RawDocument(document_id="doc-two", source_path=Path("two.txt"), text="B" * 10),
        ]

        chunks = chunker.chunk_many_documents(documents)

        self.assertEqual([chunk.document_id for chunk in chunks], ["doc-one", "doc-one", "doc-two"])

    def test_rejects_invalid_overlap_configuration(self) -> None:
        with self.assertRaises(ValueError):
            CharacterTextChunker(chunk_size=50, chunk_overlap=50)

    def test_aligns_next_chunk_start_to_word_boundary(self) -> None:
        chunker = CharacterTextChunker(chunk_size=40, chunk_overlap=12)
        document = RawDocument(
            document_id="doc-boundary",
            source_path=Path("docs/boundary.md"),
            text="## Using `BackgroundTasks`\n\nFirst, import `BackgroundTasks` and define a parameter.",
        )

        chunks = chunker.chunk_single_document(document)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("First, import"))
        for chunk in chunks[1:]:
            char_start = cast(int, chunk.metadata["char_start"])
            self.assertIsInstance(char_start, int)
            self.assertFalse(
                document.text[char_start - 1].isalnum() and document.text[char_start].isalnum(),
                msg=f"Chunk starts in the middle of a word: {chunk.text!r}",
            )


if __name__ == "__main__":
    unittest.main()

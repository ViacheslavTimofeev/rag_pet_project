from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

from scripts.index import (
    get_model_config_path,
    get_qdrant_config,
    load_chunks,
)


class IndexScriptTests(unittest.TestCase):
    def test_load_chunks_reads_jsonl_chunks(self) -> None:
        tmp_root = Path(".tmp_test") / f"index-script-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            chunks_path = tmp_root / "chunks.jsonl"
            chunks_path.write_text(
                (
                    '{"chunk_id":"c1","document_id":"d1","chunk_index":0,'
                    '"text":"hello","metadata":{"section":"intro"}}\n'
                ),
                encoding="utf-8",
            )

            chunks = load_chunks(chunks_path)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "c1")
        self.assertEqual(chunks[0].metadata, {"section": "intro"})

    def test_get_qdrant_config_reads_nested_retrieval_config(self) -> None:
        config = {
            "retrieval": {
                "llamaindex": {
                    "model_config_path": "configs/model.yaml",
                    "qdrant": {
                        "collection_name": "documents",
                        "url": "http://localhost:6333",
                    },
                }
            }
        }

        self.assertEqual(
            get_qdrant_config(config),
            {
                "collection_name": "documents",
                "url": "http://localhost:6333",
            },
        )
        self.assertEqual(get_model_config_path(config), Path("configs/model.yaml"))


if __name__ == "__main__":
    unittest.main()

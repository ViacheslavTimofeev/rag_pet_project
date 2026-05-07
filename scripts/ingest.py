from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest.chunking import CharacterTextChunker, MarkdownParentChildChunker
from src.ingest.raw_creation import FastAPITutorialIngestor
from src.ingest.types import Chunk, RawDocument


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local ingestion pipeline: normalize raw docs and chunk them."
    )
    parser.add_argument("source_dir", type=Path, help="Directory with raw source files.")
    parser.add_argument("output_dir", type=Path, help="Directory for processed ingestion artifacts.")
    parser.add_argument(
        "--chunk-strategy",
        choices=["character", "markdown-parent-child"],
        default="character",
        help="Chunking strategy to use for processed chunks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Character chunk size for the character strategy.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character chunk overlap for the character strategy.",
    )
    parser.add_argument(
        "--child-chunk-size",
        type=int,
        default=800,
        help="Child chunk size for the markdown-parent-child strategy.",
    )
    parser.add_argument(
        "--child-chunk-overlap",
        type=int,
        default=120,
        help="Child chunk overlap for the markdown-parent-child strategy.",
    )
    parser.add_argument(
        "--no-header-prefix",
        action="store_true",
        help="Disable section/symbol prefix in markdown child chunk text.",
    )
    parser.add_argument(
        "--no-symbols",
        action="store_true",
        help="Disable symbol extraction for markdown parent sections.",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


def run(args: argparse.Namespace) -> int:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    ingestor = FastAPITutorialIngestor(source_dir=source_dir)
    documents = ingestor.load()

    chunker = build_chunker(args)
    chunks = chunker.chunk_many_documents(documents)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_documents_path = output_dir / "raw_documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    _write_jsonl(raw_documents_path, (_serialize_raw_document(document) for document in documents))
    _write_jsonl(chunks_path, (_serialize_chunk(chunk) for chunk in chunks))

    print(f"Loaded {len(documents)} documents from {source_dir}")
    print(f"Chunking strategy: {args.chunk_strategy}")
    print(f"Wrote normalized documents to {raw_documents_path}")
    print(f"Created {len(chunks)} chunks")
    print(f"Wrote chunks to {chunks_path}")
    return 0


def build_chunker(args: argparse.Namespace):
    if args.chunk_strategy == "character":
        return CharacterTextChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    if args.chunk_strategy == "markdown-parent-child":
        return MarkdownParentChildChunker(
            child_chunk_size=args.child_chunk_size,
            child_chunk_overlap=args.child_chunk_overlap,
            include_header_prefix=not args.no_header_prefix,
            extract_symbols=not args.no_symbols,
        )
    raise ValueError(f"Unsupported chunk strategy: {args.chunk_strategy}")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _serialize_raw_document(document: RawDocument) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "source_path": str(document.source_path),
        "text": document.text,
        "metadata": document.metadata,
    }


def _serialize_chunk(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


if __name__ == "__main__":
    raise SystemExit(main())

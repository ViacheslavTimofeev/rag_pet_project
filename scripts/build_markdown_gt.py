from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar, cast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import safe_sparse_dot


DEFAULT_SOURCE_GT_PATH = Path("data/eval/retrieval_sources_gt.jsonl")
DEFAULT_MARKDOWN_CHUNKS_PATH = Path("data/processed/chunks_markdown.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/retrieval_markdown_gt.jsonl")
T = TypeVar("T")

MANUAL_CHUNK_OVERRIDES: dict[str, tuple[str, ...]] = {
    # PUT replacement behavior is explained before the PATCH recap.
    "q019": ("body-updates-md-chunk-0000", "body-updates-md-chunk-0001"),
    # The concrete min/max examples are stronger labels than the general recap.
    "q026": (
        "query-params-str-validations-md-chunk-0004",
        "query-params-str-validations-md-chunk-0012",
        "query-params-str-validations-md-chunk-0030",
    ),
    # Boolean conversion is a dedicated section in query-params.md.
    "q025": ("query-params-md-chunk-0004",),
    # Password/private data filtering lives in the response model filtering sections.
    "q064": (
        "response-model-md-chunk-0008",
        "response-model-md-chunk-0009",
        "response-model-md-chunk-0011",
        "response-model-md-chunk-0012",
        "response-model-md-chunk-0014",
    ),
}


@dataclass(frozen=True, slots=True)
class MarkdownChunkRecord:
    chunk_id: str
    document_id: str
    parent_id: str
    section_path: str
    symbols: str
    text: str


@dataclass(frozen=True, slots=True)
class ParentCandidate:
    parent_id: str
    document_id: str
    section_path: str
    symbols: str
    text: str
    chunk_ids: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build markdown parent/chunk retrieval ground truth labels."
    )
    parser.add_argument(
        "--source-gt",
        type=Path,
        default=DEFAULT_SOURCE_GT_PATH,
        help="Strategy-agnostic source-level ground truth JSONL.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_MARKDOWN_CHUNKS_PATH,
        help="Markdown parent-child chunks JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output markdown ground truth JSONL.",
    )
    parser.add_argument(
        "--max-parents-per-document",
        type=int,
        default=2,
        help="Maximum relevant parent labels to keep per source document.",
    )
    parser.add_argument(
        "--max-chunks-per-parent",
        type=int,
        default=2,
        help="Maximum relevant child chunk labels to keep per selected parent.",
    )
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=0.65,
        help="Keep candidates whose score is at least this fraction of the best score.",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


def run(args: argparse.Namespace) -> int:
    if args.max_parents_per_document <= 0:
        raise ValueError("max-parents-per-document must be positive.")
    if args.max_chunks_per_parent <= 0:
        raise ValueError("max-chunks-per-parent must be positive.")
    if not 0 < args.relative_threshold <= 1:
        raise ValueError("relative-threshold must be in the range (0, 1].")

    source_rows = load_jsonl(args.source_gt)
    chunks = load_markdown_chunks(args.chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    parents_by_document = group_parent_candidates(chunks)
    chunks_by_parent = group_chunks_by_parent(chunks)

    output_rows: list[dict[str, Any]] = []
    for row in source_rows:
        relevant_document_ids = require_str_list(row, "relevant_document_ids")
        query_text = build_query_text(row)
        selected_parents: list[ParentCandidate] = []
        selected_chunks: list[MarkdownChunkRecord] = []

        for document_id in relevant_document_ids:
            parent_candidates = parents_by_document.get(document_id, [])
            ranked_parents = rank_candidates(
                query_text,
                parent_candidates,
                max_items=args.max_parents_per_document,
                relative_threshold=args.relative_threshold,
            )
            selected_parents.extend(ranked_parents)

            for parent in ranked_parents:
                child_candidates = chunks_by_parent[parent.parent_id]
                selected_chunks.extend(
                    rank_candidates(
                        query_text,
                        child_candidates,
                        max_items=args.max_chunks_per_parent,
                        relative_threshold=args.relative_threshold,
                    )
                )

        manual_chunk_ids = MANUAL_CHUNK_OVERRIDES.get(require_str(row, "id"))
        if manual_chunk_ids is not None:
            selected_chunks = [
                require_chunk(chunks_by_id, chunk_id)
                for chunk_id in manual_chunk_ids
            ]
            selected_parent_ids = unique(chunk.parent_id for chunk in selected_chunks)
        else:
            selected_parent_ids = unique(
                parent.parent_id for parent in selected_parents
            )

        output_rows.append(
            {
                "id": require_str(row, "id"),
                "question": require_str(row, "question"),
                "relevant_document_ids": relevant_document_ids,
                "relevant_parent_ids": selected_parent_ids,
                "relevant_chunk_ids": unique(chunk.chunk_id for chunk in selected_chunks),
                "reference_answer": row.get("reference_answer"),
            }
        )

    save_jsonl(output_rows, args.output)
    print(f"Loaded source rows: {len(source_rows)}")
    print(f"Loaded markdown chunks: {len(chunks)}")
    print(f"Wrote markdown GT: {args.output}")
    print_summary(output_rows)
    return 0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Row {line_number} in {path} must be a JSON object.")
            rows.append(row)
    return rows


def load_markdown_chunks(path: str | Path) -> list[MarkdownChunkRecord]:
    chunks: list[MarkdownChunkRecord] = []
    for line_number, row in enumerate(load_jsonl(path), start=1):
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Chunk row {line_number} metadata must be a mapping.")
        chunks.append(
            MarkdownChunkRecord(
                chunk_id=require_str(row, "chunk_id"),
                document_id=require_str(row, "document_id"),
                parent_id=require_metadata_str(metadata, "parent_id", line_number),
                section_path=require_metadata_str(
                    metadata,
                    "section_path",
                    line_number,
                ),
                symbols=get_metadata_str(metadata, "symbols"),
                text=require_str(row, "text"),
            )
        )
    return chunks


def group_parent_candidates(
    chunks: list[MarkdownChunkRecord],
) -> dict[str, list[ParentCandidate]]:
    by_parent: dict[str, list[MarkdownChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        by_parent[chunk.parent_id].append(chunk)

    parents_by_document: dict[str, list[ParentCandidate]] = defaultdict(list)
    for parent_id, parent_chunks in by_parent.items():
        first = parent_chunks[0]
        candidate = ParentCandidate(
            parent_id=parent_id,
            document_id=first.document_id,
            section_path=first.section_path,
            symbols=first.symbols,
            text="\n\n".join(chunk.text for chunk in parent_chunks),
            chunk_ids=tuple(chunk.chunk_id for chunk in parent_chunks),
        )
        parents_by_document[first.document_id].append(candidate)
    return dict(parents_by_document)


def group_chunks_by_parent(
    chunks: list[MarkdownChunkRecord],
) -> dict[str, list[MarkdownChunkRecord]]:
    by_parent: dict[str, list[MarkdownChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        by_parent[chunk.parent_id].append(chunk)
    return dict(by_parent)


def rank_candidates(
    query_text: str,
    candidates: list[T],
    *,
    max_items: int,
    relative_threshold: float,
) -> list[T]:
    if not candidates:
        return []

    candidate_texts = [candidate_to_text(candidate) for candidate in candidates]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        token_pattern=r"(?u)\b[\w./()=-]{2,}\b",
    )
    vectorizer.fit([query_text, *candidate_texts])
    query_vector = vectorizer.transform([query_text])
    candidate_matrix = vectorizer.transform(candidate_texts)
    dot_product = safe_sparse_dot(
        query_vector,
        cast(Any, candidate_matrix).T,
        dense_output=True,
    )
    scores = cast(Any, dot_product).ravel()
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    best_score = ranked[0][1]
    if best_score <= 0:
        return [ranked[0][0]]

    threshold = best_score * relative_threshold
    selected = [
        candidate
        for candidate, score in ranked
        if score >= threshold
    ]
    return selected[:max_items]


def candidate_to_text(candidate: object) -> str:
    if isinstance(candidate, ParentCandidate):
        return "\n".join(
            [
                candidate.section_path,
                candidate.symbols,
                candidate.text,
            ]
        )
    if isinstance(candidate, MarkdownChunkRecord):
        return "\n".join(
            [
                candidate.section_path,
                candidate.symbols,
                candidate.text,
            ]
        )
    raise TypeError(f"Unsupported candidate type: {type(candidate)!r}")


def build_query_text(row: Mapping[str, Any]) -> str:
    return "\n".join(
        value
        for value in [
            require_str(row, "question"),
            get_str(row, "reference_answer"),
        ]
        if value
    )


def save_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    parent_counts = [len(row.get("relevant_parent_ids", [])) for row in rows]
    chunk_counts = [len(row.get("relevant_chunk_ids", [])) for row in rows]
    print(f"Rows with parent labels: {sum(count > 0 for count in parent_counts)}")
    print(f"Rows with chunk labels: {sum(count > 0 for count in chunk_counts)}")
    print(f"Avg parents per row: {sum(parent_counts) / len(parent_counts):.2f}")
    print(f"Avg chunks per row: {sum(chunk_counts) / len(chunk_counts):.2f}")


def unique(items: Iterable[str]) -> list[str]:
    output: list[str] = []
    for item in items:
        if item not in output:
            output.append(item)
    return output


def require_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Field {key!r} must be a non-empty string.")
    return value


def get_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def require_str_list(row: Mapping[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Field {key!r} must be a list of strings.")
    if not value:
        raise ValueError(f"Field {key!r} must not be empty.")
    return list(value)


def require_metadata_str(
    metadata: Mapping[str, Any],
    key: str,
    line_number: int,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Chunk row {line_number} metadata {key!r} is required.")
    return value


def get_metadata_str(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) else ""


def require_chunk(
    chunks_by_id: Mapping[str, MarkdownChunkRecord],
    chunk_id: str,
) -> MarkdownChunkRecord:
    try:
        return chunks_by_id[chunk_id]
    except KeyError as exc:
        raise ValueError(f"Manual markdown GT override references {chunk_id!r}.") from exc


if __name__ == "__main__":
    raise SystemExit(main())

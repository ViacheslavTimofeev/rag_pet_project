# Evaluation datasets

## `retrieval_sources_gt.jsonl`

Strategy-agnostic source-level ground-truth dataset for FastAPI tutorial retrieval
evaluation.

Each row contains:

- `id`: stable question identifier.
- `question`: user query.
- `relevant_document_ids`: source document ids that should be retrieved for the query.
- `reference_answer`: concise reference answer for answer-quality and Ragas-style evaluation.

Provenance:

- Source corpus: local FastAPI tutorial markdown files in `data/raw/`.
- Initial questions, source labels, and short reference answers were derived from
  the original chunk-level labels.
- The dataset intentionally avoids chunk ids so it can be reused across chunking
  strategies.

Recommended deterministic retrieval metrics:

- `Recall@5`
- `Recall@10`
- `Precision@5`
- `MRR@10`
- `HitRate@5`

## `retrieval_markdown_gt.jsonl`

Markdown parent-child ground-truth dataset derived from
`retrieval_sources_gt.jsonl` and `data/processed/chunks_markdown.jsonl`.

Each row contains:

- `id`: stable question identifier.
- `question`: user query.
- `relevant_document_ids`: source document ids from the source-level dataset.
- `relevant_parent_ids`: markdown parent sections selected inside the relevant
  source documents.
- `relevant_chunk_ids`: markdown child chunks selected inside the relevant
  parent sections.
- `reference_answer`: concise reference answer copied from the source-level
  dataset.

Generation:

- Run `conda run -n rag python -m scripts.build_markdown_gt`.
- The script uses deterministic TF-IDF cosine matching over `question` plus
  `reference_answer`, constrained to the trusted `relevant_document_ids`.
- A small set of manually reviewed overrides is kept in the generator for
  cases where lexical matching picked adjacent but less exact sections.
- Parent and chunk labels are therefore suitable for markdown-specific retrieval
  experiments, but still worth spot-reviewing before treating them as final
  human labels.

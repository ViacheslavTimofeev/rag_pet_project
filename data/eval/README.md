# Evaluation datasets

## `retrieval_gt.jsonl`

Ground-truth dataset for FastAPI tutorial retrieval evaluation.

Each row contains:

- `id`: stable question identifier.
- `question`: user query.
- `relevant_chunk_ids`: chunk ids from `data/processed/chunks.jsonl` that should be retrieved for the query.
- `relevant_document_ids`: document ids derived from the relevant chunks.
- `reference_answer`: concise reference answer for answer-quality and Ragas-style evaluation.

Provenance:

- Source corpus: local FastAPI tutorial markdown files in `data/raw/`.
- Chunk source: `data/processed/chunks.jsonl`.
- Initial questions, relevance labels, and short reference answers were synthetically generated with ChatGPT 5.5.
- The dataset was manually spot-checked against `chunks.jsonl`; several chunk labels were corrected before freezing this version.

Recommended deterministic retrieval metrics:

- `Recall@5`
- `Recall@10`
- `Precision@5`
- `MRR@10`
- `HitRate@5`

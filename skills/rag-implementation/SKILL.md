---
name: rag-implementation
description: Build or refine the RAG pipeline for this project. Use when Codex needs to implement ingestion, chunking, embedding generation, indexing, retrieval, reranking, context packing, grounding, or end-to-end question answering behavior.
---

# RAG Implementation

Implement the smallest reliable retrieval pipeline first, then improve recall and answer quality with measurable changes.

## Workflow

1. Inspect current configs, scripts, and `src/` modules involved in the pipeline.
2. Confirm the document flow:
   - load raw files;
   - normalize text and metadata;
   - chunk consistently;
   - embed chunks;
   - write to vector storage;
   - retrieve candidates;
   - optionally rerank;
   - assemble context for generation.
3. Keep metadata rich enough for traceability: source, chunk id, document id, section, offsets if available.
4. Prefer deterministic chunking and indexing behavior over hidden heuristics.
5. When changing retrieval quality, add or update an eval path.

## Implementation Rules

- Separate chunking policy from embedding backend.
- Separate retrieval from answer generation.
- Keep prompt assembly in `src/llm/`, not inside retrieval adapters.
- Preserve enough provenance for citations or debugging.
- Expose top-k, chunk size, overlap, and reranker toggles through config.

## Good Defaults

- Start with a plain baseline before adding hybrid search or reranking.
- Keep one reference dataset or smoke example for quick manual checks.
- Log retrieved chunk identifiers during debugging.

## Avoid

- Embedding raw unnormalized data when preprocessing is cheap.
- Coupling one vector store API directly to business logic.
- Tuning prompts before checking retrieval quality.

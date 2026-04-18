# RAG Local Assistant

## Purpose

This repository is a local-first RAG assistant project. Treat it as a production-oriented Python codebase for:
- document ingestion and preprocessing;
- embeddings and vector indexing;
- retrieval and reranking;
- LLM backend integration;
- evaluation and benchmark runs;
- local API and UI delivery.

## Working Style

- Prefer simple, inspectable pipelines over opaque frameworks.
- Keep architecture modular: ingestion, indexing, retrieval, generation, eval, API, and UI should stay loosely coupled.
- Read existing config and code before changing behavior.
- Use deterministic scripts for mechanical steps. Do not encode repeatable workflows only in prose.
- Verify claims with artifacts: command output, tests, generated files, or logs.

## Project Priorities

1. Correctness of retrieval and answer grounding.
2. Reproducibility of experiments and evaluations.
3. Replaceability of model, embedding, and vectordb backends.
4. Clear configuration in `configs/*.yaml`.
5. Local developer ergonomics and debuggability.

## Expected Architecture

- `configs/`: runtime configuration for models, retrieval, eval, and API.
- `skills/`: project-specific skills for architecture, RAG implementation, backend selection, evaluation, and delivery.
- `data/raw/`: source documents.
- `data/processed/`: cleaned or chunked artifacts.
- `data/eval/`: benchmark datasets, golden answers, and eval outputs.
- `src/ingest/`: loaders, normalizers, chunking.
- `src/embeddings/`: embedding model adapters and batching logic.
- `src/vectordb/`: vector store adapters and index lifecycle.
- `src/retrieval/`: search, reranking, context assembly.
- `src/llm/`: generation backends and prompt assembly.
- `src/eval/`: retrieval and answer quality evaluation.
- `src/api/`: service endpoints.
- `src/ui/`: local user interface.
- `scripts/`: repeatable entrypoints for ingestion, evaluation, and serving.

## Engineering Rules

- Prefer adapters and interfaces when integrating model or storage providers.
- Keep provider-specific logic out of high-level orchestration when possible.
- Put defaults and tunables in config files, not hardcoded constants.
- When adding dependencies, prefer mature packages and avoid very new releases unless required.
- When behavior changes, update the relevant config or docs in the same task.
- Keep skills concise and trigger-oriented. Detailed mechanics belong in code or references.

## Python Environment

- Always run Python-related commands in the Conda environment `rag`.
- Preferred command pattern: `conda run -n rag python -m ...`
- Do not rely on Windows App Execution Alias entries such as `python.exe` or `python3.exe`.

## Verification

When implementing or changing behavior, verify with the smallest relevant proof:
- unit or integration tests for logic changes;
- script output for pipeline entrypoints;
- sample retrieval or eval runs for ranking and generation changes;
- API smoke checks for service changes.

Do not mark work complete if the repository contains no proof artifact for the change.

## OpenAI Docs

Always use the OpenAI developer documentation MCP server if you need to work with the OpenAI API, ChatGPT Apps SDK, Codex, or related docs without me having to explicitly ask.

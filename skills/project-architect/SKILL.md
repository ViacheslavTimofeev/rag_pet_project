---
name: project-architect
description: Design and evolve the architecture of this local RAG assistant. Use when Codex needs to define module boundaries, choose extension points, organize configs, plan milestones, or refactor structure across ingestion, retrieval, generation, eval, API, and UI.
---

# Project Architect

Design for modularity first. This repository should support swapping models, vector stores, and retrieval strategies without rewriting the full stack.

## Core Workflow

1. Read `AGENTS.md` and inspect the relevant folders before proposing structure.
2. Separate concerns between pipeline stages:
   - ingestion;
   - embeddings;
   - vector database;
   - retrieval;
   - LLM generation;
   - evaluation;
   - API and UI.
3. Move provider-specific behavior behind small adapters or interfaces.
4. Put tunable behavior in `configs/*.yaml`.
5. Prefer one clear flow from raw data to answer generation over clever abstractions.

## Design Rules

- Optimize for local development and reproducibility, not premature distribution.
- Favor explicit data contracts between stages.
- Keep scripts as thin orchestration wrappers around reusable `src/` modules.
- Make room for experimentation, but keep the default path simple.
- When touching more than one subsystem, define what is stable and what is pluggable.

## Deliverables

Produce concrete artifacts, not generic advice:
- folder layout;
- interface sketches;
- config schema suggestions;
- migration plans;
- acceptance criteria for refactors.

## Avoid

- Mixing retrieval logic with transport layers.
- Hardwiring one model vendor into the whole project.
- Burying experiment-critical parameters inside code.
- Overengineering before there is a running baseline.

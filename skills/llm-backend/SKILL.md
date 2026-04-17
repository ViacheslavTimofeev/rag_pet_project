---
name: llm-backend
description: Integrate or compare generation backends for the RAG assistant. Use when Codex needs to wire local or remote LLM providers, prompt assembly, structured outputs, streaming, retries, model configs, or backend abstraction layers.
---

# LLM Backend

Treat the language model as a replaceable backend behind a stable interface.

## Workflow

1. Read the relevant config in `configs/model.yaml` and inspect `src/llm/`.
2. Define a narrow backend contract:
   - input messages or prompt;
   - optional retrieved context;
   - generation parameters;
   - structured response shape;
   - streaming or non-streaming mode.
3. Implement provider-specific translation inside adapters.
4. Keep prompt templates and response parsing explicit and testable.
5. Add safe error handling, timeouts, and retry boundaries.

## Design Rules

- Separate prompt construction from transport calls.
- Keep system prompts concise and grounded in retrieved context.
- Prefer explicit structured outputs when downstream code depends on fields.
- Put model names, temperatures, token limits, and endpoint settings in config.
- Support local backends when practical, but do not assume them.

## Verification

Validate with artifacts such as:
- smoke generation tests;
- fixture-based parsing checks;
- one retrieval-plus-generation integration path.

## Avoid

- Letting API response formats leak across the codebase.
- Mixing secret management with prompt logic.
- Hiding backend defaults in multiple files.

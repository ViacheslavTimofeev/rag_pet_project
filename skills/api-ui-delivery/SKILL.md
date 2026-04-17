---
name: api-ui-delivery
description: Deliver the local API and UI for the RAG assistant. Use when Codex needs to implement serving endpoints, request and response schemas, streaming chat behavior, debugging surfaces, or a lightweight interface for querying and inspecting retrieved context.
---

# API UI Delivery

Expose the RAG system in a way that is easy to test, debug, and iterate on locally.

## Workflow

1. Inspect `configs/api.yaml`, `src/api/`, `src/ui/`, and `scripts/serve.py`.
2. Keep the API contract explicit:
   - query input;
   - optional session or filters;
   - retrieved documents payload;
   - answer payload;
   - debug metadata when enabled.
3. Make retrieval provenance visible in development mode.
4. Keep UI logic thin; core behavior should live in reusable backend modules.

## Delivery Rules

- Prefer simple endpoints and schemas before adding session complexity.
- Add a debug mode that surfaces retrieved chunks, scores, and model settings when safe.
- Keep streaming optional and well-bounded.
- Validate inputs early and return actionable errors.
- Make local startup predictable through config and scripts.

## Good Defaults

- One health endpoint.
- One ask/query endpoint.
- One debug path for retrieval inspection.
- Minimal UI for submitting a question and viewing sources.

## Avoid

- Embedding business logic inside UI code.
- Returning unstructured blobs when the client depends on fields.
- Hiding retrieval diagnostics during early development.

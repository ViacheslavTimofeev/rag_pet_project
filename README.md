# RAG Local Assistant

`rag_pet_project` is a local-first Retrieval-Augmented Generation assistant for
working with a small documentation corpus. The project is organized as a
production-oriented Python codebase: documents are ingested and chunked, indexed
into a vector database, retrieved and assembled into grounded context, passed to
a local LLM backend, exposed through an API/UI, and evaluated with saved
benchmark artifacts.

The current corpus is a local FastAPI tutorial snapshot in `data/raw/`. The
default embedding backend is `sentence-transformers/all-MiniLM-L6-v2`, the
vector store is Qdrant, and the default LLM backend is `llama-cpp-python` with a
local GGUF model.

## Features

- Modular RAG pipeline: ingestion, chunking, embeddings, vector indexing,
  retrieval, reranking, context assembly, generation, API, UI, and evaluation.
- YAML-based runtime configuration in `configs/`.
- Local-first model setup with SentenceTransformers, Qdrant, and llama.cpp.
- Replaceable adapters for embedding, vector database, reranking, and LLM
  backends.
- FastAPI service with `/health` and `/ask` endpoints.
- Gradio UI that talks to the local API.
- Deterministic retrieval evaluation with frozen ground truth and saved run
  artifacts.
- Unit tests for ingestion, indexing, retrieval, LLM, API, UI, and evaluation
  logic.

## Components

The main runtime components are:

- `ingest`: loads local markdown files and turns them into normalized documents
  and chunks.
- `embeddings`: wraps embedding models and batching logic.
- `vectordb`: manages Qdrant collection creation, upserts, and vector search.
- `retrieval`: retrieves ranked chunks, optionally reranks them, and builds
  generation-ready context.
- `llm`: builds grounded prompts and calls a local llama.cpp model.
- `api`: exposes the RAG workflow through FastAPI.
- `ui`: provides a Gradio client for local interactive use.
- `eval`: evaluates retrieval quality against a frozen benchmark dataset.

## Project Structure

```text
rag_pet_project/
├── configs/                 # YAML configs for model, retrieval, eval, API/UI
├── data/
│   ├── raw/                 # source markdown documents
│   ├── processed/           # generated raw_documents.jsonl and chunks.jsonl
│   └── eval/                # ground truth and saved eval runs
├── models/                  # local GGUF models, ignored by git
├── scripts/                 # repeatable CLI entrypoints
├── skills/                  # project-specific Codex skills
├── src/
│   ├── api/                 # FastAPI app, routes, schemas, dependencies
│   ├── embeddings/          # embedding adapters and factory
│   ├── eval/                # retrieval metrics and eval runner
│   ├── ingest/              # loaders, normalizers, chunking
│   ├── llm/                 # prompt building, LLM service, llama.cpp backend
│   ├── retrieval/           # retriever, reranker, context builder, pipeline
│   ├── runtime/             # device checks
│   ├── ui/                  # Gradio UI and API client
│   └── vectordb/            # Qdrant adapter and indexing lifecycle
├── tests/                   # unit tests
├── pyproject.toml
└── README.md
```

Important scripts:

- `scripts/ingest.py` - normalize raw documents and create chunks.
- `scripts/index.py` - embed chunks and upsert them into Qdrant.
- `scripts/run_eval.py` - run deterministic retrieval evaluation.
- `scripts/serve.py` - start the FastAPI service, optionally with the UI.
- `scripts/ui.py` - start only the Gradio UI.

Installed entrypoints from `pyproject.toml`:

- `rag-ingest`
- `rag-index`
- `rag-eval`
- `rag-serve`
- `rag-ui`

## Configuration

Main configs:

- `configs/model.yaml` - embedding backend and LLM backend settings.
- `configs/retrieval.yaml` - Qdrant connection, retrieval `top_k`, reranker,
  and context builder settings.
- `configs/eval.yaml` - eval dataset path, `k_values`, and Ragas placeholder
  settings.
- `configs/api.yaml` - API server, request, CORS, and UI settings.

Default runtime assumptions:

- Conda environment: `rag`.
- Qdrant URL: `http://localhost:6333`.
- Qdrant collection: `documents`.
- Local model path: `models/Qwen3-14B-Q4_K_M.gguf`.
- CUDA is expected by the current embedding/retrieval factory helpers.

## CLI Workflow

Run Python commands through the project Conda environment:

```powershell
conda run -n rag python -m ...
```

### 1. Prepare Documents

The repository already contains FastAPI tutorial markdown files in `data/raw/`.
To regenerate processed artifacts:

```powershell
conda run -n rag python -m scripts.ingest data/raw data/processed
```

This writes:

- `data/processed/raw_documents.jsonl`
- `data/processed/chunks.jsonl`

### 2. Start Qdrant

Start a local Qdrant instance before indexing or retrieval. One simple option is
the official Docker image:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

If you use a different Qdrant URL or collection name, update
`configs/retrieval.yaml`.

### 3. Build the Vector Index

Embed chunks and upsert them into Qdrant:

```powershell
conda run -n rag python -m scripts.index
```

The script reads:

- chunks from `data/processed/chunks.jsonl`;
- Qdrant settings from `configs/retrieval.yaml`;
- embedding settings from `configs/model.yaml`.

### 4. Run Retrieval Evaluation

Run the deterministic retrieval benchmark:

```powershell
conda run -n rag python -m scripts.run_eval retrieval
```

The runner reads `data/eval/retrieval_gt.jsonl`, retrieves top-k chunks for each
question, computes retrieval metrics, and saves a JSON artifact under
`data/eval/runs/`.

Current baseline artifact:

```text
data/eval/runs/baseline_run_04_05_2026.json
```

Baseline metrics on 68 FastAPI tutorial questions:

```text
@1:
  recall    0.2635
  precision 0.5000
  hit_rate  0.5000
  mrr       0.5000

@3:
  recall    0.4730
  precision 0.3137
  hit_rate  0.6765
  mrr       0.5784

@5:
  recall    0.5784
  precision 0.2294
  hit_rate  0.7941
  mrr       0.6042

@10:
  recall    0.7120
  precision 0.1456
  hit_rate  0.8824
  mrr       0.6170
```

### 5. Run the API

Start the FastAPI service:

```powershell
conda run -n rag python -m scripts.serve
```

By default the API is served at:

```text
http://127.0.0.1:8000
```

Main endpoints:

- `GET /health` - service health check.
- `POST /ask` - run retrieval, context assembly, and generation.

### 6. Run the UI

If the API is already running, start only the Gradio UI:

```powershell
conda run -n rag python -m scripts.ui
```

Default UI URL:

```text
http://127.0.0.1:7860
```

`scripts.serve` can also launch the API and UI together when
`api.ui.enabled: true` in `configs/api.yaml`.

## Evaluation

The repository currently includes retrieval-level evaluation:

- frozen ground truth: `data/eval/retrieval_gt.jsonl`;
- dataset notes: `data/eval/README.md`;
- metrics implementation: `src/eval/retrieval_metrics.py`;
- runner: `src/eval/retrieval_runner.py`;
- CLI: `scripts/run_eval.py`;
- saved baseline run: `data/eval/runs/baseline_run_04_05_2026.json`.

Supported deterministic retrieval metrics:

- `Recall@k`
- `Precision@k`
- `HitRate@k`
- `MRR@k`

Ragas is listed as an optional eval dependency and `configs/eval.yaml` already
contains a placeholder section for it. Full answer-level/Ragas evaluation is a
planned next step and should use saved records with:

- question;
- retrieved contexts;
- generated answer;
- reference answer.

## Tests

Run the test suite:

```powershell
conda run -n rag python -m unittest discover tests
```

The current suite covers ingestion, embeddings, vector DB adapters, indexing,
retrieval, reranking, context building, LLM prompt/service/pipeline behavior,
API, UI client/config, and eval metrics/runner/CLI.

## Installation Notes

The project is configured with `pyproject.toml`. For a pip-based environment:

```powershell
pip install -e .
```

For development and future Ragas work:

```powershell
pip install -e ".[dev,eval]"
```

The preferred local workflow for this repository is still the Conda environment
named `rag`, especially because local CUDA/PyTorch, SentenceTransformers,
Qdrant, and llama.cpp installations can be platform-specific.

## Current Limitations

- Retrieval evaluation is implemented; answer-level and Ragas evaluation are not
  implemented yet.
- The active reranker is `BAAI/bge-reranker-base`; it reranks the vector
  retriever candidate set before context assembly.
- Docker files are not included yet.
- Local model weights are expected under `models/` and are intentionally not
  committed.

## Recommended End-to-End Order

1. Prepare or refresh `data/raw/`.
2. Run ingestion with `scripts.ingest`.
3. Start Qdrant.
4. Run indexing with `scripts.index`.
5. Run retrieval evaluation with `scripts.run_eval retrieval`.
6. Start the API with `scripts.serve`.
7. Open the UI with `scripts.ui` or through the combined server flow.

# Terminal Commands

All Python commands should run in the Conda environment `rag`.

Prefer this pattern:

```powershell
conda run -n rag python -m ...
```

On Windows, avoid running several `conda run` commands in parallel. Conda can
race on temporary activation files.

## Docker Services

Start Qdrant:

```powershell
docker run --name qdrant-local `
  -p 6333:6333 `
  -p 6334:6334 `
  qdrant/qdrant
```

Start an existing stopped Qdrant container:

```powershell
docker start qdrant-local
```

Stop Qdrant:

```powershell
docker stop qdrant-local
```

Show running containers:

```powershell
docker ps
```

Check Docker GPU passthrough:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Ingestion

Default character chunking:

```powershell
conda run -n rag python -m scripts.ingest data/raw data/processed
```

Markdown-aware parent-child chunking:

```powershell
conda run -n rag python -m scripts.ingest data/raw data/processed `
  --chunk-strategy markdown-parent-child `
  --child-chunk-size 700 `
  --child-chunk-overlap 120
```

Show ingest options:

```powershell
conda run -n rag python -m scripts.ingest --help
```

## Ground Truth

Build markdown-specific retrieval ground truth from source-level labels:

```powershell
conda run -n rag python -m scripts.build_markdown_gt
```

Build markdown GT with explicit paths:

```powershell
conda run -n rag python -m scripts.build_markdown_gt `
  --source-gt data/eval/retrieval_sources_gt.jsonl `
  --chunks data/processed/chunks_markdown.jsonl `
  --output data/eval/retrieval_markdown_gt.jsonl
```

Show markdown GT options:

```powershell
conda run -n rag python -m scripts.build_markdown_gt --help
```

## Indexing

Index chunks using the paths and Qdrant collection in `configs/retrieval.yaml`:

```powershell
conda run -n rag python -m scripts.index
```

Index a specific chunks file:

```powershell
conda run -n rag python -m scripts.index `
  --chunks data/processed/chunks_markdown.jsonl `
  --retrieval-config configs/retrieval.yaml
```

Index with a custom batch size:

```powershell
conda run -n rag python -m scripts.index --batch-size 32
```

Show index options:

```powershell
conda run -n rag python -m scripts.index --help
```

## Retrieval Evaluation

Run deterministic retrieval eval:

```powershell
conda run -n rag python -m scripts.run_eval retrieval
```

Run eval with explicit configs and output directory:

```powershell
conda run -n rag python -m scripts.run_eval retrieval `
  --eval-config configs/eval.yaml `
  --retrieval-config configs/retrieval.yaml `
  --output-dir data/eval/runs
```

Show eval options:

```powershell
conda run -n rag python -m scripts.run_eval --help
conda run -n rag python -m scripts.run_eval retrieval --help
```

Document-level eval uses `data/eval/retrieval_sources_gt.jsonl`.

Chunk-level markdown eval uses `data/eval/retrieval_markdown_gt.jsonl`.

Switch the active dataset in `configs/eval.yaml` before running eval.

## API And UI

Start the API server, and the UI too if enabled in `configs/api.yaml`:

```powershell
conda run -n rag python -m scripts.serve
```

Start only the Gradio UI:

```powershell
conda run -n rag python -m scripts.ui
```

Check API health:

```powershell
curl.exe http://localhost:8000/health
```

## Tests

Run the full test suite:

```powershell
conda run -n rag python -m pytest
```

Run selected tests:

```powershell
conda run -n rag python -m pytest tests/test_eval_retrieval_runner.py
```

Run unittest discovery:

```powershell
conda run -n rag python -m unittest discover tests
```

Run mypy on tests only, skipping imported production modules:

```powershell
conda run -n rag mypy --ignore-missing-imports --follow-imports=skip tests
```

Compile-check selected files:

```powershell
conda run -n rag python -m compileall scripts/build_markdown_gt.py
```

## vLLM Local OpenAI-Compatible Server

Start a small vLLM smoke-test model:

```powershell
docker run --rm --gpus all `
  -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface `
  -p 8000:8000 `
  --ipc=host `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --dtype float16 `
  --max-model-len 2048 `
  --gpu-memory-utilization 0.70 `
  --tensor-parallel-size 1 `
  --api-key local-vllm
```

Start the same vLLM server as a named container:

```powershell
docker run --name vllm-qwen-test --gpus all `
  -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface `
  -p 8000:8000 `
  --ipc=host `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --dtype float16 `
  --max-model-len 2048 `
  --gpu-memory-utilization 0.70 `
  --tensor-parallel-size 1 `
  --api-key local-vllm
```

Start, stop, inspect, and remove the named vLLM container:

```powershell
docker start vllm-qwen-test
docker stop vllm-qwen-test
docker logs vllm-qwen-test --tail 100
docker rm vllm-qwen-test
```

Check that the vLLM OpenAI-compatible models endpoint is alive:

```powershell
curl.exe http://localhost:8000/v1/models `
  -H "Authorization: Bearer local-vllm"
```

PowerShell-native equivalent:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/v1/models" `
  -Headers @{ Authorization = "Bearer local-vllm" }
```

Test chat completion:

```powershell
curl.exe http://localhost:8000/v1/chat/completions `
  -H "Authorization: Bearer local-vllm" `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"Qwen/Qwen2.5-1.5B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"Answer in one sentence: what is FastAPI?\"}],\"temperature\":0,\"max_tokens\":80}"
```

Optional Hugging Face token for higher rate limits:

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN=$env:HF_TOKEN `
  -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface `
  -p 8000:8000 `
  --ipc=host `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --dtype float16 `
  --max-model-len 2048 `
  --gpu-memory-utilization 0.70 `
  --tensor-parallel-size 1 `
  --api-key local-vllm
```

## Useful Diagnostics

Check GPU:

```powershell
nvidia-smi
```

Check Docker:

```powershell
docker --version
docker ps
```

Check Python version in the project env:

```powershell
conda run -n rag python -c "import sys; print(sys.version)"
```

Check current Git status:

```powershell
git status --short
```

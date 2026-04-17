---
name: eval-and-benchmark
description: Evaluate retrieval and answer quality for the local RAG assistant. Use when Codex needs to create benchmark datasets, define metrics, run regressions, compare configurations, or explain whether a change improved the system.
---

# Eval And Benchmark

Use evaluation to decide whether a RAG change helped. Prefer mechanical evidence over intuition.

## Workflow

1. Inspect `configs/eval.yaml`, `data/eval/`, `src/eval/`, and `scripts/run_eval.py`.
2. Define what is being measured:
   - retrieval quality;
   - answer quality;
   - latency;
   - cost or local resource usage.
3. Keep datasets versioned and interpretable.
4. Compare baseline and candidate settings with the same dataset and metric definitions.
5. Report deltas, not just absolute numbers.

## Recommended Metrics

- Retrieval: recall@k, precision@k, MRR, hit rate.
- Answering: exact match, F1, rubric score, groundedness, citation correctness.
- System: latency, throughput, memory footprint, token usage.

## Rules

- Freeze the eval set before comparing configurations.
- Track the config used for each run.
- Distinguish retrieval failures from generation failures.
- Prefer a small fast smoke eval plus a slower fuller benchmark.

## Avoid

- Judging improvements only from one anecdotal query.
- Mixing changed prompts, retrieval, and models without recording the baseline.
- Reporting success without saved outputs or metric summaries.

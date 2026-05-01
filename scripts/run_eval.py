from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from src.eval import (
    load_retrieval_eval_dataset,
    run_retrieval_eval,
    save_retrieval_eval_run,
)
from src.retrieval import build_retriever


DEFAULT_EVAL_CONFIG_PATH = Path("configs/eval.yaml")
DEFAULT_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")
DEFAULT_RUNS_DIR = Path("data/eval/runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run project evaluation tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieval_parser = subparsers.add_parser(
        "retrieval",
        help="Run deterministic retrieval evaluation.",
    )
    retrieval_parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help="Path to configs/eval.yaml.",
    )
    retrieval_parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
        help="Path to configs/retrieval.yaml.",
    )
    retrieval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory where the JSON run artifact is saved.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "retrieval":
        return run_retrieval_command(args)
    raise ValueError(f"Unsupported eval command: {args.command}")


def run_retrieval_command(args: argparse.Namespace) -> int:
    config = load_eval_config(args.eval_config)
    dataset_path = get_dataset_path(config)
    k_values = get_k_values(config)

    examples = load_retrieval_eval_dataset(dataset_path)
    retriever = build_retriever(args.retrieval_config)
    result = run_retrieval_eval(
        retriever,
        examples,
        k_values=k_values,
        dataset_path=dataset_path,
        metadata={
            "eval_config_path": str(args.eval_config),
            "retrieval_config_path": str(args.retrieval_config),
        },
    )

    output_path = args.output_dir / f"{timestamp()}_{result.run_id}.json"
    save_retrieval_eval_run(result, output_path)
    print_retrieval_summary(result, output_path)
    return 0


def load_eval_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs/eval.yaml.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Eval config must be a mapping at the top level.")
    return loaded


def get_dataset_path(config: Mapping[str, Any]) -> Path:
    eval_config = get_mapping(config, "eval")
    dataset_config = get_mapping(eval_config, "dataset")
    value = dataset_config.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("'eval.dataset.path' must be a non-empty string.")
    return Path(value)


def get_k_values(config: Mapping[str, Any]) -> list[int]:
    eval_config = get_mapping(config, "eval")
    retrieval_config = get_mapping(eval_config, "retrieval")
    value = retrieval_config.get("k_values")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, int) and item > 0 for item in value)
    ):
        raise ValueError("'eval.retrieval.k_values' must be a list of positive integers.")
    return value


def print_retrieval_summary(result: Any, output_path: Path) -> None:
    metrics = result.metrics
    print(f"Saved retrieval eval run: {output_path}")
    print(f"Queries: {metrics.query_count}")
    for k in metrics.k_values:
        print(
            f"@{k}: "
            f"recall={metrics.recall_at_k[k]:.4f} "
            f"precision={metrics.precision_at_k[k]:.4f} "
            f"hit_rate={metrics.hit_rate_at_k[k]:.4f} "
            f"mrr={metrics.mean_reciprocal_rank_at_k[k]:.4f}"
        )


def get_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping.")
    return value


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, cast


DEFAULT_EVAL_CONFIG_PATH = Path("configs/eval.yaml")
DEFAULT_MODEL_CONFIG_PATH = Path("configs/model.yaml")
DEFAULT_OUTPUT_PATH = Path("data/eval/ragas_synthetic.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a synthetic Ragas testset from local markdown documents."
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help="Path to configs/eval.yaml.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help="Path to configs/model.yaml.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional raw markdown source directory. Prefer --chunks for normal runs.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="JSONL chunks file. Defaults to eval.dataset.chunk_source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to eval.ragas.output_path.",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=None,
        help="Number of synthetic samples to generate. Defaults to eval.ragas.testset_size.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on loaded chunks or markdown files.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=None,
        help="Maximum characters per loaded chunk or document. Defaults to eval.ragas.max_doc_chars.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Ragas worker count. Defaults to eval.ragas.max_workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-call timeout in seconds. Defaults to eval.ragas.timeout_seconds.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum output tokens for Ragas LLM calls. Defaults to eval.ragas.max_tokens.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Ask Ragas to keep partial results instead of raising on generation errors.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_command(args)


def run_command(args: argparse.Namespace) -> int:
    eval_config = load_yaml_config(args.eval_config, label="Eval")
    model_config = load_yaml_config(args.model_config, label="Model")
    ragas_config = get_mapping(get_mapping(eval_config, "eval"), "ragas")

    chunks_path = args.chunks or get_chunks_path(eval_config)
    source_dir = args.source_dir
    output_path = args.output or get_optional_path(
        ragas_config,
        "output_path",
        default=DEFAULT_OUTPUT_PATH,
    )
    testset_size = args.testset_size or get_positive_int(
        ragas_config,
        "testset_size",
        default=5,
    )
    max_workers = args.max_workers or get_positive_int(
        ragas_config,
        "max_workers",
        default=1,
    )
    timeout = args.timeout or get_positive_int(
        ragas_config,
        "timeout_seconds",
        default=180,
    )
    max_tokens = args.max_tokens or get_positive_int(
        ragas_config,
        "max_tokens",
        default=2048,
    )
    max_doc_chars = args.max_doc_chars or get_positive_int(
        ragas_config,
        "max_doc_chars",
        default=12000,
    )

    if source_dir is not None:
        documents = load_markdown_documents(
            source_dir,
            max_docs=args.max_docs,
            max_doc_chars=max_doc_chars,
        )
    else:
        documents = load_chunk_documents(
            chunks_path,
            max_docs=args.max_docs,
            max_doc_chars=max_doc_chars,
        )
    generator = build_testset_generator(model_config, max_tokens=max_tokens)
    run_config = build_run_config(timeout=timeout, max_workers=max_workers)
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=testset_size,
        run_config=run_config,
        raise_exceptions=not args.keep_going,
    )

    save_testset_jsonl(testset, output_path)
    print(f"Saved Ragas synthetic dataset: {output_path}")
    print(f"Documents loaded: {len(documents)}")
    print(f"Requested samples: {testset_size}")
    return 0


def save_testset_jsonl(testset: Any, output_path: Path) -> None:
    to_jsonl = getattr(testset, "to_jsonl", None)
    if not callable(to_jsonl):
        raise TypeError(
            "Ragas returned an object without to_jsonl(). "
            "Check that return_executor=False is used."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_jsonl(output_path)


def load_yaml_config(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(f"PyYAML is required to load {label.lower()} config.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"{label} config must be a mapping at the top level.")
    return loaded


def load_markdown_documents(
    source_dir: str | Path,
    *,
    max_docs: int | None = None,
    max_doc_chars: int | None = None,
):
    if max_docs is not None and max_docs <= 0:
        raise ValueError("max_docs must be a positive integer or null.")
    if max_doc_chars is not None and max_doc_chars <= 0:
        raise ValueError("max_doc_chars must be a positive integer or null.")

    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "langchain-core is required to build Ragas source documents."
        ) from exc

    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Source path must be a directory: {root}")

    paths = sorted(path for path in root.rglob("*.md") if path.is_file())
    if max_docs is not None:
        paths = paths[:max_docs]
    if not paths:
        raise ValueError(f"No markdown files found under {root}")

    documents = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8-sig").strip()
        if max_doc_chars is not None and len(text) > max_doc_chars:
            text = text[:max_doc_chars].rstrip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": relative_path,
                    "document_id": document_id_from_path(relative_path),
                },
            )
        )

    if not documents:
        raise ValueError(f"No non-empty markdown files found under {root}")
    return documents


def load_chunk_documents(
    chunks_path: str | Path,
    *,
    max_docs: int | None = None,
    max_doc_chars: int | None = None,
):
    if max_docs is not None and max_docs <= 0:
        raise ValueError("max_docs must be a positive integer or null.")
    if max_doc_chars is not None and max_doc_chars <= 0:
        raise ValueError("max_doc_chars must be a positive integer or null.")

    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "langchain-core is required to build Ragas source documents."
        ) from exc

    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Chunks path must be a file: {path}")

    documents = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = parse_json_line(line, line_number=line_number)
            text = require_str(row, "text")
            if max_doc_chars is not None and len(text) > max_doc_chars:
                text = text[:max_doc_chars].rstrip()
            if not text:
                continue
            metadata = get_chunk_metadata(row)
            documents.append(Document(page_content=text, metadata=metadata))
            if max_docs is not None and len(documents) >= max_docs:
                break

    if not documents:
        raise ValueError(f"No non-empty chunks found in {path}")
    return documents


def build_testset_generator(model_config: Mapping[str, Any], *, max_tokens: int | None = None):
    try:
        import instructor
        from openai import AsyncOpenAI
        from pydantic import SecretStr
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import BaseRagasLLM
        from ragas.llms import InstructorLLM
        from ragas.testset import TestsetGenerator
    except ImportError as exc:
        raise ImportError(
            "ragas, instructor, openai, pydantic, and langchain-openai are "
            "required to build a synthetic dataset."
        ) from exc

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Install langchain-huggingface or langchain-community to use "
                "local HuggingFace embeddings with Ragas."
            ) from exc

    vllm_config = get_vllm_config(model_config)
    embedding_config = get_sentence_transformer_config(model_config)

    api_key = SecretStr(get_optional_str(vllm_config, "api_key") or "EMPTY")
    client = instructor.from_openai(
        AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            base_url=require_str(vllm_config, "base_url"),
            timeout=get_positive_int(vllm_config, "timeout_seconds", default=120),
        ),
        mode=instructor.Mode.JSON,
    )
    llm = InstructorLLM(
        client=client,
        provider="openai",
        model=require_str(vllm_config, "model"),
        max_tokens=max_tokens
        or get_positive_int(vllm_config, "max_tokens", default=1024),
        temperature=get_float(vllm_config, "temperature", default=0.0),
        top_p=get_float(vllm_config, "top_p", default=1.0),
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=require_str(embedding_config, "model_name"),
        model_kwargs={
            "local_files_only": get_bool(
                embedding_config,
                "local_files_only",
                default=False,
            )
        },
        encode_kwargs={
            "normalize_embeddings": get_bool(
                embedding_config,
                "normalize_embeddings",
                default=True,
            )
        },
    )
    return TestsetGenerator(
        llm=cast(BaseRagasLLM, llm),
        embedding_model=LangchainEmbeddingsWrapper(embeddings),
    )


def build_langchain_chat_openai(model_config: Mapping[str, Any], *, max_tokens: int):
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr
    except ImportError as exc:
        raise ImportError(
            "pydantic and langchain-openai are required to build ChatOpenAI."
        ) from exc

    vllm_config = get_vllm_config(model_config)
    return ChatOpenAI(
        model=require_str(vllm_config, "model"),
        base_url=require_str(vllm_config, "base_url"),
        api_key=SecretStr(get_optional_str(vllm_config, "api_key") or "EMPTY"),
        temperature=get_float(vllm_config, "temperature", default=0.0),
        max_completion_tokens=max_tokens,
        top_p=get_float(vllm_config, "top_p", default=1.0),
        timeout=get_positive_int(vllm_config, "timeout_seconds", default=120),
    )


def build_run_config(*, timeout: int, max_workers: int):
    try:
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise ImportError("ragas is required to build a RunConfig.") from exc

    return RunConfig(timeout=timeout, max_workers=max_workers)


def get_source_dir(config: Mapping[str, Any]) -> Path:
    dataset_config = get_mapping(get_mapping(config, "eval"), "dataset")
    value = dataset_config.get("source_corpus", "data/raw")
    if not isinstance(value, str) or not value:
        raise ValueError("'eval.dataset.source_corpus' must be a non-empty string.")
    return Path(value)


def get_chunks_path(config: Mapping[str, Any]) -> Path:
    dataset_config = get_mapping(get_mapping(config, "eval"), "dataset")
    value = dataset_config.get("chunk_source", "data/processed/chunks_markdown.jsonl")
    if not isinstance(value, str) or not value:
        raise ValueError("'eval.dataset.chunk_source' must be a non-empty string.")
    return Path(value)


def get_vllm_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    llm_config = get_mapping(config, "llm")
    active_backend = llm_config.get("active_backend")
    if active_backend != "vllm":
        raise ValueError(
            "Ragas synthetic generation currently requires llm.active_backend='vllm'."
        )
    return get_mapping(llm_config, "vllm")


def get_sentence_transformer_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    embedding_config = get_mapping(config, "embedding")
    active_backend = embedding_config.get("active_backend")
    if active_backend != "sentence_transformer":
        raise ValueError(
            "Ragas synthetic generation currently requires "
            "embedding.active_backend='sentence_transformer'."
        )
    return get_mapping(embedding_config, "sentence_transformer")


def get_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping.")
    return value


def parse_json_line(line: str, *, line_number: int) -> Mapping[str, Any]:
    import json

    row = json.loads(line)
    if not isinstance(row, Mapping):
        raise ValueError(f"JSONL row {line_number} must be an object.")
    return row


def get_chunk_metadata(row: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key in ("chunk_id", "document_id", "chunk_index"):
        value = row.get(key)
        if isinstance(value, str | int | float | bool):
            metadata[key] = value

    raw_metadata = row.get("metadata")
    if isinstance(raw_metadata, Mapping):
        for key, value in raw_metadata.items():
            if isinstance(key, str) and isinstance(value, str | int | float | bool):
                metadata.setdefault(key, value)
    return metadata


def require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value


def get_optional_str(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string or null.")
    return value


def get_optional_path(
    config: Mapping[str, Any],
    key: str,
    *,
    default: Path,
) -> Path:
    value = config.get(key, default)
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty path string.")
    return Path(value)


def get_positive_int(config: Mapping[str, Any], key: str, *, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{key}' must be a positive integer.")
    return value


def get_float(config: Mapping[str, Any], key: str, *, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"'{key}' must be a number.")
    return float(value)


def get_bool(config: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean.")
    return value


def document_id_from_path(relative_path: str) -> str:
    path = Path(relative_path)
    parts = [*path.with_suffix("").parts]
    return "-".join(part.replace("_", "-") for part in parts) + "-md"


if __name__ == "__main__":
    raise SystemExit(main())

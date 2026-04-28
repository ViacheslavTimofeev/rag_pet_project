from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from src.embeddings.factory import build_embedder, build_embedder_from_config, load_model_config


class EmbeddingsFactoryTests(unittest.TestCase):
    def test_load_model_config_reads_yaml_file(self) -> None:
        tmp_root = Path(".tmp_test") / f"config-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            config_path = tmp_root / "model.yaml"
            config_path.write_text(
                (
                    "embedding:\n"
                    "  active_backend: sentence_transformer\n"
                    "  sentence_transformer:\n"
                    "    model_name: sentence-transformers/all-MiniLM-L6-v2\n"
                ),
                encoding="utf-8",
            )

            config = load_model_config(config_path)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(config["embedding"]["active_backend"], "sentence_transformer")

    def test_build_embedder_from_config_uses_sentence_transformer_backend(self) -> None:
        config = {
            "embedding": {
                "active_backend": "sentence_transformer",
                "sentence_transformer": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 16,
                    "normalize_embeddings": False,
                    "local_files_only": True,
                },
            }
        }

        with patch("src.embeddings.factory.require_cuda_device", return_value="cuda"):
            with patch("src.embeddings.factory.SentenceTransformerEmbedder") as embedder_cls:
                instance = embedder_cls.return_value
                embedder = build_embedder_from_config(config)

        self.assertIs(embedder, instance)
        embedder_cls.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize_embeddings=False,
            batch_size=16,
            device="cuda",
            local_files_only=True,
        )

    def test_build_embedder_from_config_requires_cuda(self) -> None:
        config = {
            "embedding": {
                "active_backend": "sentence_transformer",
                "sentence_transformer": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                },
            }
        }

        with patch("src.embeddings.factory.require_cuda_device", return_value="cuda"):
            with patch("src.embeddings.factory.SentenceTransformerEmbedder") as embedder_cls:
                build_embedder_from_config(config)

        self.assertEqual(embedder_cls.call_args.kwargs["device"], "cuda")

    def test_build_embedder_loads_default_config_path(self) -> None:
        with patch("src.embeddings.factory.load_model_config") as load_config:
            with patch("src.embeddings.factory.build_embedder_from_config") as build_from_config:
                config = {"embedding": {"active_backend": "sentence_transformer"}}
                load_config.return_value = config
                build_from_config.return_value = object()

                result = build_embedder()

        load_config.assert_called_once()
        build_from_config.assert_called_once_with(config)
        self.assertIs(result, build_from_config.return_value)


if __name__ == "__main__":
    unittest.main()

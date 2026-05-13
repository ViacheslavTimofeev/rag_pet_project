from __future__ import annotations

import unittest
from unittest.mock import patch

from src.llm.factory import build_llm_backend, build_llm_backend_from_config


class LLMFactoryTests(unittest.TestCase):
    def test_build_llm_backend_from_config_builds_llamacpp_backend(self) -> None:
        config = {
            "llm": {
                "active_backend": "llamacpp",
                "llamacpp": {
                    "model_path": "models/qwen.gguf",
                    "n_ctx": 8192,
                    "n_gpu_layers": -1,
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "top_p": 0.9,
                    "stop": ["</s>"],
                    "seed": 42,
                    "chat_format": None,
                    "verbose": False,
                },
            }
        }

        with patch("src.llm.factory.LlamaCppBackend") as backend_cls:
            backend_cls.return_value = object()
            result = build_llm_backend_from_config(config)

        backend_cls.assert_called_once_with(
            model_path="models/qwen.gguf",
            n_ctx=8192,
            n_gpu_layers=-1,
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            stop=["</s>"],
            seed=42,
            chat_format=None,
            verbose=False,
        )
        self.assertIs(result, backend_cls.return_value)

    def test_build_llm_backend_from_config_builds_vllm_backend(self) -> None:
        config = {
            "llm": {
                "active_backend": "vllm",
                "vllm": {
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "Qwen3-14B-Q4_K_M",
                    "api_key": "local-vllm",
                    "api_key_env": None,
                    "timeout_seconds": 90,
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "top_p": 0.9,
                    "stop": ["</s>"],
                },
            }
        }

        with patch("src.llm.factory.VllmBackend") as backend_cls:
            backend_cls.return_value = object()
            result = build_llm_backend_from_config(config)

        backend_cls.assert_called_once_with(
            base_url="http://127.0.0.1:8000/v1",
            model="Qwen3-14B-Q4_K_M",
            api_key="local-vllm",
            api_key_env=None,
            timeout_seconds=90.0,
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            stop=["</s>"],
        )
        self.assertIs(result, backend_cls.return_value)

    def test_build_llm_backend_loads_config_before_building(self) -> None:
        config = {"llm": {"active_backend": "llamacpp", "llamacpp": {"model_path": "m.gguf"}}}

        with patch("src.llm.factory.load_model_config", return_value=config) as load_config:
            with patch("src.llm.factory.build_llm_backend_from_config") as build_from_config:
                build_from_config.return_value = object()
                result = build_llm_backend()

        load_config.assert_called_once()
        build_from_config.assert_called_once_with(config)
        self.assertIs(result, build_from_config.return_value)

    def test_build_llm_backend_from_config_rejects_unsupported_backend(self) -> None:
        config = {"llm": {"active_backend": "local"}}

        with self.assertRaises(ValueError):
            build_llm_backend_from_config(config)

    def test_build_llm_backend_from_config_requires_llm_mapping(self) -> None:
        with self.assertRaises(ValueError):
            build_llm_backend_from_config({})


if __name__ == "__main__":
    unittest.main()

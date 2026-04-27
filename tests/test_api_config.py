from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

from src.api.config import (
    ApiConfig,
    build_api_config_from_mapping,
    load_api_config,
)


class ApiConfigTests(unittest.TestCase):
    def test_load_api_config_reads_yaml_file(self) -> None:
        tmp_root = Path(".tmp_test") / f"api-config-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            config_path = tmp_root / "api.yaml"
            config_path.write_text(
                (
                    "api:\n"
                    "  title: Test API\n"
                    "  server:\n"
                    "    host: 0.0.0.0\n"
                    "    port: 9000\n"
                    "  runtime:\n"
                    "    retrieval_config_path: configs/test-retrieval.yaml\n"
                ),
                encoding="utf-8",
            )

            config = load_api_config(config_path)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(config.title, "Test API")
        self.assertEqual(config.server.host, "0.0.0.0")
        self.assertEqual(config.server.port, 9000)
        self.assertEqual(
            config.runtime.retrieval_config_path,
            Path("configs/test-retrieval.yaml"),
        )

    def test_build_api_config_uses_defaults_for_optional_sections(self) -> None:
        config = build_api_config_from_mapping({"api": {"title": "Minimal API"}})

        self.assertIsInstance(config, ApiConfig)
        self.assertEqual(config.title, "Minimal API")
        self.assertEqual(config.server.port, 8000)
        self.assertEqual(config.cors.allow_methods, ["GET", "POST"])
        self.assertEqual(config.request.timeout_seconds, 120)
        self.assertEqual(config.runtime.model_config_path, Path("configs/model.yaml"))

    def test_build_api_config_rejects_invalid_port(self) -> None:
        with self.assertRaises(ValueError):
            build_api_config_from_mapping({"api": {"server": {"port": 70000}}})

    def test_build_api_config_rejects_invalid_cors_origins(self) -> None:
        with self.assertRaises(ValueError):
            build_api_config_from_mapping({
                "api": {
                    "cors": {
                        "allow_origins": ["http://localhost:3000", ""],
                    }
                }
            })


if __name__ == "__main__":
    unittest.main()

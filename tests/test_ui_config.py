from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

from src.ui.config import load_ui_config


class UiConfigTests(unittest.TestCase):
    def test_load_ui_config_reads_shared_api_yaml(self) -> None:
        tmp_root = Path(".tmp_test") / f"ui-config-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            config_path = tmp_root / "api.yaml"
            config_path.write_text(
                (
                    "api:\n"
                    "  request:\n"
                    "    timeout_seconds: 15\n"
                    "  ui:\n"
                    "    host: 0.0.0.0\n"
                    "    port: 7861\n"
                    "    share: true\n"
                    "    inbrowser: false\n"
                    "    api_base_url: http://127.0.0.1:9000\n"
                ),
                encoding="utf-8",
            )

            config = load_ui_config(config_path)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 7861)
        self.assertTrue(config.share)
        self.assertFalse(config.inbrowser)
        self.assertEqual(config.api_base_url, "http://127.0.0.1:9000")
        self.assertEqual(config.request_timeout_seconds, 15)


if __name__ == "__main__":
    unittest.main()

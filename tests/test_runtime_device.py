from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.runtime.device import require_cuda_device


class RuntimeDeviceTests(unittest.TestCase):
    def test_requires_installed_torch(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "PyTorch is not installed"):
                require_cuda_device()

    def test_rejects_cpu_only_torch_build(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda=None),
        )

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "not compiled with CUDA"):
                require_cuda_device()

    def test_rejects_missing_visible_cuda_gpu(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda="12.1"),
        )

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "No CUDA-capable GPU"):
                require_cuda_device()

    def test_returns_cuda_when_available(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            version=SimpleNamespace(cuda="12.1"),
        )

        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(require_cuda_device(), "cuda")


if __name__ == "__main__":
    unittest.main()

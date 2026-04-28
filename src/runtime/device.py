from __future__ import annotations

CUDA_DEVICE = "cuda"


def require_cuda_device() -> str:
    """Return the only supported torch device, or fail with a clear error."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA is required, but PyTorch is not installed in this environment."
        ) from exc

    if not torch.cuda.is_available():
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        build_detail = (
            "Installed PyTorch was not compiled with CUDA support."
            if cuda_version is None
            else "No CUDA-capable GPU is visible to PyTorch."
        )
        raise RuntimeError(f"CUDA is required for this project. {build_detail}")

    return CUDA_DEVICE

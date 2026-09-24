#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from large_kernel_eroder import LargeKernelEroder
from large_kernel_gaussian_blur import LargeKernelGaussianBlur
from triton_lanczos_resize import lanczos3_resize_bchw


def compare_native_lanczos(image: torch.Tensor, size: tuple[int, int]) -> None:
    """Compare against the PyTorch 2.13 CPU oracle when the mode is available."""
    try:
        reference = F.interpolate(
            image.float().cpu(),
            size=size,
            mode="lanczos",
            align_corners=False,
            antialias=True,
        )
    except (TypeError, ValueError, NotImplementedError) as error:
        print(f"native Lanczos comparison skipped: {error}")
        return

    candidate = lanczos3_resize_bchw(
        image, size, antialias=True, border_mode="edge", tmp_dtype="float32"
    ).float().cpu()
    error = (candidate - reference).abs()
    print(
        "native Lanczos comparison: "
        f"max_abs={error.max().item():.6g}, mean_abs={error.mean().item():.6g}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    mask = torch.zeros((1, 128, 160), device="cuda", dtype=torch.float32)
    mask[:, 16:-16, 16:-16] = 1
    eroded = LargeKernelEroder(max_kernel_per_pass=13).erode(
        mask, 31, device=mask.device, kernel_shape="rect"
    )
    assert eroded.shape == mask.shape and torch.isfinite(eroded).all()

    blurred = LargeKernelGaussianBlur(split_kernel_threshold=31, num_passes=2).blur(mask, 61)
    assert blurred.shape == mask.shape and torch.isfinite(blurred).all()

    image = torch.rand((1, 3, 64, 80), device="cuda", dtype=torch.float16)
    resized = lanczos3_resize_bchw(image, (37, 53), antialias=True, border_mode="reflect")
    assert resized.shape == (1, 3, 37, 53) and torch.isfinite(resized).all()
    compare_native_lanczos(image, (37, 53))
    print("test_gpu_image_ops: OK")


if __name__ == "__main__":
    main()

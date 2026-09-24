from __future__ import annotations

import math

import torch
from kornia.filters import gaussian_blur2d


def _opencv_auto_sigma(kernel_size: int) -> float:
    return 0.3 * (((kernel_size - 1) * 0.5) - 1) + 0.8


class LargeKernelGaussianBlur:
    """GPU-friendly Gaussian blur helper for very large kernels.

    For kernels above `split_kernel_threshold`, the blur is approximated by
    repeated smaller Gaussian blurs using sigma composition:
    sigma_per_pass = sigma_total / sqrt(num_passes).
    """

    def __init__(
        self,
        *,
        split_kernel_threshold: int = 257,
        num_passes: int = 2,
    ):
        self.split_kernel_threshold = max(3, int(split_kernel_threshold))
        self.num_passes = max(1, int(num_passes))

    @staticmethod
    def _odd(value: int) -> int:
        value = max(3, int(value))
        return value if value % 2 == 1 else value + 1

    @classmethod
    def _kernel_size_from_sigma(cls, sigma: float) -> int:
        return cls._odd(round((float(sigma) - 0.35) / 0.15))

    @classmethod
    def _clamp_reflect_kernel_size(cls, kernel_size: int, height: int, width: int) -> tuple[int, int]:
        max_kernel_h = max(1, 2 * int(height) - 1)
        max_kernel_w = max(1, 2 * int(width) - 1)
        kernel_h = min(int(kernel_size), max_kernel_h)
        kernel_w = min(int(kernel_size), max_kernel_w)
        if kernel_h % 2 == 0:
            kernel_h = max(1, kernel_h - 1)
        if kernel_w % 2 == 0:
            kernel_w = max(1, kernel_w - 1)
        return kernel_h, kernel_w

    def _single_pass(self, image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        kernel_h, kernel_w = self._clamp_reflect_kernel_size(kernel_size, image.shape[-2], image.shape[-1])
        if kernel_h == 1 and kernel_w == 1:
            return image
        blurred = gaussian_blur2d(
            image.unsqueeze(0),
            (kernel_h, kernel_w),
            (sigma, sigma),
            border_type="reflect",
            separable=True,
        )
        return blurred.squeeze(0)

    def blur(
        self,
        image: torch.Tensor,
        kernel_size: int,
        sigma: float | None = None,
        *,
        split_kernel_threshold: int | None = None,
        num_passes: int | None = None,
    ) -> torch.Tensor:
        sigma_total = _opencv_auto_sigma(kernel_size) if sigma is None else float(sigma)
        split_threshold = self.split_kernel_threshold if split_kernel_threshold is None else int(split_kernel_threshold)
        passes = self.num_passes if num_passes is None else max(1, int(num_passes))

        if kernel_size < split_threshold or passes <= 1:
            return self._single_pass(image, kernel_size, sigma_total)

        sigma_per_pass = sigma_total / math.sqrt(passes)
        kernel_per_pass = self._kernel_size_from_sigma(sigma_per_pass)
        out = image
        for _ in range(passes):
            out = self._single_pass(out, kernel_per_pass, sigma_per_pass)
        return out

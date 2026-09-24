import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.morphology import erosion as kornia_erosion


def _kornia_kernel(kernel_size: int, device: str | torch.device, shape: str) -> torch.Tensor:
    if shape == "rect":
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    elif shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported kernel shape: {shape}")
    return torch.from_numpy(kernel).to(device=device, dtype=torch.float32)


class LargeKernelEroder:
    """GPU-friendly morphology helper.

    Large flat kernels are split by radius addition so each pass stays under
    `max_kernel_per_pass`. This is exact for `rect` kernels and a close
    discrete approximation for OpenCV-style `ellipse` kernels.
    """

    def __init__(
        self,
        *,
        max_kernel_per_pass: int = 13,
        rect_direct_side: int = 3072,
        rect_tile_hw: tuple[int, int] = (2048, 2048),
        ellipse_direct_side: int = 384,
        ellipse_tile_hw: tuple[int, int] = (384, 384),
        ellipse_engine: str = "convolution",
    ):
        self.max_kernel_per_pass = max_kernel_per_pass
        self.rect_direct_side = rect_direct_side
        self.rect_tile_hw = rect_tile_hw
        self.ellipse_direct_side = ellipse_direct_side
        self.ellipse_tile_hw = ellipse_tile_hw
        self.ellipse_engine = ellipse_engine

    def split_odd_kernel_by_radius(self, kernel_size: int, max_kernel_per_pass: int | None = None) -> list[int]:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        if max_kernel_per_pass is None:
            max_kernel_per_pass = self.max_kernel_per_pass
        if max_kernel_per_pass is None:
            return [kernel_size]

        max_kernel_per_pass = max(3, int(max_kernel_per_pass))
        if max_kernel_per_pass % 2 == 0:
            max_kernel_per_pass -= 1

        if kernel_size <= max_kernel_per_pass:
            return [kernel_size]

        remaining_radius = (kernel_size - 1) // 2
        max_radius = (max_kernel_per_pass - 1) // 2
        kernel_sizes: list[int] = []

        while remaining_radius > 0:
            pass_radius = min(max_radius, remaining_radius)
            kernel_sizes.append(pass_radius * 2 + 1)
            remaining_radius -= pass_radius

        return kernel_sizes

    def _apply_tiled_chw(
        self,
        tensor: torch.Tensor,
        pad: int,
        core_tile_h: int,
        core_tile_w: int,
        op,
    ) -> torch.Tensor:
        _, h, w = tensor.shape
        tiled = torch.empty_like(tensor)

        for y in range(0, h, core_tile_h):
            core_y1 = min(y + core_tile_h, h)
            y0 = max(0, y - pad)
            y1 = min(h, core_y1 + pad)
            local_y0 = y - y0
            local_y1 = local_y0 + (core_y1 - y)

            for x in range(0, w, core_tile_w):
                core_x1 = min(x + core_tile_w, w)
                x0 = max(0, x - pad)
                x1 = min(w, core_x1 + pad)
                local_x0 = x - x0
                local_x1 = local_x0 + (core_x1 - x)

                tile = tensor[:, y0:y1, x0:x1]
                tile_out = op(tile)
                tiled[:, y:core_y1, x:core_x1] = tile_out[:, local_y0:local_y1, local_x0:local_x1]

        return tiled

    def erode_single_pass(
        self,
        mask: torch.Tensor,
        kernel_size: int,
        device: str | torch.device,
        kernel_shape: str,
    ) -> torch.Tensor:
        pad = kernel_size // 2
        _, h, w = mask.shape

        if kernel_shape == "rect":
            def rect_op(tile: torch.Tensor) -> torch.Tensor:
                return -F.max_pool2d(
                    (-tile).unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=pad,
                ).squeeze(0)

            if max(h, w) <= self.rect_direct_side:
                return rect_op(mask)
            return self._apply_tiled_chw(mask, pad, *self.rect_tile_hw, op=rect_op)

        kernel = _kornia_kernel(kernel_size, device, kernel_shape).to(dtype=mask.dtype)

        def ellipse_op(tile: torch.Tensor) -> torch.Tensor:
            return kornia_erosion(tile.unsqueeze(0), kernel, engine=self.ellipse_engine).squeeze(0)

        if max(h, w) <= self.ellipse_direct_side:
            return ellipse_op(mask)
        return self._apply_tiled_chw(mask, pad, *self.ellipse_tile_hw, op=ellipse_op)

    def erode(
        self,
        mask: torch.Tensor,
        kernel_size: int,
        device: str | torch.device,
        kernel_shape: str = "ellipse",
        max_kernel_per_pass: int | None = None,
    ) -> torch.Tensor:
        kernel_sizes = self.split_odd_kernel_by_radius(
            kernel_size,
            max_kernel_per_pass=max_kernel_per_pass,
        )
        eroded = mask
        for pass_kernel_size in kernel_sizes:
            eroded = self.erode_single_pass(
                eroded,
                pass_kernel_size,
                device=device,
                kernel_shape=kernel_shape,
            )
        return eroded

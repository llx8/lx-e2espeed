"""Standalone Triton Lanczos3 resize for CUDA BCHW tensors.

This is intentionally narrow:
  * PyTorch CUDA tensors only
  * BCHW layout
  * float16/float32 input
  * 2D Lanczos3, separable horizontal + vertical passes

The coordinate transform uses the common half-pixel convention:
  src = (dst + 0.5) * (in_size / out_size) - 0.5
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _sinc(x):
    z = x * 3.141592653589793
    az = tl.abs(z)
    safe_z = tl.where(az < 1.0e-5, 1.0, z)
    y = tl.sin(z) / safe_z
    taylor = 1.0 - z * z * 0.16666666666666666
    return tl.where(az < 1.0e-5, taylor, y)


@triton.jit
def _lanczos3(x):
    ax = tl.abs(x)
    return tl.where(ax < 3.0, _sinc(x) * _sinc(x * 0.3333333333333333), 0.0)


@triton.jit
def _map_index(idx, size: tl.constexpr, border_mode: tl.constexpr):
    if border_mode == 0:
        return tl.minimum(tl.maximum(idx, 0), size - 1)
    if border_mode == 1:
        if size <= 1:
            return idx * 0
        period = 2 * size - 2
        wrapped = tl.abs(idx) % period
        return tl.where(wrapped >= size, period - wrapped, wrapped)
    return tl.minimum(tl.maximum(idx, 0), size - 1)


@triton.jit
def _horizontal_lanczos3_kernel(
    x,
    tmp,
    rows: tl.constexpr,
    in_w: tl.constexpr,
    out_w: tl.constexpr,
    scale_w: tl.constexpr,
    filter_scale_w: tl.constexpr,
    radius_w: tl.constexpr,
    support_w: tl.constexpr,
    border_mode: tl.constexpr,
    block_w: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    ow = block * block_w + tl.arange(0, block_w)
    mask = (row < rows) & (ow < out_w)

    center = (ow.to(tl.float32) + 0.5) * scale_w - 0.5
    start = tl.ceil(center - radius_w).to(tl.int32)

    acc = tl.zeros((block_w,), tl.float32)
    norm = tl.zeros((block_w,), tl.float32)

    for k in range(support_w):
        ix = start + k
        dist = ix.to(tl.float32) - center
        w = _lanczos3(dist / filter_scale_w)
        w = tl.where(tl.abs(dist) < radius_w, w, 0.0)
        ix_mapped = _map_index(ix, in_w, border_mode)
        v = tl.load(x + row * in_w + ix_mapped, mask=mask, other=0.0).to(tl.float32)
        acc += v * w
        norm += w

    y = acc / tl.where(norm != 0.0, norm, 1.0)
    tl.store(tmp + row * out_w + ow, y, mask=mask)


@triton.jit
def _vertical_lanczos3_kernel(
    tmp,
    y,
    total: tl.constexpr,
    channels_h: tl.constexpr,
    in_h: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    scale_h: tl.constexpr,
    filter_scale_h: tl.constexpr,
    radius_h: tl.constexpr,
    support_h: tl.constexpr,
    border_mode: tl.constexpr,
    do_clamp: tl.constexpr,
    clamp_lo: tl.constexpr,
    clamp_hi: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < total

    ow = offs % out_w
    oh = (offs // out_w) % out_h
    bc = offs // (out_h * out_w)

    center = (oh.to(tl.float32) + 0.5) * scale_h - 0.5
    start = tl.ceil(center - radius_h).to(tl.int32)

    acc = tl.zeros((block,), tl.float32)
    norm = tl.zeros((block,), tl.float32)

    for k in range(support_h):
        ih = start + k
        dist = ih.to(tl.float32) - center
        w = _lanczos3(dist / filter_scale_h)
        w = tl.where(tl.abs(dist) < radius_h, w, 0.0)
        ih_mapped = _map_index(ih, in_h, border_mode)
        tmp_offs = (bc * in_h + ih_mapped) * out_w + ow
        v = tl.load(tmp + tmp_offs, mask=mask, other=0.0).to(tl.float32)
        acc += v * w
        norm += w

    out = acc / tl.where(norm != 0.0, norm, 1.0)
    if do_clamp:
        out = tl.minimum(tl.maximum(out, clamp_lo), clamp_hi)
    tl.store(y + offs, out, mask=mask)


def _filter_params(in_size: int, out_size: int, antialias: bool) -> Tuple[float, float, int]:
    scale = in_size / out_size
    filter_scale = max(scale, 1.0) if antialias else 1.0
    radius = 3.0 * filter_scale
    support = int(math.ceil(2.0 * radius))
    return filter_scale, radius, support


def lanczos3_resize_bchw(
    x: torch.Tensor,
    size: Tuple[int, int],
    *,
    antialias: bool = True,
    border_mode: Literal["edge", "reflect"] = "edge",
    tmp_dtype: Literal["auto", "fp16", "fp32"] = "auto",
    clamp: bool = False,
    block_w: int = 128,
    block: int = 256,
) -> torch.Tensor:
    """Resize a BCHW CUDA tensor with Lanczos3.

    Args:
        x: CUDA tensor with shape ``[B, C, H, W]`` and dtype fp16 or fp32.
        size: ``(out_h, out_w)``.
        antialias: widen the filter footprint when downscaling.
        border_mode: coordinate handling outside the input extent.
            ``"edge"`` clamps to the nearest border pixel.
            ``"reflect"`` mirrors indices without repeating edge pixels.
        tmp_dtype: intermediate buffer precision. ``"auto"`` uses fp16 for fp16 input
            and fp32 for fp32 input.
        clamp: clamp the output to ``[0, 1]`` in-place after resize.
        block_w: Triton horizontal tile width.
        block: Triton vertical tile size.

    Returns:
        A contiguous tensor with shape ``[B, C, out_h, out_w]`` and the same dtype as ``x``.
    """
    if x.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("expected a CUDA tensor")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected float16 or float32 input, got {x.dtype}")
    if border_mode not in ("edge", "reflect"):
        raise ValueError(f"unsupported border_mode {border_mode!r}")
    if tmp_dtype not in ("auto", "fp16", "fp32"):
        raise ValueError(f"unsupported tmp_dtype {tmp_dtype!r}")

    out_h, out_w = map(int, size)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"output size must be positive, got {size}")

    x = x.contiguous()
    b, c, in_h, in_w = map(int, x.shape)
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"input spatial size must be positive, got {(in_h, in_w)}")
    border_mode_id = 0 if border_mode == "edge" else 1
    if tmp_dtype == "auto":
        tmp_torch_dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
    elif tmp_dtype == "fp16":
        tmp_torch_dtype = torch.float16
    else:
        tmp_torch_dtype = torch.float32

    y = torch.empty((b, c, out_h, out_w), device=x.device, dtype=x.dtype)
    tmp = torch.empty((b * c * in_h, out_w), device=x.device, dtype=tmp_torch_dtype)

    scale_w = in_w / out_w
    scale_h = in_h / out_h
    filter_scale_w, radius_w, support_w = _filter_params(in_w, out_w, antialias)
    filter_scale_h, radius_h, support_h = _filter_params(in_h, out_h, antialias)

    rows = b * c * in_h
    grid_h = (rows, triton.cdiv(out_w, block_w))
    _horizontal_lanczos3_kernel[grid_h](
        x,
        tmp,
        rows,
        in_w,
        out_w,
        scale_w,
        filter_scale_w,
        radius_w,
        support_w,
        border_mode_id,
        block_w,
        num_warps=4,
    )

    total = b * c * out_h * out_w
    grid_v = (triton.cdiv(total, block),)
    _vertical_lanczos3_kernel[grid_v](
        tmp,
        y,
        total,
        b * c,
        in_h,
        out_h,
        out_w,
        scale_h,
        filter_scale_h,
        radius_h,
        support_h,
        border_mode_id,
        clamp,
        0.0,
        1.0,
        block,
        num_warps=4,
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    inp = torch.rand((2, 3, 64, 80), device="cuda", dtype=torch.float16)
    out = lanczos3_resize_bchw(inp, (37, 53), border_mode="reflect", tmp_dtype="auto", clamp=True)
    print(out.shape, out.dtype, float(out.min()), float(out.max()))

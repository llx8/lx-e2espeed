"""Bit-exact `cv2.warpAffine` (INTER_LINEAR, BORDER_CONSTANT) for uint8 HWC images on the GPU, in Triton.

OpenCV's 8U warpAffine is not a float bilinear, and its arithmetic changed between majors; replicate the OpenCV
of the production image, not the local one (`opencv_major`):
- 4.x, fixed point (imgwarp.cpp WarpAffineInvoker + remapBilinear): the inverse matrix in double; per output
  pixel X = (rint((M1 y + M2) 1024) + 16 + rint(M0 x 1024)) >> 5, taps at X >> 5 saturated to int16, fraction
  fx = X & 31, 15-bit weights (32 - fy)(32 - fx) 32 and so on, taps outside the image read the border value,
  result (sum + 16384) >> 15 saturated to uint8.
- 5.x, float (warp_kernels.simd.hpp): the inverse matrix in double, then truncated to float32. A row runs in
  SIMD steps of `simd_step` pixels (16 on the x86-64 wheels) and a scalar tail for the last dst_w % simd_step
  columns, and the two round the coordinate differently: the body computes mx = y M1 + M2 once per row (two
  roundings) and sx = fma(M0, x, mx); the tail, as the compiler contracted it, sx = fma(x, M0, y M1) + M2. Then
  ix = floor(sx), a = sx - ix, lerps fma(a, f01 - f00, f00) along x and the same along y, rint half to even,
  saturate.
Every rounding above is separate, so both kernels launch with `enable_fp_fusion=False` and spell out each fma.
Checked byte for byte by scripts/tests/test_opencv_warp_affine.py against cv2 4.10.0 and cv2 5.0.0 (every size,
channel count, border, and matrix tried; the 5.0.0 wheel stepped 16 pixels on an AVX-512 CPU too).

Two outputs from the same kernels: `warp_affine_u8` writes the uint8 HWC image cv2 would, `warp_affine_lut`
fuses what follows in a preprocessing chain, the channel reversal (BGR to RGB), HWC to CHW, and the normalize as
a (C, 256) lookup from `make_lut`, and writes the engine's input planes. Sizes, strides, and pointers are not
specialized, so one compiled variant serves every request shape; warm each (opencv_major, channels, output mode)
used once at start-up with a synthetic input. A fixed output width may be made a constexpr for vector stores.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 256  # output pixels per program along x
CV5_SIMD_STEP = 16  # pixels per SIMD step of OpenCV 5's warp: two v_float32 vectors; the x86-64 wheels use 8 lanes

__all__ = ["BLOCK", "CV5_SIMD_STEP", "cv_inverse_affine", "make_lut", "pin_libcuda_path", "warp_affine_lut",
           "warp_affine_u8"]


def pin_libcuda_path() -> None:
    """Point Triton at the libcuda this process already loaded, so it skips `ldconfig -p`, which segfaults on
    glibc 2.32 once LD_LIBRARY_PATH reaches 128 characters (cv2 prepends to it at every import)."""
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.rsplit(" ", 1)[-1].strip()
                if os.path.basename(path).startswith("libcuda.so"):
                    os.environ["TRITON_LIBCUDA_PATH"] = os.path.dirname(path)
                    return
    except OSError:
        pass


def cv_inverse_affine(m, opencv_major: int = 4, inverse_map: bool = False) -> list[float]:
    """The six entries cv::warpAffine computes from the 2x3 matrix M: inverted in double unless WARP_INVERSE_MAP
    (`inverse_map`), kept in double for 4.x and truncated to float32 for 5.x."""
    M = [float(v) for v in np.asarray(m, dtype=np.float64)[:2].reshape(-1)]
    if not inverse_map:
        D = M[0] * M[4] - M[1] * M[3]
        D = 1.0 / D if D != 0 else 0.0
        A11, A22 = M[4] * D, M[0] * D
        M[0], M[1], M[3], M[4] = A11, M[1] * -D, M[3] * -D, A22
        M[2], M[5] = -M[0] * M[2] - M[1] * M[5], -M[3] * M[2] - M[4] * M[5]
    if opencv_major >= 5:
        M = [float(v) for v in np.asarray(M, dtype=np.float32)]
    return M


@triton.jit
def _store(out, lut, r, y, xs, xmask, o_sy, plane, c: tl.constexpr, C: tl.constexpr, LUT: tl.constexpr,
           REVERSE: tl.constexpr):
    """r: the warped uint8 values of channel c as int32. LUT: plane p of the CHW output gets lut[p, r], where p is
    c or, with REVERSE, C - 1 - c. Otherwise channel c of the HWC uint8 output."""
    row = y.to(tl.int64) * o_sy
    if LUT:
        if REVERSE:
            tl.store(out + (C - 1 - c) * plane + row + xs, tl.load(lut + (C - 1 - c) * 256 + r), mask=xmask)
        else:
            tl.store(out + c * plane + row + xs, tl.load(lut + c * 256 + r), mask=xmask)
    else:
        tl.store(out + row + xs * C + c, r.to(tl.uint8), mask=xmask)


@triton.jit(do_not_specialize=["img", "i_sy", "i_sx", "H", "W", "out", "o_sy", "plane", "dst_w", "lut"])
def _warp_fixed_kernel(img, i_sy, i_sx, H, W, out, o_sy, plane, dst_w, lut, border,
                       m0: tl.float64, m1: tl.float64, m2: tl.float64,
                       m3: tl.float64, m4: tl.float64, m5: tl.float64,
                       C: tl.constexpr, BLOCK: tl.constexpr, LUT: tl.constexpr, REVERSE: tl.constexpr):
    """OpenCV 4.x: the fixed-point coordinate tables and remapBilinear's integer taps."""
    y = tl.program_id(0)
    xs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    xmask = xs < dst_w
    yd = y.to(tl.float64)
    xd = xs.to(tl.float64)
    x0 = libdevice.rint((yd * m1 + m2) * 1024.0).to(tl.int32) + 16
    y0 = libdevice.rint((yd * m4 + m5) * 1024.0).to(tl.int32) + 16
    X = (x0 + libdevice.rint(m0 * xd * 1024.0).to(tl.int32)) >> 5
    Y = (y0 + libdevice.rint(m3 * xd * 1024.0).to(tl.int32)) >> 5
    ix = tl.minimum(tl.maximum(X >> 5, -32768), 32767)
    iy = tl.minimum(tl.maximum(Y >> 5, -32768), 32767)
    fx = X & 31
    fy = Y & 31
    w00 = (32 - fy) * (32 - fx) * 32
    w01 = (32 - fy) * fx * 32
    w10 = fy * (32 - fx) * 32
    w11 = fy * fx * 32
    in_x0 = (ix >= 0) & (ix < W)
    in_x1 = (ix + 1 >= 0) & (ix + 1 < W)
    in_y0 = (iy >= 0) & (iy < H)
    in_y1 = (iy + 1 >= 0) & (iy + 1 < H)
    m00 = in_y0 & in_x0
    m01 = in_y0 & in_x1
    m10 = in_y1 & in_x0
    m11 = in_y1 & in_x1
    oi = iy.to(tl.int64) * i_sy + ix.to(tl.int64) * i_sx
    for c in tl.static_range(C):
        bv = tl.load(border + c)
        base = img + oi + c
        f00 = tl.where(m00, tl.load(base, mask=m00, other=0).to(tl.int32), bv)
        f01 = tl.where(m01, tl.load(base + i_sx, mask=m01, other=0).to(tl.int32), bv)
        f10 = tl.where(m10, tl.load(base + i_sy, mask=m10, other=0).to(tl.int32), bv)
        f11 = tl.where(m11, tl.load(base + i_sy + i_sx, mask=m11, other=0).to(tl.int32), bv)
        v = (f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11 + 16384) >> 15
        r = tl.minimum(tl.maximum(v, 0), 255)
        _store(out, lut, r, y, xs, xmask, o_sy, plane, c, C, LUT, REVERSE)


@triton.jit(do_not_specialize=["img", "i_sy", "i_sx", "H", "W", "out", "o_sy", "plane", "dst_w", "tail", "lut"])
def _warp_float_kernel(img, i_sy, i_sx, H, W, out, o_sy, plane, dst_w, tail, lut, border,
                       m0, m1, m2, m3, m4, m5,
                       C: tl.constexpr, BLOCK: tl.constexpr, LUT: tl.constexpr, REVERSE: tl.constexpr):
    """OpenCV 5.x in float32 (the matrix entries arrive as fp32). Columns before `tail` follow the SIMD body: per
    row mx = y M1 + M2, per pixel sx = fma(M0, x, mx). Columns from `tail` on follow the scalar tail as the
    compiler contracted it: sx = fma(x, M0, y M1) + M2. Both lerp with fma."""
    y = tl.program_id(0)
    xs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    xmask = xs < dst_w
    scalar = xs >= tail
    yf = y.to(tl.float32)
    xf = xs.to(tl.float32)
    z = tl.zeros_like(xf)
    mx = yf * m1 + m2
    my = yf * m4 + m5
    sx = tl.where(scalar, tl.fma(xf, m0 + z, yf * m1 + z) + m2, tl.fma(m0 + z, xf, mx + z))
    sy = tl.where(scalar, tl.fma(xf, m3 + z, yf * m4 + z) + m5, tl.fma(m3 + z, xf, my + z))
    fx = tl.floor(sx)
    fy = tl.floor(sy)
    ix = fx.to(tl.int32)
    iy = fy.to(tl.int32)
    a = sx - fx
    b = sy - fy
    in_x0 = (ix >= 0) & (ix < W)
    in_x1 = (ix + 1 >= 0) & (ix + 1 < W)
    in_y0 = (iy >= 0) & (iy < H)
    in_y1 = (iy + 1 >= 0) & (iy + 1 < H)
    m00 = in_y0 & in_x0
    m01 = in_y0 & in_x1
    m10 = in_y1 & in_x0
    m11 = in_y1 & in_x1
    oi = iy.to(tl.int64) * i_sy + ix.to(tl.int64) * i_sx
    for c in tl.static_range(C):
        bv = tl.load(border + c).to(tl.float32)
        base = img + oi + c
        f00 = tl.where(m00, tl.load(base, mask=m00, other=0).to(tl.float32), bv)
        f01 = tl.where(m01, tl.load(base + i_sx, mask=m01, other=0).to(tl.float32), bv)
        f10 = tl.where(m10, tl.load(base + i_sy, mask=m10, other=0).to(tl.float32), bv)
        f11 = tl.where(m11, tl.load(base + i_sy + i_sx, mask=m11, other=0).to(tl.float32), bv)
        top = tl.fma(a, f01 - f00, f00)
        bot = tl.fma(a, f11 - f10, f10)
        v = tl.fma(b, bot - top, top)
        r = tl.minimum(tl.maximum(libdevice.rint(v), 0.0), 255.0).to(tl.int32)
        _store(out, lut, r, y, xs, xmask, o_sy, plane, c, C, LUT, REVERSE)


_BORDERS: dict[tuple, torch.Tensor] = {}


def _border_tensor(border_value, device: torch.device) -> torch.Tensor:
    """cv2's borderValue as a cached (4,) int32 tensor, with cv2's Scalar semantics: a number sets channel 0 only
    (the usual surprise), a sequence sets its channels and the missing ones are 0; each value rounded half to even
    and saturated like saturate_cast<uchar>."""
    values = (border_value,) if np.isscalar(border_value) else tuple(border_value)
    values = tuple(int(min(max(np.rint(float(v)), 0), 255)) for v in values)[:4]
    values = values + (0,) * (4 - len(values))
    key = (values, device.index)
    if key not in _BORDERS:
        _BORDERS[key] = torch.tensor(values, dtype=torch.int32, device=device)
    return _BORDERS[key]


def _check_image(img: torch.Tensor) -> tuple[int, int, int]:
    if not (img.is_cuda and img.dtype == torch.uint8 and img.dim() == 3 and img.stride(2) == 1):
        raise ValueError("img must be a CUDA uint8 (H, W, C) tensor whose channels are contiguous")
    H, W, C = img.shape
    if not 1 <= C <= 4:
        raise ValueError(f"1 to 4 channels, got {C}")
    return H, W, C


def _launch(img, m, out, lut, border_value, dst_w, dst_h, o_sy, plane, opencv_major, inverse_map, reverse, block,
            simd_step):
    """lut None: uint8 HWC output; else the fused CHW lookup output."""
    H, W, C = _check_image(img)
    if opencv_major not in (4, 5):
        raise ValueError("opencv_major is 4 (fixed point) or 5 (float)")
    pin_libcuda_path()
    grid = (dst_h, triton.cdiv(dst_w, block))
    head = (img, img.stride(0), img.stride(1), H, W, out, o_sy, plane, dst_w)
    rest = (out if lut is None else lut, _border_tensor(border_value, img.device),
            *cv_inverse_affine(m, opencv_major, inverse_map))
    opts = dict(C=C, BLOCK=block, LUT=lut is not None, REVERSE=reverse, enable_fp_fusion=False)
    if opencv_major == 4:
        _warp_fixed_kernel[grid](*head, *rest, **opts)
    else:
        _warp_float_kernel[grid](*head, dst_w - dst_w % simd_step, *rest, **opts)


def warp_affine_u8(img: torch.Tensor, m, dsize, border_value=0, opencv_major: int = 4, inverse_map: bool = False,
                   out: torch.Tensor | None = None, block: int = BLOCK, simd_step: int = CV5_SIMD_STEP) -> torch.Tensor:
    """cv2.warpAffine(img, m, dsize, INTER_LINEAR, BORDER_CONSTANT, border_value) on the GPU, launched on torch's
    current stream. img: (H, W, C) uint8 CUDA, C 1 to 4, channels contiguous (a crop view is fine); m: the 2x3
    matrix as cv2 takes it; dsize: (width, height) as cv2; border_value as cv2 reads it (a number fills channel 0
    only; give a sequence for every channel); simd_step (5.x only): pixels per SIMD step of the production cv2,
    whose last dst_w % simd_step columns take the scalar arithmetic. Returns (height, width, C) uint8 (`out` if
    given)."""
    dst_w, dst_h = (int(v) for v in dsize)
    C = img.shape[2]
    if out is None:
        out = torch.empty((dst_h, dst_w, C), dtype=torch.uint8, device=img.device)
    if not (out.is_contiguous() and out.dtype == torch.uint8 and tuple(out.shape) == (dst_h, dst_w, C)):
        raise ValueError(f"out must be a contiguous uint8 tensor of shape {(dst_h, dst_w, C)}")
    _launch(img, m, out, None, border_value, dst_w, dst_h, dst_w * C, 0, opencv_major, inverse_map, False, block,
            simd_step)
    return out


def warp_affine_lut(img: torch.Tensor, m, dsize, lut: torch.Tensor, out: torch.Tensor | None = None,
                    reverse_channels: bool = False, border_value=0, opencv_major: int = 4,
                    inverse_map: bool = False, block: int = BLOCK, simd_step: int = CV5_SIMD_STEP) -> torch.Tensor:
    """The warp above fused with the layout change and the normalize: plane p of the (C, height, width) output is
    lut[p, warped[..., q]], q = p or, with `reverse_channels` (BGR input, RGB planes), C - 1 - p. lut: (C, 256)
    contiguous CUDA tensor, float32 or float16 (`make_lut`); out: contiguous, the dtype of lut."""
    dst_w, dst_h = (int(v) for v in dsize)
    C = img.shape[2]
    if not (lut.is_cuda and lut.is_contiguous() and tuple(lut.shape) == (C, 256)):
        raise ValueError(f"lut must be a contiguous CUDA tensor of shape {(C, 256)}")
    if out is None:
        out = torch.empty((C, dst_h, dst_w), dtype=lut.dtype, device=img.device)
    if not (out.is_contiguous() and out.dtype == lut.dtype and tuple(out.shape) == (C, dst_h, dst_w)):
        raise ValueError(f"out must be a contiguous {lut.dtype} tensor of shape {(C, dst_h, dst_w)}")
    _launch(img, m, out, lut, border_value, dst_w, dst_h, dst_w, dst_h * dst_w, opencv_major, inverse_map,
            reverse_channels, block, simd_step)
    return out


def make_lut(normalize, channels: int = 3, device="cuda") -> torch.Tensor:
    """(channels, 256) table for warp_affine_lut: row p holds the project's own `normalize` applied to plane p at
    each uint8 value. normalize maps a (1, channels, 1, 256) float32 CUDA tensor holding 0..255 along the last
    axis to a tensor of the same shape, so a lookup equals normalizing the warped uint8 image bit for bit."""
    u = torch.arange(256, dtype=torch.float32, device=device).view(1, 1, 1, 256)
    table = normalize(u.expand(1, channels, 1, 256).contiguous())
    return table.reshape(channels, 256).contiguous()

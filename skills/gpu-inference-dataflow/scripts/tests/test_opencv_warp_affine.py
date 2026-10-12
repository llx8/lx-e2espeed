#!/usr/bin/env python3
"""Byte-for-byte check of opencv_warp_affine_u8 against the cv2 installed in this environment.

cv2's major version picks the kernel under test: 4 the fixed-point kernel, 5 the float kernel; run the test once
in an environment with each OpenCV major the production images use. Every case warps a random image with
cv2.warpAffine (INTER_LINEAR, BORDER_CONSTANT) and with the GPU kernel and prints the number of differing bytes;
the fused LUT output is checked against a NumPy lookup of cv2's result. The other major's kernel runs on the
same inputs and its difference is printed, not judged. Exit status 1 when a case that must be exact is not.

Usage: python test_opencv_warp_affine.py [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from opencv_warp_affine_u8 import make_lut, warp_affine_lut, warp_affine_u8  # noqa: E402

MAJOR = int(cv2.__version__.split(".")[0])
OTHER = 5 if MAJOR == 4 else 4
SIMD_WIDTH = 16  # cv2 5.x: columns beyond the last multiple of 16 take the scalar tail


def matrices(H, W, dw, dh):
    """Name, 2x3 matrix, border value: the warps the image services use, plus shear, flips, and a random one."""
    s = min(dw / W, dh / H)
    rot = cv2.getRotationMatrix2D((W / 2, H / 2), 33.0, 0.8)
    align = cv2.getRotationMatrix2D((W * 0.4, H * 0.6), -12.5, 1.7)
    align[:, 2] += (dw / 2 - W * 0.4, dh / 2 - H * 0.6)
    return [
        ("translate", np.array([[1.0, 0.0, 3.5], [0.0, 1.0, -2.25]]), 0),
        ("rotate_scale", rot, 0),
        ("align", align, 0),
        ("letterbox", np.array([[s, 0.0, (dw - s * W) / 2], [0.0, s, (dh - s * H) / 2]]), 114),
        ("shear", np.array([[1.0, 0.3, 0.0], [0.1, 1.0, 0.0]]), (255, 0, 128)),
        ("zoom_out", np.array([[0.4, 0.0, dw * 0.3], [0.0, 0.4, dh * 0.3]]), (7, 200, 90, 255)),
        ("flip_x", np.array([[-1.0, 0.0, W - 1.0], [0.0, 1.0, 0.0]]), 0),
        ("flip_y", np.array([[1.0, 0.0, 0.0], [0.0, -1.0, H - 1.0]]), 0),
        ("downscale", np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]), 0),
    ]


def cv_warp(img, m, dsize, border, inverse_map):
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse_map else 0)
    ref = cv2.warpAffine(img, m, dsize, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return ref.reshape(dsize[1], dsize[0], img.shape[2])


def diff_report(ref, got):
    d = np.abs(ref.astype(np.int16) - got.astype(np.int16))
    n = int(np.count_nonzero(d))
    cols = np.nonzero(d.any(axis=(0, 2)))[0]
    body_cols = int(np.count_nonzero(cols < ref.shape[1] - ref.shape[1] % SIMD_WIDTH))
    return n, int(d.max()) if n else 0, body_cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rng = np.random.default_rng(args.seed)
    print(f"cv2 {cv2.__version__} (kernel under test: opencv_major={MAJOR}), torch {torch.__version__}, "
          f"{torch.cuda.get_device_name()}")

    failures = 0
    cases = 0
    other_bytes = other_total = 0
    images = [(480, 640, 3), (97, 131, 3), (256, 256, 1), (120, 200, 4), (33, 1000, 3)]
    dsizes = [None, (300, 200), (257, 129), (512, 512), (640, 48), (16, 16)]
    for H, W, C in images:
        base = torch.from_numpy(rng.integers(0, 256, (H + 40, W + 60, 4), dtype=np.uint8)).cuda()
        img_t = base[20:20 + H, 30:30 + W, :C]  # a crop view with row and pixel strides of the parent
        img = img_t.cpu().numpy().copy()
        for dsize in dsizes:
            dw, dh = dsize or (W, H)
            for name, m, border in matrices(H, W, dw, dh):
                border = border if np.isscalar(border) else border[:C] if C > 1 else border[0]
                for inverse_map in (False, True) if name == "rotate_scale" else (False,):
                    cases += 1
                    ref = cv_warp(img, m, (dw, dh), border, inverse_map)
                    got = warp_affine_u8(img_t, m, (dw, dh), border, MAJOR, inverse_map).cpu().numpy()
                    n, mx, body = diff_report(ref, got)
                    other = warp_affine_u8(img_t, m, (dw, dh), border, OTHER, inverse_map).cpu().numpy()
                    other_bytes += diff_report(ref, other)[0]
                    other_total += ref.size
                    must_be_exact = MAJOR == 4 or dw % SIMD_WIDTH == 0
                    bad = n > 0 if must_be_exact else body > 0
                    failures += bad
                    if n or dsize is None:
                        tag = "FAIL" if bad else ("ok" if n == 0 else "tail")
                        print(f"{tag:4} {H}x{W}x{C} -> {dw}x{dh} {name}{' inv' if inverse_map else ''}: "
                              f"{n} bytes differ (max {mx}, {body} body columns)")
        # fused output: lookup tables in both dtypes, both channel orders
        m, border = matrices(H, W, 320, 256)[2][1:]
        ref = cv_warp(img, m, (320, 256), border, False)
        for dtype in (torch.float32, torch.float16):
            lut = torch.from_numpy(rng.standard_normal((C, 256)).astype(np.float32)).cuda().to(dtype)
            for reverse in (False, True):
                cases += 1
                got = warp_affine_lut(img_t, m, (320, 256), lut, None, reverse, border, MAJOR)
                lut_np = lut.cpu().numpy()
                exp = np.stack([lut_np[p][ref[..., C - 1 - p if reverse else p]] for p in range(C)])
                exact = np.array_equal(got.cpu().numpy(), exp)
                failures += not exact
                print(f"{'ok' if exact else 'FAIL':4} {H}x{W}x{C} lut {dtype} reverse={reverse}")

    # make_lut equals the normalize it was built from
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    lut = make_lut(lambda u: (u / 255.0 - mean) / std)
    values = torch.arange(256, dtype=torch.float32, device="cuda").view(1, 1, 1, 256).expand(1, 3, 1, 256)
    assert torch.equal(lut, ((values / 255.0 - mean) / std).reshape(3, 256))

    # one timing line: the service case, 3-channel image to a 512x512 fused input
    img_t = torch.randint(0, 256, (1080, 1920, 3), dtype=torch.uint8, device="cuda")
    m = cv2.getRotationMatrix2D((960.0, 540.0), 10.0, 0.5)
    out = torch.empty((3, 512, 512), dtype=torch.float32, device="cuda")
    for _ in range(20):
        warp_affine_lut(img_t, m, (512, 512), lut, out, True, 0, MAJOR)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(200):
        warp_affine_lut(img_t, m, (512, 512), lut, out, True, 0, MAJOR)
    end.record()
    torch.cuda.synchronize()
    print(f"1920x1080x3 -> 3x512x512 fused, opencv_major={MAJOR}: {start.elapsed_time(end) / 200 * 1000:.1f} us "
          f"per launch (CUDA events, 200 launches)")

    print(f"{cases} cases, {failures} failures; the opencv_major={OTHER} kernel differs from this cv2 in "
          f"{other_bytes} of {other_total} bytes ({100 * other_bytes / max(other_total, 1):.3f}%)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

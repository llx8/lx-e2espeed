#!/usr/bin/env python3
"""Standalone numerical test for the strided multi-head d=16 INT8/FP8 attention.

Builds the plugin kernels + a torch binding via cpp_extension and compares the
fused kernel against torch SDPA on self-attn (large N) and cross-attn (N~576)
shapes, matching the ColorAdjustor IO contract: Q[B,H,Nq,16], K^T[B,H,16,Nkv],
V[B,H,Nkv,16].
"""
from __future__ import annotations

import math
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.join(os.path.dirname(HERE), "plugins", "color_attn_d16")


def build():
    return load(
        name="color_attn_d16_ext",
        sources=[
            os.path.join(PLUGIN_DIR, "color_attn_d16_torch.cpp"),
            os.path.join(PLUGIN_DIR, "color_attn_d16_kernels.cu"),
        ],
        extra_include_paths=[PLUGIN_DIR],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120"],
        extra_cflags=["-O3"],
        verbose=True,
    )


def reference(q, k, v, sm_scale, bias=None):
    scores = (q.float() @ k.float().transpose(-1, -2)) * sm_scale
    if bias is not None:
        scores = scores + bias.float()  # [H,Nq,Nkv] broadcast over batch
    probs = torch.softmax(scores, dim=-1)
    return probs @ v.float()


def metric(name, ref, got):
    diff = (ref.float() - got.float()).abs()
    flat = diff.flatten()
    p99_idx = min(max(math.ceil(0.99 * flat.numel()), 1), flat.numel())
    p99 = flat.cpu().kthvalue(p99_idx).values.item()
    finite = torch.isfinite(got).all().item()
    rel = diff.mean().item() / (ref.float().abs().mean().item() + 1e-8)
    print(
        f"{name}: max={diff.max().item():.5f} mean={diff.mean().item():.5f} "
        f"p99={p99:.5f} rel_mean={rel:.4f} finite={finite}"
    )


def run_case(ext, name, B, H, Nq, Nkv, dtype, with_bias=False):
    torch.manual_seed(0)
    q = torch.randn(B, H, Nq, 16, device="cuda", dtype=dtype)
    k = torch.randn(B, H, Nkv, 16, device="cuda", dtype=dtype)
    v = torch.randn(B, H, Nkv, 16, device="cuda", dtype=dtype)
    bias = 2.0 * torch.randn(H, Nq, Nkv, device="cuda", dtype=dtype) if with_bias else None
    sm_scale = 16 ** -0.5
    kt = k.transpose(-1, -2).contiguous()  # [B,H,16,Nkv]
    got = ext.color_attn_d16(q, kt, v, sm_scale, bias)
    ref = reference(q, k, v, sm_scale, bias)
    tag = "+bias" if with_bias else ""
    metric(f"{name}{tag} [{B},{H},{Nq},{Nkv}] {dtype}", ref, got)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    ext = build()
    cc = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} sm{cc[0]}{cc[1]}")
    for dtype in (torch.float16, torch.bfloat16):
        run_case(ext, "cross-attn", 1, 4, 256, 576, dtype)
        run_case(ext, "cross-attn", 1, 8, 1024, 576, dtype)
        run_case(ext, "self-attn", 1, 4, 4096, 4096, dtype)
        run_case(ext, "self-attn-mid", 1, 16, 2048, 2048, dtype)
        run_case(ext, "ragged", 1, 4, 100, 333, dtype)
        # OmniSR window attention: B = batched 8x8 windows, H=4, N=64, rel_pos_bias [H,64,64].
        run_case(ext, "omnisr-win", 4096, 4, 64, 64, dtype, with_bias=True)
        # Full 8-tile batch: B*H = 131072 exercises the grid.y 65535 split.
        run_case(ext, "omnisr-win-8tile", 32768, 4, 64, 64, dtype, with_bias=True)
        run_case(ext, "cross-attn", 1, 8, 1024, 576, dtype, with_bias=True)
        run_case(ext, "ragged", 1, 4, 100, 333, dtype, with_bias=True)
        # Small-window fast path (nq,nkv <= 64) ragged branches.
        run_case(ext, "small-ragged", 33, 4, 64, 48, dtype, with_bias=True)
        run_case(ext, "small-ragged", 33, 4, 50, 48, dtype)
        run_case(ext, "small-ragged", 7, 2, 64, 33, dtype, with_bias=True)


if __name__ == "__main__":
    main()

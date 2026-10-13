---
name: gpu-inference-mps-hang
description: "MPS workers all hang in cudaStreamSynchronize at 100% util and low power: TensorRT grid-barrier cause, coredump forensics, engine scanner."
---

# Multi-process MPS hang

## Symptom

Several worker processes share one GPU with MPS enabled. All of them hang at once in `cudaStreamSynchronize` / `torch.cuda.Stream.synchronize`; `nvidia-smi` shows GPU-Util 100% at very low power (about 60 W on an RTX 5090); nothing recovers until every worker is killed. A single process, or the same workers without MPS, never hangs.

## Root cause

TensorRT (seen with 10.16 on sm_120) compiles ONNX `GroupNormalization` and `InstanceNormalization`, and two-pass mean reductions written out by hand, into Myelin kernels named like `__myl_Inor*`, `__myl_Gnor*`, `__myl_ReluGnor*`, or `__myl_*MeanMean*`. These kernels contain a grid-level software barrier: thread 0 spins on a global counter while other warps wait, and the grid is sized to the whole GPU (for example 170 SMs x 2 = 340 blocks), so every block must be resident at once. Under MPS, other workers hold SMs, only part of the grid becomes resident, and the rest can never be scheduled: a deadlock. With the GPU to itself, a process always fits, so the bug never shows.

One path without the fix is enough to hang the whole machine: an engine where one network used a plugin but another kept `InstanceNormalization` still deadlocked.

## Fix

Run GroupNorm/InstanceNorm through a TensorRT plugin that reduces within a block (no grid barrier) instead of TensorRT's built-in normalization. The `gpu-inference-cnn-tensorrt` skill bundles such plugins and ONNX rewrites. Rewriting GroupNorm as InstanceNormalization does not help; those lower to the same barrier kernels.

In practice the plugin fixed the hang under 6 workers with MPS and concurrency 12 for 8 stress rounds, with unchanged latency and outputs matching the old engine (worst case 0.055% of pixels off by more than one level). It can even help: one embedder no longer needed an FP32 lock for precision and dropped from 9.4 ms to 5.1 ms.

## Before delivering a multi-worker service

Scan every engine the service loads:

```bash
python3 scripts/scan_trt_grid_barrier.py models/*.plan          # name prefilter, then disassemble matches
python3 scripts/scan_trt_grid_barrier.py --names models/*.plan  # names only, milliseconds
python3 scripts/scan_trt_grid_barrier.py --all -j 24 engine.plan  # full disassembly, to confirm the prefilter
```

The quickest check is the ONNX graph itself: any `GroupNormalization` or `InstanceNormalization` node left for Myelin is a candidate, and it points straight at the export code to change. Map a flagged kernel back to its layer with `engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)`.

## Cascade in production

A hung worker hits the dispatcher's worker timeout and is killed with SIGKILL. MPS documents that a client killed with GPU work outstanding puts the MPS server into teardown, so every restarted worker fails with `CUDA error 807 (MPS server is not ready)` and the service dies until the container restarts. See [references/forensics.md](references/forensics.md) for GPU coredump capture under MPS, dead ends to avoid, and MPS operations.

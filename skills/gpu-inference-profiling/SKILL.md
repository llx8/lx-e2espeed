---
name: gpu-inference-profiling
description: "Baseline and profile batch-1 PyTorch GPU inference: cold start vs warm, stage timing, CUDA events, torch.profiler, memory snapshots."
---

# Profile GPU Inference

Establish evidence before proposing an optimization.

## Workflow

1. Identify the production entry point and representative input matrix.
2. Draw the stage boundary from input decode through final output.
3. Measure initialization, warmup, steady-state end-to-end latency, and peak memory separately.
4. Add coarse stage timing before collecting operator-level traces.
5. Use CUDA events for asynchronous GPU regions and wall time for the complete request.
6. Use `torch.profiler` only after coarse timing identifies a region worth tracing.
7. Capture a CUDA memory snapshot when peak lifetime, fragmentation, or OOM is unclear.
8. Return a ranked bottleneck table with evidence and confidence.

## Tools

- Run `scripts/benchmark.py` as a reusable benchmark scaffold.
- Run `scripts/e2e_profiler.py` for multi-stage pipelines.
- Adapt `scripts/gpu_profiler.py` for PyTorch profiler traces.
- Route OOM and allocation-lifetime analysis to `gpu-inference-memory`, which owns `scripts/cuda_memory_snapshot.py` and the maximum-image E2E protocol.

Treat these as offline analysis tools. Do not copy their CLI and diagnostic output into a production request path.

Read [references/measurement.md](references/measurement.md) for timing boundaries and common measurement failures.

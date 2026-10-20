---
tier: on-demand
read_when: profiling shows the symptom below
---
# Small iterative solver as custom kernels

- Category: compute
- Applies to: iterative solvers with a few parameters and per-pixel or per-point terms (Levenberg-Marquardt, Gauss-Newton) after a network, run as eager PyTorch with a small or fixed step count; CUDA GPUs, Triton

## Symptom
The solver is the largest stage while its kernels take a fraction of it: thousands of tiny ops per request (about 3900 aten ops and 1300 kernel launches for 1.8 ms of kernel time in one service), host syncs for the stop decision, the GPU mostly idle.

## Technique
Rungs, each measured on the production GPU before the next:
1. Remove redundant work: residuals and costs once per estimate (the end of step i is the start of step i+1); constants built on the device (card `one-wait-per-request`); per-pixel tiny matrix products as elementwise sums (card `tiny-batched-gemm-elementwise`).
2. Replay the initial residuals and each step from CUDA graphs per input shape, one host read per step for the stop decision, one arena for every shape's fixed buffers (per-shape buffers cost +942 MiB over 61 shapes). Only with few shapes (card `cuda-graph-trt`).
3. Unroll the step count into a fixed graph (every step computed, a converged step keeps its estimate, the damped small system by closed-form Cholesky) and build it as its own FP32 TensorRT engine; the network's and the solver's contexts share one scratch buffer.
4. Write the solver as Triton kernels, one per step, with a fixed program count independent of the image size. Each program sums the previous step's per-program partial sums in a fixed order, does the scalar bookkeeping (damping, stop test, accept), solves the small system and updates the parameters (every program computes the same scalars), then writes the partial sums of costs, gradient and Hessian over its own pixels; after convergence it skips the pixel work.
   The kernel boundary is the only synchronization: no grid-wide wait, no atomics, so results are bit-identical run to run, across MPS shares, and here across the two GPUs. FP32, one launch config, no runtime autotune; compile in Init's warm-up (card `init-warm-up`).

## Effect
Per rung, A/B: on-device constants -1.60 ms; per-step CUDA graphs -10.0 ms (-38%, RTX 5090); the unrolled engine instead of the eager loop -8.4 ms (L20); Triton instead of that engine -4.6 ms (-32%, L20), the solver alone 3.36 -> 0.48 ms, 8 workers under MPS +42% requests/s. Production's solver stage 24.1 ms, Triton's 0.56 ms (L20, stage mode, different days).

## Side effects
Rung 3 fixes the solver preset at build. Rung 4: cold start +1.6 s (L20) to +2.1 s (RTX 5090), the kernels JIT-compiled in Init with an empty cache; Triton as a dependency (it ships with torch); a hand-written solver to keep equal to a reference (keep the unrolled version for checks); outputs L1 (summation order); CI no longer builds a solver engine (-148 s).

## Does not apply
- Rung 2 where input shapes follow each request and capturing every shape is not wanted: removed in one service at +11.3 ms (L20).
- Rung 3 inside the network's engine: with the FP16 flag, FP32 pins on the Hessian assembly made a Myelin node TensorRT 10.16 could not compile for sm_89; it built on sm_120. Two engines built on both.
- FP16 in the solver: its update gave no result and the Hessian assembly 45 px instead of 3 px median; a mixed build was no faster (8.30 against 8.22 ms), as the FP32 parts held 74-88% of the time.
- torch.compile of the solver's functions: -0.7 ms for +13 s cold start (card `torch-compile`).

## Evidence
- ✓ 2026-10: camera-calibration CNN-transformer service with an iterative least-squares refinement (3 parameters, 15 steps, about 150k pixels), batch 1, RTX 5090: one residuals and costs pass per estimate instead of two per step: -2.08 ms (-6.9%), L0.
- ✓ 2026-10: the same, initial residuals and each step from CUDA graphs per shape: -10.00 ms (-38.3%), PNG requests -12.96 ms, L0, +148 MiB over all 61 input shapes.
- ✓ 2026-10: the same, the unrolled solver as a second FP32 TensorRT engine instead of the eager loop without graphs: L20 23.87 -> 15.48 ms, RTX 5090 26.63 -> 12.16 ms, L1, 100 images equal.
- ✓ 2026-10: the same, Triton kernels (17 launches) instead of that engine: L20 14.57 -> 9.95 ms (-31.7%), RTX 5090 14.45 -> 9.70 ms (-32.9%), L1, 100 images 0.00 px; 8 workers under MPS 120.9 -> 172.2 requests/s (L20).
